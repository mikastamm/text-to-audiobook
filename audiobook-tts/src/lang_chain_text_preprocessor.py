#!/usr/bin/env python3
"""
Long-text speaker-annotation pipeline
✓ Thread-safe
✓ `tqdm` progress bars (per chunk)
✓ Robust retry & voice-checking
"""

from __future__ import annotations

import os
import re
import sys
import time
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

from generate_helper import listAvailableVoices
from text_utils import TextUtils

# ────────────────────────────  CLI / utils ──────────────────────────────


def colored(text: str, color: str) -> str:
    _codes = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'cyan': '\033[36m',
        'magenta': '\033[35m',
        'blue': '\033[34m',
        'gray': '\033[90m',
        'orange': '\033[38;5;208m',
        'white': '\033[97m',
    }
    reset = '\033[0m'
    return f"{_codes.get(color, '')}{text}{reset}"


print_lock = threading.Lock()


def safe_print(*args, **kwargs) -> None:
    """Thread-safe print that plays nicely with tqdm."""
    with print_lock:
        tqdm.write(' '.join(map(str, args)), **kwargs)


# ─────────────────────────  Progress Bar Helpers ────────────────────────


class MyProgressBar:
    """Per-chunk tqdm wrapper (one bar per thread/chunk)."""

    def __init__(self, total_expected_chars: int, chunk_index: int, identifier: str):
        # Using `position` keeps bars in deterministic vertical order
        self._pbar = tqdm(
            total=total_expected_chars,
            desc=f"{identifier:>10}",
            unit="char",
            position=chunk_index,
            leave=False,
            bar_format="{desc} {bar}| {n_fmt}/{total_fmt}",
        )
        self._last = 0

    def set_current(self, count: int) -> None:
        delta = count - self._last
        if delta > 0:
            self._pbar.update(delta)
            self._last = count

    def close(self) -> None:
        self._pbar.close()


class ProgressCallback(BaseCallbackHandler):
    """Streams token events into a tqdm bar."""

    def __init__(self, progress_bar: MyProgressBar):
        self.progress_bar = progress_bar
        self.current_chars = 0
        self.collected_text: str = ""

    def on_llm_new_token(self, token: str, **kwargs):  # type: ignore[override]
        self.current_chars += len(token)
        self.progress_bar.set_current(self.current_chars)
        self.collected_text += token


# ────────────────────────────  Main Pipeline  ────────────────────────────


class LongChainTextPreprocessor:
    """End-to-end speaker tagging & formatting pipeline."""

    # ── construction ────────────────────────────────────────────────────

    def __init__(self, config_path: str = "configuration.yaml"):
        self.config = self._load_config(config_path)
        self._load_secrets()
        self._llm_cfg = self.config["llm_preprocessing"]

        # voice catalogue cache
        self._voices = listAvailableVoices()
        self._voice_names = {v["name"] for v in self._voices}

        # load prompts once
        self.edit_prompt_str = self._load_prompt("prompts/edit-raw-story-prompt.md")
        self.speaker_prompt_str = self._load_prompt(
            "prompts/determine-and-summarize-speakers-prompt.md"
        )

    # ── config & LLM helpers ────────────────────────────────────────────

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if "llm_preprocessing" not in cfg:
            raise ValueError("Missing 'llm_preprocessing' section in config.")
        return cfg

    def _load_secrets(self) -> None:
        try:
            with open("gpt_secret_key.txt", encoding="utf-8") as fh:
                self._api_key = fh.read().strip()
        except FileNotFoundError as exc:
            safe_print(colored("Error: gpt_secret_key.txt not found", "red"))
            raise exc

    def _create_llm(
        self,
        *,
        streaming: bool = False,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> ChatOpenAI:
        cb_mgr = CallbackManager(handlers=callbacks or [])
        return ChatOpenAI(
            api_key=self._api_key,  # custom param names kept
            base_url=self._llm_cfg["endpoint"],
            model_name=self._llm_cfg["model"],
            streaming=streaming,
            callback_manager=cb_mgr,
        )

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    # ── voice utilities ────────────────────────────────────────────────

    def get_voice_catalogue(self) -> str:
        tiers: dict[str, list[str]] = {"S": [], "A": [], "B": [], "X": []}
        for v in self._voices:
            tiers[v["tier"]].append(v["name"])
        lines: list[str] = []
        for tier in ["S", "A", "B", "X"]:
            if tiers[tier]:
                lines.append(f"\n{tier} tier voices:")
                lines.extend(f"- {name}" for name in sorted(tiers[tier]))
        return "\n".join(lines)

    @staticmethod
    def _extract_voices(text: str) -> List[str]:
        return re.findall(r'<speaker[^>]*voice="([^"]+)"[^>]*>', text, re.IGNORECASE)

    # ---------- prefix-aware existence check & auto-fix -----------------

    def _voice_exists(self, name: str) -> bool:
        if name in self._voice_names:
            return True
        core = name[2:] if len(name) > 2 and name[1] == "_" else name
        return any(f"{p}{core}" in self._voice_names for p in ("a_", "s_", "b_", "x_"))

    def _fix_voice_prefixes(self, text: str) -> str:
        pattern = re.compile(r'(<speaker[^>]*voice=")([^"]+)(")', flags=re.IGNORECASE)

        def repl(match):
            pre, name, post = match.groups()
            if name in self._voice_names:
                return match.group(0)
            core = name[2:] if len(name) > 2 and name[1] == "_" else name
            for prefix in ("a_", "s_", "b_", "x_"):
                candidate = f"{prefix}{core}"
                if candidate in self._voice_names:
                    return f"{pre}{candidate}{post}"
            return match.group(0)

        return pattern.sub(repl, text)

    def _missing_voices(self, text: str) -> List[str]:
        """Return ONLY the voices that are still unknown."""
        return sorted({v for v in self._extract_voices(text) if not self._voice_exists(v)})

    # ── high-level steps ────────────────────────────────────────────────

    def _determine_speakers(self, text: str) -> str:
        tmpl = ChatPromptTemplate.from_template(self.speaker_prompt_str)
        messages = tmpl.format_messages(voices=self.get_voice_catalogue(), text=text)
        llm = self._create_llm(streaming=False)
        result = llm.invoke(messages)
        return result.content.strip() if hasattr(result, "content") else str(result)

    # ────────────────────────────────────────────────────────────────────
    #                          Chunk handling
    # ────────────────────────────────────────────────────────────────────

    def _stream_process_chunk(
        self, chunk: str, chunk_index: int, summary: str, file_id: str
    ) -> str:
        """Process one chunk using its own tqdm bar and LLM instance."""
        pb = MyProgressBar(len(chunk), chunk_index, f"{file_id}-{chunk_index}")
        callback = ProgressCallback(pb)
        llm = self._create_llm(streaming=True, callbacks=[callback])

        prompt = ChatPromptTemplate.from_template(self.edit_prompt_str)
        messages = prompt.format_messages(
            voices=self.get_voice_catalogue(),
            character_summary=summary,
            text=chunk,
        )

        for _ in llm.stream(messages):
            pass  # ProgressCallback handles updates

        pb.set_current(len(chunk))
        pb.close()
        return callback.collected_text

    # ────────────────────────────────────────────────────────────────────
    #                             Retry loop
    # ────────────────────────────────────────────────────────────────────

    def _process_chunk_with_retry(
        self,
        chunk: str,
        chunk_index: int,
        summary: str,
        file_id: str,
        max_attempts: int = 3,
    ) -> Tuple[int, Optional[str]]:
        for attempt in range(1, max_attempts + 1):
            try:
                result = self._stream_process_chunk(chunk, chunk_index, summary, file_id)
                result = self._fix_voice_prefixes(result)

                in_len, out_len = len(chunk), len(result)
                safe_print(
                    colored(
                        f"[{file_id}-{chunk_index}] attempt {attempt}: in={in_len}, out={out_len}",
                        "blue",
                    )
                )

                # heuristic sanity checks
                if not (0.6 * in_len <= out_len <= 2.5 * in_len):
                    safe_print(
                        colored("  ↳ discarded: length ratio out of bounds", "yellow")
                    )
                    continue

                missing = self._missing_voices(result)
                if missing:
                    safe_print(
                        colored(f"  ↳ discarded: unknown voices {missing}", "yellow")
                    )
                    continue

                return chunk_index, result

            except Exception as exc:  # noqa: BLE001
                safe_print(colored(f"  ↳ exception on attempt {attempt}: {exc}", "red"))
                time.sleep(1)

        return chunk_index, None  # all retries failed

    # ────────────────────────────────────────────────────────────────────
    #                           File processing
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _split_text_into_chunks(text: str, *, max_chars: int = 10_000) -> List[str]:
        return TextUtils.chunk_text(text, max_chars=max_chars)

    def _get_unprocessed_files(self) -> List[str]:
        raw_files = {f for f in os.listdir("1-raw-text") if f.lower().endswith(".txt")}
        processed = set(os.listdir("2-annotated-text"))
        return [f for f in raw_files if f not in processed]

    # --------------------------------------------------------------------

    def process_file(self, filename: str) -> None:
        file_id = os.path.basename(filename)
        safe_print(colored(f"\nAssigning speakers in {filename}", "white"))

        with open(os.path.join("1-raw-text", filename), encoding="utf-8") as fh:
            text = fh.read()

        speaker_summary = self._determine_speakers(text)
        safe_print(colored(f"Speakers determined: {speaker_summary}", "green"))

        chunks = self._split_text_into_chunks(text)
        if not chunks:
            safe_print(colored("No content to process; skipping.", "yellow"))
            return

        results: list[Optional[str]] = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as pool:
            futures = {
                pool.submit(
                    self._process_chunk_with_retry, chunk, idx, speaker_summary, file_id
                ): idx
                for idx, chunk in enumerate(chunks)
            }

            try:
                for future in as_completed(futures):
                    idx, res = future.result()
                    if res is None:
                        safe_print(
                            colored(
                                "Failed to generate all chunks. Aborting file.", "red"
                            )
                        )
                        for f in futures:
                            f.cancel()
                        try:
                            pool.shutdown(wait=False, cancel_futures=True)  # Py ≥3.11
                        except TypeError:
                            pool.shutdown(wait=False)
                        return
                    results[idx] = res
            finally:
                pool.shutdown(wait=True)

        output_path = os.path.join("2-annotated-text", filename)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(results))

        safe_print(colored(f"\n✓ Completed: 2-annotated-text/{filename}\n", "green"))

    # --------------------------------------------------------------------

    def process_all_files(self) -> None:
        files = self._get_unprocessed_files()
        if not files:
            safe_print(colored("No unprocessed .txt files in 1-raw-text.", "gray"))
            return

        safe_print(colored(f"Found {len(files)} file(s) to process\n", "yellow"))
        for fname in files:
            try:
                self.process_file(fname)
            except Exception as exc:  # noqa: BLE001
                safe_print(colored(f"Error processing {fname}: {exc}", "red"))


# ──────────────────────────────  entry point ─────────────────────────────

if __name__ == "__main__":
    try:
        LongChainTextPreprocessor().process_all_files()
    except Exception as exc:  # noqa: BLE001
        safe_print(colored(f"Fatal error: {exc}", "red"))
