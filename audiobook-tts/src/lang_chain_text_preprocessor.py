import os
import re
import yaml
from typing import List, Optional, Tuple
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

from generate_helper import listAvailableVoices, resolve_voice_name
from text_utils import TextUtils

def colored(text: str, color: str) -> str:
    color_codes = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'cyan': '\033[36m',
        'magenta': '\033[35m',
        'blue': '\033[34m',
        'gray': '\033[90m',
        'orange': '\033[38;5;208m',
        'white': '\033[97m'
    }
    reset = '\033[0m'
    return f"{color_codes.get(color, '')}{text}{reset}"

# Lock used to synchronize console output across threads
print_lock = threading.Lock()
progress_lock = threading.Lock()
progress_states: dict[str, str] = {}

def update_progress(identifier: str, text: str):
    """Update the progress buffer for a specific file."""
    with progress_lock:
        progress_states[identifier] = text

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def progress_printer(stop_event: threading.Event, refresh_interval: float = 0.5):
    """Continuously print all progress bars from the buffer."""
    last_lines = 0
    while not stop_event.is_set():
        with progress_lock:
            lines = [progress_states[k] for k in sorted(progress_states.keys())]
        with print_lock:
            if last_lines:
                sys.stdout.write('\x1b[%dA' % last_lines)
            for line in lines:
                sys.stdout.write('\r' + line.ljust(80) + '\n')
            sys.stdout.flush()
        last_lines = len(lines)
        time.sleep(refresh_interval)
    # final render
    with progress_lock:
        lines = [progress_states[k] for k in sorted(progress_states.keys())]
    with print_lock:
        if last_lines:
            sys.stdout.write('\x1b[%dA' % last_lines)
        for line in lines:
            sys.stdout.write('\r' + line.ljust(80) + '\n')
        sys.stdout.flush()

# Custom progress bar class.
class MyProgressBar:
    def __init__(self, accepted_segments: List[Tuple[int, int]], total_expected_chars: int,
                 current_chunk_index: int, identifier: str, bar_length: int = 50):
        self.accepted_segments = accepted_segments
        self.total_expected_chars = total_expected_chars
        self.current_chunk_index = current_chunk_index
        self.bar_length = bar_length
        self.identifier = identifier
        self.current_chunk_token_count01 = 0  # number of characters generated so far in this chunk

    def set_current(self, count: int):
        self.current_chunk_token_count01 = count

    def render_string(self) -> str:
        """Return the rendered progress bar string."""
        accepted_total = sum(count for (_, count) in self.accepted_segments)
        overall_progress = accepted_total + self.current_chunk_token_count01
        overall_filled = int(overall_progress / self.total_expected_chars * self.bar_length)

        progress_bar = ""
        accepted_blocks = 0
        for (chunk_idx, count) in self.accepted_segments:
            blocks = int(count / self.total_expected_chars * self.bar_length)
            accepted_blocks += blocks
            progress_bar += colored('|' * blocks, self._get_chunk_color(chunk_idx))
        current_blocks = overall_filled - accepted_blocks
        progress_bar += colored('|' * current_blocks, self._get_chunk_color(self.current_chunk_index))
        remaining = self.bar_length - overall_filled
        progress_bar += colored('-' * remaining, 'gray')
        progress_bar += f" {overall_progress}/{self.total_expected_chars} chars"
        return f"[{self.identifier}] {progress_bar}"

    def _get_chunk_color(self, chunk_index: int) -> str:
        colors = ['green', 'cyan', 'yellow', 'magenta', 'blue', 'red']
        return colors[chunk_index % len(colors)]

# Callback that updates the progress bar and accumulates generated text.
class ProgressCallback(BaseCallbackHandler):
    def __init__(self, progress_bar: MyProgressBar):
        self.progress_bar = progress_bar
        self.token_count01 = 0  # character count for current chunk
        self.collected_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        # Increment character count (using length of token)
        self.token_count01 += len(token)
        self.progress_bar.set_current(self.token_count01)
        update_progress(self.progress_bar.identifier, self.progress_bar.render_string())
        self.collected_text += token

class LongChainTextPreprocessor:
    def __init__(self, config_path: str = "configuration.yaml"):
        self.config = self._load_config(config_path)
        self.llm = self._setup_llm()
        # Load prompts
        self.edit_prompt = self._load_prompt("prompts/edit-raw-story-prompt.md")
        self.speaker_prompt = self._load_prompt(
            "prompts/determine-and-summarize-speakers-prompt.md")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not config.get('llm_preprocessing'):
                    raise ValueError("LLM preprocessing configuration not found")
                return config
        except Exception as e:
            safe_print(colored(f"Error loading config: {str(e)}", 'red'))
            raise

    def _setup_llm(self) -> ChatOpenAI:
        """Initialize the language model with streaming enabled and a CallbackManager."""
        llm_config = self.config['llm_preprocessing']
        try:
            with open("gpt_secret_key.txt", 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
        except Exception as e:
            safe_print(colored(f"Error loading API key: {str(e)}", 'red'))
            raise
        # Create a CallbackManager and pass it to the LLM.
        callback_manager = CallbackManager(handlers=[])
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=llm_config['endpoint'],
            model_name=llm_config['model'],
            streaming=True,
            callback_manager=callback_manager
        )
        # If callback_manager is still None, force-assign it.
        if not hasattr(llm, "callback_manager") or llm.callback_manager is None:
            llm.callback_manager = callback_manager
        return llm

    def _load_prompt(self, path: str) -> str:
        """Load prompt template from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            safe_print(colored(f"Error loading prompt from {path}: {str(e)}", 'red'))
            raise

    def _get_unprocessed_files(self) -> List[str]:
        """Get list of .txt files in 1-raw-text that are not present in the 2-annotated-text folder."""
        raw_files = {f for f in os.listdir("1-raw-text") if f.lower().endswith(".txt")}
        processed_files = set(os.listdir("2-annotated-text"))
        return [f for f in raw_files if f not in processed_files]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rudimentary token estimate used for chunk sizing."""
        return max(1, len(text) // 4)

    def _split_text_into_chunks(self, text: str, max_chars: int = 10000) -> List[str]:
        """Split text into chunks without cutting sentences."""
        return TextUtils.chunk_text(text, max_chars=max_chars)

    @staticmethod
    def getVoiceString() -> str:
        voices = listAvailableVoices()
        voice_tiers = {'S': [], 'A': [], 'B': [], 'X': []}
        for voice in voices:
            voice_tiers[voice['tier']].append(voice['name'])
        voice_list = []
        for tier in ['S', 'A', 'B', 'X']:
            if voice_tiers[tier]:
                voice_list.append(f"\n{tier} tier voices:")
                for voice in sorted(voice_tiers[tier]):
                    voice_list.append(f"- {voice}")
        return "\n".join(voice_list)

    @staticmethod
    def _extract_voices(text: str) -> List[str]:
        """Return all voice names referenced in <speaker> tags in the text."""
        return re.findall(r'<speaker[^>]*voice="([^"]+)"[^>]*>', text, flags=re.IGNORECASE)

    @staticmethod
    def _voice_exists(voice: str) -> bool:
        """Check if a voice exists, trying alternate prefixes for mixups."""
        return resolve_voice_name(voice) is not None

    def _all_voices_exist(self, text: str) -> Tuple[bool, List[str]]:
        """Return True if every voice referenced in the text exists."""
        voices = self._extract_voices(text)
        missing = [v for v in voices if not self._voice_exists(v)]
        return len(missing) == 0, missing

    def _determine_speakers(self, text: str) -> str:
        """Use the language model to summarize all speakers in the text."""
        prompt = ChatPromptTemplate.from_template(self.speaker_prompt)
        messages = prompt.format_messages(
            voices=self.getVoiceString(),
            text=text,
        )
        result = self.llm.invoke(messages)
        return result.content.strip() if hasattr(result, "content") else str(result)

    def _stream_process_chunk(self, chunk: str, chunk_index: int, summary: str) -> str:
        """
        Process a single chunk with streaming.
        A MyProgressBar instance is created to reflect overall progress,
        and a ProgressCallback is attached to update it as tokens arrive.
        """
        prompt = ChatPromptTemplate.from_template(self.edit_prompt)
        messages = prompt.format_messages(
            voices=self.getVoiceString(),
            character_summary=summary,
            text=chunk
        )
        identifier = f"{self.current_file_identifier}-{chunk_index}"
        progress_bar = MyProgressBar([], len(chunk), chunk_index, identifier)
        update_progress(identifier, progress_bar.render_string())
        callback = ProgressCallback(progress_bar)
        # Temporarily add the callback to the LLM's callback manager.
        self.llm.callback_manager.add_handler(callback)
        for _ in self.llm.invoke(messages, stream=True):
            pass
        self.llm.callback_manager.remove_handler(callback)
        with progress_lock:
            progress_states.pop(identifier, None)
        return callback.collected_text

    def _process_chunk_with_retry(self, chunk: str, chunk_index: int, summary: str, max_attempts: int = 3) -> Tuple[int, Optional[str]]:
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                result = self._stream_process_chunk(chunk, chunk_index, summary)
                input_length = len(chunk)
                output_length = len(result)
                if output_length < 0.6 * input_length or output_length > 2.5 * input_length:
                    time.sleep(1)
                    continue
                voices_ok, missing = self._all_voices_exist(result)
                if not voices_ok:
                    safe_print(colored(f"Retrying chunk due to unknown voices: {', '.join(missing)}", 'yellow'))
                    time.sleep(1)
                    continue
                return chunk_index, result
            except Exception:
                time.sleep(20)
        return chunk_index, None

    def process_file(self, filename: str):
        """Process a single file through the pipeline only if all chunks succeed."""
        self.current_file_identifier = os.path.basename(filename)
        safe_print(colored(f"\nAssigning speakers in text file {filename}", 'white'))
        input_path = os.path.join("1-raw-text", filename)
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Determine speakers first
        speaker_summary = self._determine_speakers(text)
        safe_print(colored("Speakers determined", 'green'))

        chunks = self._split_text_into_chunks(text)
        results: List[Optional[str]] = [None] * len(chunks)
        stop_event = threading.Event()
        printer = threading.Thread(target=progress_printer, args=(stop_event,), daemon=True)
        printer.start()
        with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as executor:
            futures = {executor.submit(self._process_chunk_with_retry, chunk, idx, speaker_summary): idx for idx, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                idx, res = future.result()
                if res is None:
                    stop_event.set()
                    printer.join()
                    safe_print(colored("Failed to generate. Aborting file.", 'red'))
                    return
                results[idx] = res
        stop_event.set()
        printer.join()

        output_path = os.path.join("2-annotated-text", filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))
        safe_print(colored(f"\n✓ Completed processing: 2-annotated-text/{filename}\n", 'green'))

    def process_all_files(self):
        """Process all unprocessed .txt files sequentially."""
        unprocessed_files = self._get_unprocessed_files()
        if not unprocessed_files:
            safe_print(colored("No unprocessed .txt files found in 1-raw-text. Exiting.", 'gray'))
            sys.exit(0)
        safe_print(colored(f"Found {len(unprocessed_files)} files to process", 'yellow'))

        for fname in unprocessed_files:
            try:
                self.process_file(fname)
            except Exception as e:
                safe_print(colored(f"Error processing {fname}: {str(e)}", 'red'))

if __name__ == "__main__":
    try:
        processor = LongChainTextPreprocessor()
        processor.process_all_files()
    except Exception as e:
        safe_print(colored(f"Fatal error: {str(e)}", 'red'))
