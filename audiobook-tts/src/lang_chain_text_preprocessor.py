import os
import yaml
from typing import List, Optional, Tuple
import time
import sys

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

from generate_helper import listAvailableVoices

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

# Custom progress bar class.
class MyProgressBar:
    def __init__(self, accepted_segments: List[Tuple[int, int]], total_expected_chars: int,
                 current_chunk_index: int, bar_length: int = 50):
        self.accepted_segments = accepted_segments
        self.total_expected_chars = total_expected_chars
        self.current_chunk_index = current_chunk_index
        self.bar_length = bar_length
        self.current_chunk_token_count01 = 0  # number of characters generated so far in this chunk

    def set_current(self, count: int):
        self.current_chunk_token_count01 = count

    def render(self):
        # Calculate overall progress (in characters)
        accepted_total = sum(count for (_, count) in self.accepted_segments)
        overall_progress = accepted_total + self.current_chunk_token_count01
        overall_filled = int(overall_progress / self.total_expected_chars * self.bar_length)
        
        progress_bar = ""
        accepted_blocks = 0
        for (chunk_idx, count) in self.accepted_segments:
            blocks = int(count / self.total_expected_chars * self.bar_length)
            accepted_blocks += blocks
            progress_bar += colored('|' * blocks, self._get_chunk_color(chunk_idx))
        # Blocks for current chunk:
        current_blocks = overall_filled - accepted_blocks
        progress_bar += colored('|' * current_blocks, self._get_chunk_color(self.current_chunk_index))
        remaining = self.bar_length - overall_filled
        progress_bar += colored('-' * remaining, 'gray')
        sys.stdout.write("\r" + progress_bar)
        sys.stdout.flush()

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
        self.progress_bar.render()
        self.collected_text += token

class LongChainTextPreprocessor:
    def __init__(self, config_path: str = "configuration.yaml"):
        self.config = self._load_config(config_path)
        self.llm = self._setup_llm()
        # Load only the editing prompt (character analysis prompt removed)
        self.edit_prompt = self._load_prompt("prompts/edit-raw-story-prompt.md")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not config.get('llm_preprocessing'):
                    raise ValueError("LLM preprocessing configuration not found")
                return config
        except Exception as e:
            print(colored(f"Error loading config: {str(e)}", 'red'))
            raise

    def _setup_llm(self) -> ChatOpenAI:
        """Initialize the language model with streaming enabled and a CallbackManager."""
        llm_config = self.config['llm_preprocessing']
        try:
            with open("gpt_secret_key.txt", 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
        except Exception as e:
            print(colored(f"Error loading API key: {str(e)}", 'red'))
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
            print(colored(f"Error loading prompt from {path}: {str(e)}", 'red'))
            raise

    def _get_unprocessed_files(self) -> List[str]:
        """Get list of .txt files in 1-raw-text that are not present in the 2-annotated-text folder."""
        raw_files = {f for f in os.listdir("1-raw-text") if f.lower().endswith(".txt")}
        processed_files = set(os.listdir("2-annotated-text"))
        return [f for f in raw_files if f not in processed_files]

    def _split_text_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks of specified number of lines."""
        lines = text.splitlines()
        return ['\n'.join(lines[i:i + chunk_size])
                for i in range(0, len(lines), chunk_size)]

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

    def _stream_process_chunk(self, chunk: str, previous_chunk: Optional[str],
                              accepted_segments: List[Tuple[int, int]],
                              total_expected_chars: int, chunk_index: int) -> str:
        """
        Process a single chunk with streaming.
        A MyProgressBar instance is created to reflect overall progress,
        and a ProgressCallback is attached to update it as tokens arrive.
        """
        prompt = ChatPromptTemplate.from_template(self.edit_prompt)
        messages = prompt.format_messages(
            voices=self.getVoiceString(),
            previous_chunk=previous_chunk if previous_chunk else "None",
            text=chunk
        )
        progress_bar = MyProgressBar(accepted_segments, total_expected_chars, chunk_index)
        callback = ProgressCallback(progress_bar)
        # Temporarily add the callback to the LLM's callback manager.
        self.llm.callback_manager.add_handler(callback)
        for _ in self.llm.invoke(messages, stream=True):
            pass
        self.llm.callback_manager.remove_handler(callback)
        return callback.collected_text

    def process_file(self, filename: str):
        """Process a single file through the pipeline only if all chunks succeed."""
        print(colored(f"\nAssigning speakers in text file {filename}", 'white'))
        input_path = os.path.join("1-raw-text", filename)
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        total_expected_chars = len(text)
        chunks = self._split_text_into_chunks(text)
        processed_chunks = []
        accepted_segments: List[Tuple[int, int]] = []
        all_chunks_succeeded = True
        for i, chunk in enumerate(chunks, 1):
            previous_chunks_text = '\n'.join(processed_chunks)
            max_attempts = 3
            attempt = 0
            success = False
            current_generated = ""
            while attempt < max_attempts and not success:
                attempt += 1
                sys.stdout.write("\r" + " " * 80 + "\r")
                current_generated = self._stream_process_chunk(chunk, previous_chunks_text,
                                                               accepted_segments, total_expected_chars, chunk_index=i-1)
                input_length = len(chunk)
                output_length = len(current_generated)
                if output_length < 0.75 * input_length or output_length > 2 * input_length:
                    time.sleep(1)
                else:
                    success = True
            if success:
                accepted_segments.append((i-1, len(current_generated)))
                processed_chunks.append(current_generated)
            else:
                print(colored("Failed to generate. Language model did not produce enough or produced too many characters after 3 tries. Process for this file will be aborted.", 'red'))
                all_chunks_succeeded = False
                break
            all_chunks_succeeded = all_chunks_succeeded and success
            time.sleep(1)
        if all_chunks_succeeded:
            output_path = os.path.join("2-annotated-text", filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(processed_chunks))
            print(colored(f"\n✓ Completed processing: 2-annotated-text/{filename}\n", 'green'))
        else:
            print(colored("File processing incomplete due to failed chunks. Processed file not written.", 'red'))

    def process_all_files(self):
        """Process all unprocessed .txt files in the 1-raw-text directory.
        Exits immediately if no unprocessed files are found."""
        unprocessed_files = self._get_unprocessed_files()
        if not unprocessed_files:
            print(colored("No unprocessed .txt files found in 1-raw-text. Exiting.", 'gray'))
            sys.exit(0)
        print(colored(f"Found {len(unprocessed_files)} files to process", 'yellow'))
        for filename in unprocessed_files:
            try:
                self.process_file(filename)
            except Exception as e:
                print(colored(f"Error processing {filename}: {str(e)}", 'red'))

if __name__ == "__main__":
    try:
        processor = LongChainTextPreprocessor()
        processor.process_all_files()
    except Exception as e:
        print(colored(f"Fatal error: {str(e)}", 'red'))
