import os
import re
from text_utils import TextUtils
import json
import yaml
import argparse

from generate_helper import print_header

with open("configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

kokoro_min_chars = config["models"]["kokoro"]["min_characters"]
kokoro_max_chars = config["models"]["kokoro"]["max_characters"]
zonos_min_chars = config["models"]["zonos"]["min_characters"]
zonos_max_chars = config["models"]["zonos"]["max_characters"]

# --- Splitting Functions ---

def parse_speakers(full_text, default_voice):
    """
    Extract segments of text under <speaker ...>...</speaker> tags.
    Recognizes tags with extra attributes:
      - voice (required)
      - emotion (optional)
    Anything outside such tags is assigned the default_voice and no emotion.
    Returns a list of (voice, text_segment, emotion) tuples.
    """
    # This regex captures the voice attribute, then any additional attributes,
    # then the inner text. The additional attributes may include emotion.
    pattern = r'<speaker\b(?=[^>]*\bvoice="([^"]+)")([^>]*)>(.*?)</speaker>'
    segments = []
    last_end = 0

    for match in re.finditer(pattern, full_text, flags=re.DOTALL):
        start, end = match.span()
        speaker_voice = match.group(1)
        attr_string = match.group(2)
        speaker_text = match.group(3)

        # Extract optional emotion attribute from the attribute string.
        emotion_match = re.search(r'\bemotion="([^"]+)"', attr_string)
        speaker_emotion = emotion_match.group(1) if emotion_match else ""

        if start > last_end:
            default_text = full_text[last_end:start].strip()
            if default_text:
                segments.append((default_voice, default_text, ""))
        speaker_text = speaker_text.strip()
        if speaker_text:
            segments.append((speaker_voice, speaker_text, speaker_emotion))
        last_end = end

    if last_end < len(full_text):
        final_text = full_text[last_end:].strip()
        if final_text:
            segments.append((default_voice, final_text, ""))
    return segments


# --- Main function: split_voice_lines ---

def split_voice_lines(input_file=None, input_dir="2-annotated-text", temp_folder="temp"):
    """
    Processes a single text file (input_file) or all text files in input_dir and creates a unified metadata file in temp_folder.
    Each metadata entry:
      - index: integer for ordering
      - voice: speaker name
      - emotion: optional emotion attribute from the speaker tag (empty if not provided)
      - text: the text chunk to generate
      - model: TTS engine (determined by checking the speakers folder)
      - source: which text file it came from
    Writes temp_folder/metadata_split.json
    """
    default_voice = config["default_speaker"]
    
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Determine available speakers by listing files in the speakers folder.
    speakers_folder = "speakers"
    speaker_files = set()
    if os.path.exists(speakers_folder):
        for file in os.listdir(speakers_folder):
            speaker_name = os.path.splitext(file)[0]
            speaker_files.add(speaker_name)

    metadata_entries = []
    idx = 1

    if input_file:
        if not input_file.lower().endswith(".txt"):
            print(f"Error: {input_file} is not a .txt file.")
            return

        print_header(f"{input_file}")

        filename = os.path.basename(input_file)
        with open(input_file, "r", encoding="utf-8") as f:
            full_text = f.read()

        segments = parse_speakers(full_text, default_voice=default_voice)
        for voice, segment_text, emotion in segments:
            # Use zonos if a file matching the speaker exists in the speakers folder, otherwise use kokoro.
            model = "zonos" if voice in speaker_files else "kokoro"
            model_min_chars = zonos_min_chars if model == "zonos" else kokoro_min_chars
            model_max_chars = zonos_max_chars if model == "zonos" else kokoro_max_chars

            sentences = TextUtils.initial_split_into_sentences(segment_text)
            refined = []
            for s in sentences:
                if len(s) > model_max_chars:
                    refined.extend(TextUtils.split_on_delims(s, max_len=model_max_chars))
                else:
                    refined.append(s)

            blocks = TextUtils.combine_into_blocks(refined, min_chars=model_min_chars, max_chars=model_max_chars)
            for block_text, pause in blocks:
                block_text = block_text.strip()
                if block_text:
                    metadata_entries.append({
                        "index": idx,
                        "voice": voice,
                        "emotion": emotion,
                        "text": block_text,
                        "model": model,
                        "source": filename,
                        "pause_end": pause if pause is not None else ""
                    })
                    idx += 1
    else:
        print("Error: No input file specified to split_voice_lines.")

    metadata_file = os.path.join(temp_folder, "metadata_split.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_entries, f, indent=2)

    print(f"\033[90mGenerating {len(metadata_entries)} voicelines...\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split voice lines from a text file or directory.")
    parser.add_argument("--input_file", type=str, help="Path to the input text file.")
    args = parser.parse_args()

    split_voice_lines(input_file=args.input_file)
