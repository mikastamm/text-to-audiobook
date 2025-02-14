import os
import re
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

ABBREVS = {"dr", "mr", "mrs", "ms", "etc"}  # Lowercase set for quick check

def parse_speakers(full_text, default_voice="af_bella"):
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

def _split_regular_text(text):
    sentences = []
    buffer = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buffer.append(ch)
        if ch in {'.', '!', '?'}:
            next_char = text[i + 1] if (i + 1 < n) else ''
            if next_char.isspace() or i == n - 1:
                temp_str = "".join(buffer)
                match = re.search(r"\b([A-Za-z]+)\.$", temp_str)
                if match:
                    word_before_dot = match.group(1).lower()
                    if word_before_dot in ABBREVS:
                        pass
                    else:
                        sentences.append(temp_str.strip())
                        buffer = []
                        while i + 1 < n and text[i + 1].isspace():
                            i += 1
                else:
                    sentences.append(temp_str.strip())
                    buffer = []
                    while i + 1 < n and text[i + 1].isspace():
                        i += 1
        i += 1
    leftover = "".join(buffer).strip()
    if leftover:
        sentences.append(leftover)
    return sentences

def initial_split_into_sentences(text):
    text = re.sub(r'\s*</pause>', '', text)
    parts = re.split(r'(<pause\s+duration="(?:short|long)"\s*/?>)', text)
    sentences = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            sub_sentences = _split_regular_text(part)
            if sub_sentences:
                sentences.extend(sub_sentences)
        else:
            m = re.search(r'duration="(short|long)"', part)
            duration = m.group(1) if m else "short"
            if sentences:
                sentences[-1] = sentences[-1].strip() + " ||PAUSE:" + duration + "||"
            else:
                sentences.append("||PAUSE:" + duration + "||")
    return sentences

def split_on_delims(long_text, max_len):
    def splitter(text, delimiters):
        pieces = [text]
        for delim in delimiters:
            new_pieces = []
            for p in pieces:
                if len(p) <= max_len:
                    new_pieces.append(p)
                else:
                    sub_parts = p.split(delim)
                    for sp in sub_parts:
                        sp = sp.strip()
                        if sp:
                            new_pieces.append(sp)
            pieces = new_pieces
        return pieces

    candidate_delims = [" - ", ", "]
    splitted = splitter(long_text, candidate_delims)
    final_list = []
    for piece in splitted:
        if len(piece) > max_len:
            final_list.extend(force_split_long_text(piece, max_chars=max_len))
        else:
            final_list.append(piece)
    return final_list

def force_split_long_text(text, max_chars):
    if len(text) <= max_chars:
        return [text]

    splits = []
    start = 0
    length = len(text)
    while start < length:
        end = start + max_chars
        if end >= length:
            splits.append(text[start:].strip())
            break
        space_index = text.rfind(" ", start, end)
        if space_index == -1 or space_index < start:
            splits.append(text[start:end].strip())
            start = end
        else:
            splits.append(text[start:space_index].strip())
            start = space_index + 1
    return splits

def combine_into_blocks(sentences, min_chars, max_chars):
    blocks = []
    current = ""
    for sentence in sentences:
        m = re.search(r'\|\|PAUSE:(short|long)\|\|$', sentence)
        if m:
            pause = m.group(1)
            sentence_text = sentence[:sentence.rfind("||PAUSE:")].strip()
            if current:
                blocks.append((current.strip(), ""))
                current = ""
            blocks.append((sentence_text, pause))
        else:
            if not current:
                current = sentence
            else:
                if len(current) + 1 + len(sentence) <= max_chars:
                    current += " " + sentence
                else:
                    if len(current) >= min_chars:
                        blocks.append((current.strip(), ""))
                        current = sentence
                    else:
                        current += " " + sentence
    if current.strip():
        blocks.append((current.strip(), ""))
    return blocks

# --- Main function: split_voice_lines ---

def split_voice_lines(input_file=None, input_dir="2-annotated-text", temp_folder="temp", default_voice="af_bella"):
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

            sentences = initial_split_into_sentences(segment_text)
            refined = []
            for s in sentences:
                if len(s) > model_max_chars:
                    refined.extend(split_on_delims(s, max_len=model_max_chars))
                else:
                    refined.append(s)

            blocks = combine_into_blocks(refined, min_chars=model_min_chars, max_chars=model_max_chars)
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
