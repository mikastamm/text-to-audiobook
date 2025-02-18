import os
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import numpy as np

def print_generation_status(model_name, i, total, num_chars_of_sentence, min_characters_for_model, max_characters_for_model, text):
    """
    Prints the generation status of a voice line.

    Args:
        model_name (str): The name of the model being used.
        i (int): The index of the current sentence being generated.
        total (int): The total number of sentences to generate for the model.
        num_chars_of_sentence (int): The number of characters in the current sentence.
        min_characters_for_model (int): The minimum character limit for the model.
        max_characters_for_model (int): The maximum character limit for the model.
        text (str): The text of the current sentence.
    """
    range_size = max_characters_for_model - min_characters_for_model
    if range_size == 0:
        progress01 = 1.0
    else:
        progress01 = (num_chars_of_sentence - min_characters_for_model) / range_size

    bar_length = 20
    filled_length = int(bar_length * progress01)
    bar = '\033[38;5;214m' + '|' * filled_length + '\033[0m' + '-' * (bar_length - filled_length)

    light_gray = '\033[38;5;244m'
    orange = '\033[38;5;208m'
    reset = '\033[0m'

    print(f"{orange}{model_name}{reset} {i}/{total} - {num_chars_of_sentence} - {min_characters_for_model}  {light_gray}|{bar}| {max_characters_for_model}{reset}")
    print(f"{light_gray}-> {text}{reset}")

def listAvailableVoices():
    """
    Returns a list of available voices for each model, categorized by tier.
    
    For Zonos voices:
      - S tier: Best voices for main characters with lots of dialogue
      - A tier: Good voices for regular characters
      - B tier: Backup voices when running out of better options
      - X tier: Special voices for specific character types

    For Cockroach voices:
      - Only S tier voices are available

    Supports multiple common audio file formats such as .wav, .mp3, and .ogg.
    
    Returns:
        list: A list of dictionaries containing voice information with the following structure:
              {
                  "name": str,  # Name of the voice file (without extension for Zonos)
                  "tier": str,  # S, A, B, or X
              }
    """
    voices = []

    # Add hardcoded Cockroach voices (S tier)
    voices.extend([
        {"name": "af_bella", "tier": "S"},
        {"name": "af_heart", "tier": "S"}
    ])

    # Scan Zonos voices from speakers directory
    speakers_dir = "speakers"
    allowed_extensions = ('.wav', '.mp3', '.ogg', '.bin')
    if os.path.exists(speakers_dir):
        for filename in os.listdir(speakers_dir):
            # Skip files with unsupported extensions
            if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
                continue

            # Remove extension from filename
            name = os.path.splitext(filename)[0]

            # Determine tier from filename prefix
            if name.startswith('s_'):
                tier = "S"
            elif name.startswith('a_'):
                tier = "A"
            elif name.startswith('b_'):
                tier = "B"
            elif name.startswith('x_'):
                tier = "X"
            else:
                tier="B"

            voices.append({
                "name": name,
                "tier": tier
            })

    return voices

def print_header(text):
    """
    Prints the provided text within an ASCII border with pretty colors.
    
    Args:
        text (str): The header text to be displayed. It can span multiple lines.
    """
    lines = text.splitlines() if text else [""]
    padding = 2
    max_len = max(len(line) for line in lines)
    total_width = max_len + padding * 2

    border_color = '\033[38;5;45m'
    text_color = '\033[38;5;226m'
    reset = '\033[0m'

    top_border = border_color + "+" + "-" * total_width + "+" + reset
    bottom_border = top_border

    print(top_border)
    for line in lines:
        padded_line = line.ljust(max_len)
        print(border_color + "|" + reset + " " * padding + text_color + padded_line + reset + " " * padding + border_color + "|" + reset)
    print(bottom_border)



def exceeds_silence_duration_threshold(file_path: str, max_consecutive_no_vad_timeSeconds: float) -> bool:
    """
    Checks whether an audio file exceeds the maximum allowed consecutive silence duration 
    as detected by Silero VAD.
    
    Parameters:
    file_path: str
        Path to the audio file.
    max_consecutive_no_vad_timeSeconds: float
        Maximum allowed consecutive silence (no VAD) duration in seconds.
    
    Returns:
    bool: True if no silence gap in the audio exceeds the threshold, otherwise False.
    """
    model = load_silero_vad()
    wav = read_audio(file_path)
    sampling_rate = 16000  # assuming audio is sampled at 16 kHz
    file_durationSeconds = len(wav) / sampling_rate

    speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)
    
    # If no speech is detected, the entire file is silent.
    if not speech_timestamps:
        return file_durationSeconds <= max_consecutive_no_vad_timeSeconds

    # Check silence before the first speech segment.
    if speech_timestamps[0]['start'] > max_consecutive_no_vad_timeSeconds:
        return False

    # Check gaps between consecutive speech segments.
    for i in range(1, len(speech_timestamps)):
        gap = speech_timestamps[i]['start'] - speech_timestamps[i-1]['end']
        if gap > max_consecutive_no_vad_timeSeconds:
            return False

    # Check silence after the last speech segment.
    if (file_durationSeconds - speech_timestamps[-1]['end']) > max_consecutive_no_vad_timeSeconds:
        return False

    return True

def all_exceeds_audio_and_silence_threshold(audio, silence_threshold01: float) -> bool:
    """
    Checks whether the trailing portion of the provided audio data exceeds the given silence threshold.
    
    For stereo audio, the function averages across channels. It computes the RMS amplitude for the last frame
    (512 samples) and also the average RMS over the last ten frames (if available). If either the last frame's RMS 
    or the average RMS of the last ten frames exceeds silence_threshold01, the function returns True—indicating 
    that the audio does not end in silence (which may signal that speech was not clipped).
    
    Parameters:
    audio: np.ndarray
        Audio data as a 1D array (mono) or 2D array (samples, channels) for stereo.
    silence_threshold01: float
        Threshold (between 0 and 1) that the RMS amplitude must exceed.
    
    Returns:
    bool: True if the trailing audio exceeds the silence threshold, False otherwise.
    """
    audio_np = np.array(audio)
    # If audio is stereo, average the channels.
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)

    frame_size = 512
    total_samples = audio_np.shape[0]
    if total_samples == 0:
        return False

    # Compute RMS for the last frame.
    start_idx = max(0, total_samples - frame_size)
    last_frame = audio_np[start_idx:]
    last_frame_rms = np.sqrt(np.mean(last_frame ** 2))


    return last_frame_rms > silence_threshold01 

