import os

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
