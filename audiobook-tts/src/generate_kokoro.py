import os
import json
import numpy as np
import soundfile as sf
import requests
import yaml
from kokoro_onnx import Kokoro
from kokoro_onnx.config import SAMPLE_RATE
from generate_helper import print_generation_status

with open("configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

kokoro_min_chars = config["models"]["kokoro"]["min_characters"]
kokoro_max_chars = config["models"]["kokoro"]["max_characters"]

def download_latest_model_files():
    models_dir = "/app/models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    release_url = "https://api.github.com/repos/thewh1teagle/kokoro-onnx/releases/latest"
    response = requests.get(release_url)
    if response.status_code != 200:
        print("\033[91mFailed to automatically download the Kokoro model. Please download it manually from https://github.com/thewh1teagle/kokoro-onnx/releases/latest and place its files into the audiobook-tts/models models directory\033[0m")        
        return None, None
    release_data = response.json()
    assets = release_data.get("assets", [])
    largest_onnx = None
    largest_bin = None
    max_size_onnx = -1
    max_size_bin = -1
    for asset in assets:
        name = asset.get("name", "")
        size = asset.get("size", 0)
        if name.endswith(".onnx") and size > max_size_onnx:
            max_size_onnx = size
            largest_onnx = asset
        elif name.endswith(".bin") and size > max_size_bin:
            max_size_bin = size
            largest_bin = asset
    if not largest_onnx or not largest_bin:
        print("\033[91mFailed to automatically download the Kokoro model. Please download it manually from https://github.com/thewh1teagle/kokoro-onnx/releases/latest and place its files into the audiobook-tts/models models directory\033[0m")        
        return None, None
    onnx_path = os.path.join(models_dir, largest_onnx["name"])
    bin_path = os.path.join(models_dir, largest_bin["name"])
    if not os.path.exists(onnx_path):
        print("Downloading Kokoro model:", largest_onnx["name"])
        r = requests.get(largest_onnx["browser_download_url"])
        with open(onnx_path, "wb") as f:
            f.write(r.content)
    
    if not os.path.exists(bin_path):
        r = requests.get(largest_bin["browser_download_url"])
        with open(bin_path, "wb") as f:
            f.write(r.content)
    return onnx_path, bin_path

def generate_kokoro_voice_lines(modelPath, voicesPath, temp_folder="temp", voiceSpeed01=0.9, sentenceSilenceSec=0.5):
    metadata_file = os.path.join(temp_folder, "metadata_split.json")
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found. Please run split_voice_lines.py first.")
        return
    with open(metadata_file, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    # Filter for entries where model is "kokoro" (case-insensitive).
    kokoro_entries = [e for e in entries if e.get("model", "kokoro").lower() == "kokoro"]
    if not kokoro_entries:
        print("No entries found for Kokoro generation.")
        return

    # Create Kokoro instance using the downloaded files.
    kokoro = Kokoro(modelPath, voicesPath)
    
    generated_metadata = []
    for entry in kokoro_entries:
        idx = entry["index"]
        voice = entry["voice"]
        text_chunk = entry["text"]
        print_generation_status("Kokoro", idx + 1, len(kokoro_entries), len(text_chunk),
                                kokoro_min_chars, kokoro_max_chars, text_chunk)
        current_sample_rate = SAMPLE_RATE
        if not any(ch.isalnum() for ch in text_chunk):
            silence_durationSec = 0.25
            samples = np.zeros(int(silence_durationSec * current_sample_rate), dtype=np.float32)
        else:
            samples, _ = kokoro.create(text_chunk, voice=voice, speed=voiceSpeed01)
        out_filename = os.path.join(temp_folder, f"chunk_{idx:04d}.wav")
        sf.write(out_filename, samples, current_sample_rate)
        entry["filename"] = out_filename
        generated_metadata.append(entry)
    
    gen_metadata_file = os.path.join(temp_folder, "metadata_generated_kokoro.json")
    with open(gen_metadata_file, "w", encoding="utf-8") as f:
        json.dump(generated_metadata, f, indent=2)

if __name__ == "__main__":
    modelPath, voicesPath = download_latest_model_files()
    if modelPath is None or voicesPath is None:
        print("\033[91mFailed to automatically download the Kokoro model. Please download it manually from https://github.com/thewh1teagle/kokoro-onnx/releases/latest and place its files into the audiobook-tts/models models directory\033[0m")        
    else:
        generate_kokoro_voice_lines(modelPath, voicesPath)
