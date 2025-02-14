import os
import json
import numpy as np
import soundfile as sf
import torchaudio
import torch
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import yaml
from generate_helper import print_generation_status

SILENCE_THRESHOLD = 0.001

with open("configuration.yaml", "r") as f:
    config = yaml.safe_load(f)

zonos_min_chars = config["models"]["zonos"]["min_characters"]
zonos_max_chars = config["models"]["zonos"]["max_characters"]




def get_emotion(emotion_name):
    """
    Returns the corresponding emotion tensor if recognized.
    If unknown, falls back to the neutral emotion vector.
    """
    if emotion_name:
        emotion_name = emotion_name.strip().lower()
    return EMOTION_VECTORS.get(emotion_name, EMOTION_VECTORS["neutral"])

def generate_zonos_voice_lines(
    temp_folder="temp",
    voiceSpeed01=0.9,
    sentenceSilenceSec=0.5
):
    """
    Loads metadata_split.json from temp_folder, filters for chunks with model == 'zonos',
    then uses the Zonos model to generate audio for each chunk.
    The resulting WAV files are named chunk_XXXX.wav in temp_folder,
    where XXXX is the metadata 'index'.
    """

    metadata_file = os.path.join(temp_folder, "metadata_split.json")
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found. Please run split_voice_lines.py first.")
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    zonos_entries = [e for e in entries if e.get("model", "kokoro").lower() == "zonos"]
    if not zonos_entries:
        print("\033[90mNo entries found for Zonos generation.\033[0m")
        return

    speakers_dir = "speakers"
    if not os.path.exists(speakers_dir):
        print(f"Error: {speakers_dir} directory not found. Cannot load Zonos voices.")
        return

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print("\033[90m" + f"Loading Zonos model...\033[0m")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

    speaker_embedding_cache = {}

    def get_speaker_embedding(voice_name):
        if voice_name in speaker_embedding_cache:
            return speaker_embedding_cache[voice_name]

        emb_filename = os.path.join("speakers", voice_name + ".bin")
        if os.path.exists(emb_filename):
            spk_embedding = torch.load(emb_filename, map_location=device)
            speaker_embedding_cache[voice_name] = spk_embedding
            return spk_embedding

        sample_path = os.path.join("speakers", voice_name)
        possible_extensions = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
        found_sample = None
        for ext in possible_extensions:
            sample_path_with_ext = sample_path + ext
            if os.path.exists(sample_path_with_ext):
                found_sample = sample_path_with_ext
                break

        if found_sample is None:
            print(f"Warning: No valid voice sample found for '{voice_name}' in speakers directory.")
            print("Falling back to a zero embedding.")
            spk_embedding = torch.zeros((1, 768), dtype=torch.float32, device=device)
            speaker_embedding_cache[voice_name] = spk_embedding
            return spk_embedding

        wav, sr = torchaudio.load(found_sample)
        print("\033[90mGenerating speaker embedding for {} using sample '{}'...\033[0m".format(voice_name, found_sample))
        spk_embedding = model.make_speaker_embedding(wav, sr)
        spk_embedding = spk_embedding.to(device)
        torch.save(spk_embedding, emb_filename)
        speaker_embedding_cache[voice_name] = spk_embedding
        return spk_embedding

    generated_metadata = []
    
    i = 0
    for entry in zonos_entries:
        i += 1
        idx = entry["index"]
        voice_name = entry["voice"]
        text_chunk = entry["text"]
        emotion_tag = entry.get("emotion", "")

        print_generation_status("Zonos", i, len(zonos_entries), len(text_chunk), zonos_min_chars, zonos_max_chars, text_chunk)

        speaker_embedding = get_speaker_embedding(voice_name)
        emotion = get_emotion(emotion_tag).to(device)

        cond_dict = make_cond_dict(text=text_chunk, speaker=speaker_embedding, emotion=emotion, language="en-us")
        conditioning = model.prepare_conditioning(cond_dict)

        min_p = 0.15
        if len(text_chunk) < zonos_min_chars:
            min_p = 0.1
        generation_params = dict(min_p=min_p)

        duration_ms = 250
        sampling_rate = model.autoencoder.sampling_rate
        num_samples = int(sampling_rate * duration_ms / 1000)

        wav_prefix = torch.zeros(1, num_samples, device=device, dtype=torch.float32)

        with torch.autocast(device_str, dtype=torch.float32):
            audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))

        print("\033[90m")
        codes = model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=audio_prefix_codes,
            sampling_params=generation_params)
        print("\033[0m")

        wavs = model.autoencoder.decode(codes)
        wavs = wavs.cpu().numpy()
        samples = wavs[0]

        while samples.ndim > 2:
            samples = np.squeeze(samples, axis=0)

        samples = samples.astype(np.float32)
        sampling_rate = model.autoencoder.sampling_rate

        if samples.ndim == 2:
            samples = samples.transpose(1, 0)

        out_filename = os.path.join(temp_folder, f"chunk_{idx:04d}.wav")
        sf.write(out_filename, samples, sampling_rate)

        data, samplerate = sf.read(out_filename)
        average_amplitude = np.mean(np.abs(data))
        if average_amplitude < SILENCE_THRESHOLD:
            print(f"\033[31mEmpty output detected, trying again\033[0m")
            continue

        entry["filename"] = out_filename
        generated_metadata.append(entry)

    gen_metadata_file = os.path.join(temp_folder, "metadata_generated_zonos.json")
    with open(gen_metadata_file, "w", encoding="utf-8") as f:
        json.dump(generated_metadata, f, indent=2)
        
        
# Predefined emotion vectors
EMOTION_VECTORS = {
    "happy": torch.tensor([[
        0.8,  # Happiness 😊
        0.05, # Sadness 😢
        0.02, # Disgust 🤢
        0.02, # Fear 😨
        0.05, # Surprise 😲
        0.03, # Anger 😡
        0.02, # Other 🤷
        0.01  # Neutral 😐
    ]], dtype=torch.float32),

    "sad": torch.tensor([[
        0.05, # Happiness 😊
        0.8,  # Sadness 😢
        0.02, # Disgust 🤢
        0.02, # Fear 😨
        0.03, # Surprise 😲
        0.03, # Anger 😡
        0.02, # Other 🤷
        0.03  # Neutral 😐
    ]], dtype=torch.float32),

    "disgusted": torch.tensor([[
        0.02, # Happiness 😊
        0.02, # Sadness 😢
        0.8,  # Disgust 🤢
        0.05, # Fear 😨
        0.02, # Surprise 😲
        0.05, # Anger 😡
        0.02, # Other 🤷
        0.02  # Neutral 😐
    ]], dtype=torch.float32),

    "fearful": torch.tensor([[
        0.02, # Happiness 😊
        0.02, # Sadness 😢
        0.05, # Disgust 🤢
        0.8,  # Fear 😨
        0.02, # Surprise 😲
        0.03, # Anger 😡
        0.02, # Other 🤷
        0.02  # Neutral 😐
    ]], dtype=torch.float32),

    "surprised": torch.tensor([[
        0.05, # Happiness 😊
        0.02, # Sadness 😢
        0.02, # Disgust 🤢
        0.02, # Fear 😨
        0.8,  # Surprise 😲
        0.05, # Anger 😡
        0.02, # Other 🤷
        0.02  # Neutral 😐
    ]], dtype=torch.float32),

    "angry": torch.tensor([[
        0.02, # Happiness 😊
        0.02, # Sadness 😢
        0.05, # Disgust 🤢
        0.05, # Fear 😨
        0.02, # Surprise 😲
        0.8,  # Anger 😡
        0.02, # Other 🤷
        0.02  # Neutral 😐
    ]], dtype=torch.float32),

    "neutral": torch.tensor([[
        0.1,  # Happiness 😊
        0.1,  # Sadness 😢
        0.1,  # Disgust 🤢
        0.1,  # Fear 😨
        0.1,  # Surprise 😲
        0.1,  # Anger 😡
        0.1,  # Other 🤷
        0.3   # Neutral 😐 (Favoring neutral slightly)
    ]], dtype=torch.float32)
}

if __name__ == "__main__":
    generate_zonos_voice_lines()
