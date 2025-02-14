import os
import json
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import yaml

from generate_helper import print_header
from ffmpeg_normalize import FFmpegNormalize  # New import for normalization

TARGET_SAMPLE_RATEHz = 24000
PROGRESS_BAR_WIDTH = 50  # total width including the boundary pipes

def downsample_if_needed(samples: np.ndarray, sr: int, target_sr: int = TARGET_SAMPLE_RATEHz) -> np.ndarray:
    """
    If sr != target_sr, resample the audio data to target_sr using scipy.signal.resample_poly.
    - samples can be mono (shape=(frames,)) or multi-channel (shape=(frames, channels)).
    - The returned samples are always np.float32.
    """
    if sr == target_sr:
        return samples
    up = target_sr
    down = sr
    resampled = resample_poly(samples, up, down, axis=0)
    return resampled.astype(np.float32)

def merge_voice_lines(temp_folder="temp", output_folder="3-output"):
    """
    Reads chunk_{index:04d}.wav files from temp_folder,
    downsamples them to 24 kHz if necessary, concatenates all chunks for each source file,
    adds one second of silence to the beginning and end of the merged file,
    writes the merged audio as an intermediate WAV file in output_folder, and then
    applies the specified encoding via an FFmpeg command. If loudness normalization is enabled,
    the FFmpeg command adds the corresponding filter.
    """
    with open("configuration.yaml", "r") as f:
        config = yaml.safe_load(f)
    use_loudness_normalization = config.get("useLoudnessNormalization", True)

    metadata_file = os.path.join(temp_folder, "metadata_split.json")
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found. Cannot merge voice lines.")
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    sources = {}
    for entry in metadata:
        source = entry["source"]
        if source not in sources:
            sources[source] = []
        sources[source].append(entry)

    # Mapping of codec names to their corresponding file extensions.
    codec_extensions = {
        "aac": "m4a",
        "mp3": "mp3",
        "flac": "flac",
        "wav": "wav",
        "opus": "opus",
        "ogg": "ogg",
    }

    for source_filename, entries in sources.items():
        entries_sorted = sorted(entries, key=lambda m: m["index"])
        audio_pieces = []

        # Pre-calculate how many chunks need downsampling.
        downsample_total = 0
        for entry in entries_sorted:
            chunk_file = os.path.join(temp_folder, f"chunk_{entry['index']:04d}.wav")
            if os.path.exists(chunk_file):
                try:
                    info = sf.info(chunk_file)
                except Exception as e:
                    print(f"Error reading info for {chunk_file}: {e}")
                    continue
                if info.samplerate != TARGET_SAMPLE_RATEHz:
                    downsample_total += 1

        downsample_done = 0

        for entry in entries_sorted:
            chunk_file = os.path.join(temp_folder, f"chunk_{entry['index']:04d}.wav")
            if os.path.exists(chunk_file):
                samples, sr = sf.read(chunk_file, always_2d=False)
                if sr != TARGET_SAMPLE_RATEHz:
                    downsample_done += 1
                    # Calculate inner width (excluding the two boundary pipes)
                    inner_width = PROGRESS_BAR_WIDTH - 2
                    filled = int((downsample_done / downsample_total) * inner_width)
                    bar = "|" + ("|" * filled) + ("-" * (inner_width - filled)) + "|"
                    print("\r" + bar, end="", flush=True)
                    samples = downsample_if_needed(samples, sr, TARGET_SAMPLE_RATEHz)

                # Add silence to the end of each chunk based on pause metadata.
                pause_type = entry.get("pause_end", "")
                if pause_type == "short":
                    silence_duration = config.get("shortPauseSec", 0.5)
                elif pause_type == "long":
                    silence_duration = config.get("longPauseSec", 2.0)
                else:
                    silence_duration = config.get("sentenceSilenceSec", 0.0)
                
                if silence_duration > 0:
                    silence_frames = int(TARGET_SAMPLE_RATEHz * silence_duration)
                    if samples.ndim == 2:
                        # For multi-channel, create silence for each channel.
                        silence_2d = np.zeros((samples.shape[1], silence_frames), dtype=np.float32)
                        samples = np.concatenate([samples, silence_2d.T], axis=0)
                    else:
                        silence_1d = np.zeros(silence_frames, dtype=np.float32)
                        samples = np.concatenate([samples, silence_1d], axis=0)

                # Apply loudnorm, speechnorm here using FFmpegNormalize.
                if use_loudness_normalization:
                    processed_chunk_file = os.path.join(temp_folder, f"chunk_{entry['index']:04d}_processed.wav")
                    norm_chunk_file = os.path.join(temp_folder, f"chunk_{entry['index']:04d}_norm.wav")
                    # Write the current processed segment to a temporary file.
                    sf.write(processed_chunk_file, samples, TARGET_SAMPLE_RATEHz, format="WAV")
                    # Instantiate FFmpegNormalize with desired parameters.
                    normalizer = FFmpegNormalize(normalization_type="ebu", target_level=-23.0, print_stats=False, sample_rate=TARGET_SAMPLE_RATEHz)
                    normalizer.add_media_file(processed_chunk_file, norm_chunk_file)
                    normalizer.run_normalization()
                    # Read back the normalized audio.
                    samples, sr = sf.read(norm_chunk_file, always_2d=False)
                    os.remove(processed_chunk_file)
                    os.remove(norm_chunk_file)

                audio_pieces.append(samples)
            else:
                print(f"Warning: chunk file {chunk_file} not found.")

        


        if downsample_total > 0:
            print()  # Move to the next line after the progress bar

        if not audio_pieces:
            print(f"No audio pieces found for {source_filename}")
            continue

        # Concatenate audio pieces for the current source file.
        merged_audio = np.concatenate(audio_pieces, axis=0)
        # Add one second of silence to the beginning and end of the merged audio.
        silence_duration_sec = 1.0
        silence_frames = int(TARGET_SAMPLE_RATEHz * silence_duration_sec)
        if merged_audio.ndim == 2:
            silence = np.zeros((silence_frames, merged_audio.shape[1]), dtype=merged_audio.dtype)
        else:
            silence = np.zeros(silence_frames, dtype=merged_audio.dtype)
        merged_audio = np.concatenate([silence, merged_audio, silence], axis=0)

        base_name = os.path.splitext(source_filename)[0]
        os.makedirs(output_folder, exist_ok=True)

        # Write intermediate WAV file.
        intermediate_wav = os.path.join(output_folder, f"{base_name}_temp.wav")
        sf.write(intermediate_wav, merged_audio, TARGET_SAMPLE_RATEHz, format="WAV")

        # Determine final output file and encoding parameters.
        selected_codec = config.get("encoding", {}).get("codec", "aac")
        final_ext = codec_extensions.get(selected_codec, "wav")
        final_output = os.path.join(output_folder, f"{base_name}.{final_ext}")

        # Build FFmpeg command for final conversion.
        ffmpeg_command = [
            "ffmpeg",
            "-i", intermediate_wav
        ]
        if use_loudness_normalization:
            ffmpeg_command += ["-filter:a", "speechnorm,loudnorm"]
        ffmpeg_command += [
            "-c:a", selected_codec,
            "-b:a", config.get("encoding", {}).get("bitrate", "96k"),
            "-y",  # Overwrite output file if it exists.
            final_output
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode == 0:
            os.remove(intermediate_wav)
            print_header(f"Success! Final output at {final_output}")
        else:
            print(f"\033[31mFFmpeg conversion failed: {result.stderr}\033[0m")

if __name__ == "__main__":
    merge_voice_lines()
