import os
import time
import numpy as np
import scipy.signal
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"
repo_id = "parler-tts/parler-tts-mini-v1"

print("Loading Parler-TTS model...")
model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
print(f"Loaded on {device}")

output_dir = "ai_clips"
os.makedirs(output_dir, exist_ok=True)


def add_recording_artifacts(audio, sr):
    """Layer synthetic mic/room artifacts onto a clean Parler clip."""
    audio = audio.astype(np.float32)

    # 1. Low-pass filter — mimics consumer mic frequency rolloff
    cutoff = np.random.uniform(6000, 8000)
    b, a = scipy.signal.butter(6, cutoff / (sr / 2), btype='low')
    audio = scipy.signal.filtfilt(b, a, audio)

    # 2. Background hiss at realistic SNR (15-30 dB room noise)
    noise = np.random.randn(len(audio)).astype(np.float32)
    snr_db = np.random.uniform(15, 30)
    sig_p = np.mean(audio ** 2) + 1e-12
    noise_p = sig_p / (10 ** (snr_db / 10))
    noise *= np.sqrt(noise_p / (np.mean(noise ** 2) + 1e-12))
    audio = audio + noise

    # 3. Tiny DC offset (cheap mic preamps)
    audio = audio + np.random.uniform(-0.001, 0.001)

    # 4. Soft saturation / mild compression
    audio = np.tanh(audio * 1.2) * 0.8

    # 5. Re-normalize to avoid clipping
    peak = np.max(np.abs(audio))
    if peak > 0.99:
        audio = audio / peak * 0.99

    return audio.astype(np.float32)


voice_prompt = "A 20-year-old male speaker with an American accent."
text_to_read = "I love my house and my car and my dog."

input_ids = tokenizer(voice_prompt, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(text_to_read, return_tensors="pt").input_ids.to(device)
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

audio_arr = generation.cpu().numpy().squeeze()

sr = model.config.sampling_rate
audio_arr = add_recording_artifacts(audio_arr, sr)

filename = os.path.join(output_dir, "custom2.wav")
sf.write(filename, audio_arr, sr)
print(f"Saved {filename}")
