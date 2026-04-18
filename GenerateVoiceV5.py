"""import csv
import requests
import os
from dotenv import load_dotenv


load_dotenv('key.env')
API_KEY = os.getenv('ElevenLabsKey')


VOICE_IDS = {
    "American": "DVkobYCXexp3d14o2zTJ",
    "British": "mvw85byz3FSwrmGqKyQn",
    "Hong Kong": "KyuYUneXtdcuNOXAlVwn"
}

TSV_FILE = "sample_data.tsv"
OUTPUT_DIR = "ai_clips"


os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_audio(text, voice_id, output_path):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }
    data = {
        "text": text,
       
        "model_id": "eleven_monolingual_v1", 
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Successfully saved {output_path}")
    else:
        print(f"Error generating audio for {output_path}: {response.text}")

# Read the TSV and generate clips
with open(TSV_FILE, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        sentence = row['sentence']
        accent = row['accents'].strip()
        file_name = row['path']
        
        # Construct the full save path inside the ai_clips folder
        save_path = os.path.join(OUTPUT_DIR, file_name)

        # Skip if the file already exists so you don't waste your character quota
        if os.path.exists(save_path):
            print(f"Skipping {file_name}, already exists.")
            continue
            
        voice_id = VOICE_IDS.get(accent)
        
        if voice_id:
            print(f"Generating: {file_name} using {accent} voice...")
            generate_audio(sentence, voice_id, save_path)
        else:
            print(f"Warning: No Voice ID mapped for accent '{accent}'")

print("Processing complete.")"""
import csv
import os
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

TSV_FILE = "sample_data.tsv"
OUTPUT_DIR = "ai_clips"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Setup Parler-TTS (Load this BEFORE the loop)
print("Loading Parler-TTS model into memory...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
repo_id = "parler-tts/parler-tts-mini-v1"

model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
print(f"Model loaded successfully on {device.upper()}!")

# 2. Read the TSV and generate clips
with open(TSV_FILE, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        sentence = row['sentence']
        accent = row['accents'].strip()
        
        # Safely grab age and gender if they exist in this TSV, otherwise default
        age = row.get('age', '').strip()
        gender = row.get('gender', 'speaker').strip()
        
        # Parler-TTS generates WAV data, so we ensure the filename ends in .wav
        file_name = row['path'].replace('.mp3', '.wav')
        
        # Construct the full save path inside the ai_clips folder
        save_path = os.path.join(OUTPUT_DIR, file_name)

        # Skip if the file already exists so you don't waste time
        if os.path.exists(save_path):
            print(f"Skipping {file_name}, already exists.")
            continue
            
        print(f"Generating: {file_name} using {accent} accent...")
        
        # Construct the dynamic Parler-TTS text prompt
        demographics = f"{age} {gender}".strip()
        voice_prompt = f"A {demographics} with a {accent} accent delivers their words clearly. The recording is of very high quality."
        
        # Tokenize the descriptions and the sentence (with the attention mask fix)
        input_tokens = tokenizer(voice_prompt, return_tensors="pt").to(device)
        prompt_tokens = tokenizer(sentence, return_tensors="pt").to(device)
        
        # Generate the raw audio data
        generation = model.generate(
            input_ids=input_tokens.input_ids,
            attention_mask=input_tokens.attention_mask,
            prompt_input_ids=prompt_tokens.input_ids,
            prompt_attention_mask=prompt_tokens.attention_mask
        )
        
        # Convert the raw data to a numpy array for saving
        audio_arr = generation.cpu().numpy().squeeze()
        
        # Save the output audio file
        sf.write(save_path, audio_arr, model.config.sampling_rate)
        print(f"Successfully saved {save_path}")

print("Processing complete.")