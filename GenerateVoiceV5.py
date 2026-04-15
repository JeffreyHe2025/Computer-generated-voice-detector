import csv
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

print("Processing complete.")