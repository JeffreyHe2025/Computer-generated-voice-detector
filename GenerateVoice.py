import pandas as pd
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import os

# 1. Set up the ElevenLabs client with your API key
client = ElevenLabs(api_key="sk_f2f39330358251559869066ec182c289e094a054e32f5063")

# 2. Load the TSV file containing the transcripts
# Make sure this points to where you saved the Common Voice data
tsv_path = "C:/Users/jeffr/Downloads/human voices large collection/cv-corpus-24.0-2025-12-05/en/train.tsv"
df = pd.read_csv(tsv_path, sep='\t')

# 3. Create a new folder to hold the AI-generated clips
output_dir = "ai_clips"
os.makedirs(output_dir, exist_ok=True)

# 4. Loop through the dataset
for index, row in df.head(10).iterrows():
    original_filename = row['path']  
    text_to_speak = row['sentence']  
    
    print(f"Processing: {original_filename}...")
    
    try:
        # NEW SYNTAX: Use text_to_speech.convert and provide the specific voice_id
        audio_stream = client.text_to_speech.convert(
            text=text_to_speak,
            voice_id="21m00Tcm4TlvDq8ikWAM", # This is Rachel's ID
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        # NEW SYNTAX: Write the audio bytes directly to a file
        output_filepath = os.path.join(output_dir, f"ai_{original_filename}")
        with open(output_filepath, "wb") as f:
            for chunk in audio_stream:
                if chunk:
                    f.write(chunk)
                    
        print(f"Successfully saved {output_filepath}")
        
    except Exception as e:
        print(f"Failed to generate audio for {original_filename}. Error: {e}")