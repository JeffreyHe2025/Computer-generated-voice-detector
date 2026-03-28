import pandas as pd
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import os

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Load the variables from the .env file into your environment
load_dotenv() 

# Securely fetch the key
my_api_key = os.getenv("ELEVENLABS_API_KEY")

# Initialize the client
client = ElevenLabs(api_key=my_api_key)
# 2. Load the TSV file containing the transcripts
# Make sure this points to where you saved the Common Voice data
tsv_path = "C:/Users/jeffr/Downloads/human voices large collection/cv-corpus-24.0-2025-12-05/en/train.tsv"
df = pd.read_csv(tsv_path, sep='\t')

df.drop(['variant'],inplace=True,axis=1)
df.drop(['locale'],inplace=True,axis=1)
df.drop(['segment'],inplace=True,axis=1)
df.drop(['sentence_domain'],inplace=True,axis=1)
df.drop(['sentence_id'],inplace=True,axis=1)
df = df[df['accents'] != 'Non-native']
df = df[~df['accents'].astype(str).str.contains(',')]
df = df.dropna(subset=["age", "gender", "accents"])
df = df[df['gender'].isin(['male_masculine', 'female_feminine'])]
df['sentence'] = df['sentence'].str.replace('"', '')



df_high_quality = df[(df["up_votes"] > 2) & (df["down_votes"] == 0)]
df_high_quality.drop(["up_votes"],inplace=True,axis=1)
df_high_quality.drop(["down_votes"],inplace=True,axis=1)

df_high_quality.info()
df_high_quality.to_csv("new_data.tsv", sep='\t', index=False)
print(df_high_quality['accents'].value_counts().to_string()) #find how many different accents there are in the file, then compare to ElevenLabs to see which accents are avaliable

"""# 3. Create a new folder to hold the AI-generated clips
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
        print(f"Failed to generate audio for {original_filename}. Error: {e}")"""