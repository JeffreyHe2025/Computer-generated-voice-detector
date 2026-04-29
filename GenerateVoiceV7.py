import pandas as pd
import os
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf


tsv_path = "/Users/jeffreyhe/Downloads/Computer-generated-voice-detector-main/df_high_quality_pre_placeholder.tsv"
df = pd.read_csv(tsv_path, sep='\t')


"""df.drop(['variant'],inplace=True,axis=1)
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

df_high_quality['accents'] = df_high_quality['accents'].str.replace('English', '', case=False).str.strip()
accent_mapping = {
    'United States': 'American',
    'England': 'British',
    'New Zealand': 'Kiwi',
    'nigeria': 'Nigerian'
}
df_high_quality['accents'] = df_high_quality['accents'].replace(accent_mapping)
df_high_quality = df_high_quality[df_high_quality['accents'] != 'Eastern European']

top_26_accents = df_high_quality['accents'].value_counts().head(26).index
specific_accents = df_high_quality['accents'].value_counts().iloc[28:31].index
combined_accents = list(top_26_accents) + list(specific_accents)
df_high_quality = df_high_quality[df_high_quality['accents'].isin(combined_accents)]

df_high_quality['gender'] = df_high_quality['gender'].str.replace('male_masculine', 'male').str.replace('female_feminine', 'female')

age_map = {
    'teens': '15-year-old',
    'twenties': '25-year-old', 
    'thirties': '35-year-old',
    'fourties': '45-year-old',
    'fifties': '55-year-old',
    'sixties': '65-year-old', 
    'seventies': '75-year-old',
    'eighties': '85-year-old',
    'nineties': '95-year-old'
}
df_high_quality['age'] = df_high_quality['age'].map(age_map).fillna(df_high_quality['age'])
df_high_quality.drop(['client_id'],inplace=True,axis=1)
df_high_quality.to_csv("df_high_quality_pre_placeholder.tsv", sep='\t', index=False)
df_high_quality.info()"""



print("Loading Parler-TTS model into memory...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
repo_id = "parler-tts/parler-tts-mini-v1"

model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
print(f"Model loaded successfully on {device.upper()}!")

output_dir = "ai_clips"
os.makedirs(output_dir, exist_ok=True)

human_clips_dir = "filtered_human_clips"

for index, row in df.iterrows():
    
    audio_filename = row['path'] 
    human_audio_path = os.path.join(human_clips_dir, audio_filename)
    
    
    if os.path.exists(human_audio_path):
        print(f"Processing row {index} (Found {audio_filename})...")
        
        voice_prompt = f"A {row['age']} {row['gender']} speaker with a {row['accents']} accent delivers their words clearly. The recording is of very high quality."
        text_to_read = row['sentence']
        
        input_ids = tokenizer(voice_prompt, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text_to_read, return_tensors="pt").input_ids.to(device)
        
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        
        audio_arr = generation.cpu().numpy().squeeze()
        
        filename = os.path.join(output_dir, f"output_row_{index}.wav")
        sf.write(filename, audio_arr, model.config.sampling_rate)
    
print("end")