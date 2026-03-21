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
#print(df_high_quality['accents'].value_counts().to_string()) #find how many different accents there are in the file, then compare to ElevenLabs to see which accents are avaliable
print(df_high_quality['age'].value_counts().to_string())


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
df['age'] = df['age'].map(age_map).fillna(df['age'])

"""for index, row in df_high_quality.iterrows():
    # 1. Format the text prompt for the voice design
    voice_prompt = f"A {row['age']} {row['gender']} with a {row['accents']} accent."
    print(f"Designing voice for row {index}: {voice_prompt}")
    
    # 2. Call the Voice Design endpoint
    previews = client.text_to_voice.design(
        model_id="eleven_multilingual_ttv_v2",
        voice_description=voice_prompt,
        text=dummy_preview_text 
    )
    
    # 3. Save the generated preview to your account to get a usable voice_id
    generated_id = previews.previews[0].generated_voice_id
    
    temp_voice = client.text_to_voice.create(
        voice_name=f"Temp_DF_Voice_{index}",
        voice_description=voice_prompt,
        generated_voice_id=generated_id
    )
    
    # 4. Generate the actual dialogue (your specific sentence) using the newly created voice
    print(f"Generating audio for sentence: '{row['sentence']}'")
    audio_generator = client.text_to_speech.convert(
        text=row['sentence'],
        voice_id=temp_voice.voice_id,
        model_id="eleven_multilingual_v2"
    )
    
    # Save the output audio file
    filename = f"output_row_{index}.mp3"
    with open(filename, "wb") as f:
        for chunk in audio_generator:
            if chunk:
                f.write(chunk)
                
    # 5. CRITICAL: Delete the voice to free up your voice slot limit
    client.voices.delete(voice_id=temp_voice.voice_id)"""











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