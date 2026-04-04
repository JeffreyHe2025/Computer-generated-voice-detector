import pandas as pd
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import os

from dotenv import load_dotenv






script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, "key.env")


load_dotenv(env_path)


my_api_key = os.getenv("ElevenLabsKey")



client = ElevenLabs(api_key=my_api_key)





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


#placeholder to bypass ElevenLabs 100-character minimum
placeholder = " Lorem ipsum dolor sit amet consectetur adipiscing elit quisque faucibus ex sapien vitae pellentesque"

df_high_quality['design_sentence'] = df_high_quality['sentence'] + placeholder


output_dir = "ai_clips"
os.makedirs(output_dir, exist_ok=True)
for index, row in df_high_quality.head(10).iterrows():
    
    
    voice_prompt = "A "+ row['age']+" "+ row['gender']+ " with a "+row['accents'] +" accent."
    
    
    
    
  
    previews = client.text_to_voice.design(
        model_id="eleven_multilingual_ttv_v2", 
        voice_description=voice_prompt,
        text=row['design_sentence']  
    )
    
    # 3. Save the generated preview to your account to get a usable voice_id
    generated_id = previews.previews[0].generated_voice_id
    
    temp_voice = client.text_to_voice.create(
        voice_name=f"Temp_DF_Voice_{index}",
        voice_description=voice_prompt,
        generated_voice_id=generated_id
    )
    
    # 4. Generate the actual dialogue using the newly created voice
    
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
    client.voices.delete(voice_id=temp_voice.voice_id)
    










