import pandas as pd
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import os

# 1. Set up the ElevenLabs client with your API key

from dotenv import load_dotenv


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
#print(df_high_quality['accents'].value_counts().to_string()) #find how many different accents there are in the file, then compare to ElevenLabs to see which accents are avaliable



top_26_accents = df_high_quality['accents'].value_counts().head(26).index

# 2. Get rows 29-31
specific_accents = df_high_quality['accents'].value_counts().iloc[28:31].index

# 3. Combine them together into one master list
combined_accents = list(top_26_accents) + list(specific_accents)

# 4. Filter the dataframe using the combined list
df_high_quality = df_high_quality[df_high_quality['accents'].isin(combined_accents)]

df_high_quality['accents'] = df_high_quality['accents'].str.replace('English', '', case=False).str.strip()
accent_mapping = {
    'United States': 'American',
    'England': 'British',
    'New Zealand': 'Kiwi',

}
df_high_quality['accents'] = df_high_quality['accents'].replace(accent_mapping)
df_high_quality = df_high_quality[df_high_quality['accents'] != 'Eastern European']
print(df_high_quality['accents'].value_counts().to_string())





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

"""for index, row in df_high_quality.head(10).iterrows():
    
    # 1. Format the text prompt for the voice design
    voice_prompt = "A "+ row['age']+" "+ row['gender']+ " with a "+row['accents'] +" accent."
    
    
    
    
    # 2. Call the Voice Design endpoint
    # We pass row['sentence'] so it uses your actual text to generate the preview
    previews = client.text_to_voice.design(
        model_id="eleven_multilingual_v2", # Note: use standard multilingual model ID
        voice_description=voice_prompt,
        text=row['sentence'] 
    )
    
    # 3. Save the generated preview to your account to get a usable voice_id
    generated_id = previews.previews[0].generated_voice_id
    
    temp_voice = client.text_to_voice.create(
        voice_name=f"Temp_DF_Voice_{index}",
        voice_description=voice_prompt,
        generated_voice_id=generated_id
    )
    
    # 4. Generate the actual dialogue using the newly created voice
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
    client.voices.delete(voice_id=temp_voice.voice_id)
    print(f"Finished and deleted Temp_DF_Voice_{index}\n")"""










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