import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf


device = "cuda:0" if torch.cuda.is_available() else "cpu"


print("Loading AI model... (This might take a minute the very first time)")
repo_id = "parler-tts/parler-tts-mini-v1"
model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)


text_to_read = "The Smallest Ship that Ever Crossed the Atlantic!"
voice_description = "A 25-year-old male speaker with an American accent."

print("Processing prompt...")

input_ids = tokenizer(voice_description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(text_to_read, return_tensors="pt").input_ids.to(device)


print("Generating audio...")
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)


audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_test.wav", audio_arr, model.config.sampling_rate)

print("Success! Check your folder for parler_tts_test.wav")
