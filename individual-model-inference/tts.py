from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf

model = VitsModel.from_pretrained("facebook/mms-tts-tam")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tam")

text = "வணக்கம், எப்படி இருக்கீங்க ?"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(inputs["input_ids"])

sf.write("output.wav", output.waveform.squeeze().detach().numpy(), 16000)
