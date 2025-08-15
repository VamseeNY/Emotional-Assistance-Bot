from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "vamsss/gemma-finetune-gguf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Your existing inference code structure
tamil_input="வேலை மிகவும் மன அழுத்தமாகி வருகிறது. எனது எதிர்காலம் மற்றும் வேலையைப் பற்றி நான் கவலைப்படுகிறேன்"
messages = [
    {
        "role": "user",
        "content": "நீங்கள் ஒரு பயிற்சி பெற்ற மனநல ஆலோசகர். 50 வார்த்தைகளில் தெளிவான மற்றும் சுருக்கமான பதிலுடன் பயனரை ஆறுதல்படுத்துங்கள். பயனர்: " + tamil_input
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer([text], return_tensors="pt").to("cpu")

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.5,
    top_p=0.95,
    top_k=64,
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("Generated Response:")
print(response)
