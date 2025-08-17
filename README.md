# Emotional-Assistance-Bot
The Tamil Mental-Health Voice Assistant is an end-to-end, voice-only AI system that converts a user’s spoken Tamil (or mixed Tamil-English) question into an empathetic answer and then speaks that answer back to the user. 

# System Architecture
## Speech Recognition
- Whisper-small (98M parameters) used with Tamil forced decoding.
- The model outputs a string

## Natural language understanding and generation
- Gemma 3-1B fine-tuned on MentalChat16K dataset.
- Safety Guardrails implemented using domain-restricted system prompt.

## Speech Synthesis
- Meta MMS TTS model implemented with fixed style descriptor for consistent output. 
- The model returns an output audio file

- Error handling strategy: catch-and-log at each stage, return user-friendly fallback message on failure. 

# Relevant Links
- Original Dataset: https://huggingface.co/datasets/ShenLab/MentalChat16K
- Tamil Translated dataset: https://huggingface.co/datasets/vamsss/translated-16k-samples
- Trained model: https://huggingface.co/vamsss/gemma-finetune-gguf
- Trained model (Q4_K_M quantization): https://huggingface.co/vamsss/gemma-finetune-gguf-Q4_K_M-GGUF
- Trained model (Q8 quantization): https://huggingface.co/vamsss/gemma-finetune-gguf-Q8_0-GGUF
- Training, inference code, sample outputs: https://drive.google.com/file/d/1Sp-5pX-h5RhjSIzSap2ZcyEWhFUGPugh/view?usp=sharing 
