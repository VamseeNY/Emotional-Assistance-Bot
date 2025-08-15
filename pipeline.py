# -*- coding: utf-8 -*-
import whisper
import torch
from transformers import VitsModel
from transformers import AutoTokenizer
import soundfile as sf
from transformers import AutoModelForCausalLM
import re
import os

class VoiceAssistantPipeline:
    def __init__(self, model_name="vamsss/gemma-finetune-gguf"):
        """Initialize all models"""
        print("Loading models...")
        
        # Load Whisper ASR model
        self.whisper_model = whisper.load_model("small")
        print("✓ Whisper ASR model loaded")
        
        # Load LLM with Transformers
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✓ LLM model loaded")

        # Load Meta MMS-TTS Tamil model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tts_model = VitsModel.from_pretrained("facebook/mms-tts-tam").to(self.device)
        self.tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tam")
        print("✓ Meta MMS-TTS Tamil model loaded")

        
        print("All models loaded successfully!\n")

    def transcribe_audio(self, audio_path):
        """Transcribe audio to text using Whisper"""
        print(f"Transcribing audio: {audio_path}")
        
        # Transcribe with Tamil language specified
        result = self.whisper_model.transcribe(
            audio_path,
            language="ta",  # Tamil language code
            task="transcribe"
        )
        
        transcription = result["text"]
        print(f"Transcription: {transcription}")
        return transcription

    def detect_language(self, text):
        """Detect if text is Tamil or English"""
        tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
        return "tamil" if tamil_pattern.search(text) else "english"

    def build_system_prompt(self, language):
        """Build system prompt based on language"""
        if language == "tamil":
            return (
                "நீங்கள் ஒரு மனநல ஆதரவு நிபுணர். ஒரு உறுதியளிக்கும் செய்தி, பின்னர் மூன்று முதல் நான்கு வாக்கியங்கள் ஆலோசனை வழங்குதல், பின்னர் ஒரு பொருத்தமான தொடர் கேள்வி மூலம் பயனர்களை ஆறுதல்படுத்தும் வகையில் நீங்கள் வடிவமைக்கப்பட்டுள்ளீர்கள். பயனரின் மன ஆரோக்கியம் தொடர்பான கேள்விகளுக்கு மட்டுமே நீங்கள் பதிலளிக்க முடியும். பதில் சுருக்கமாகவும் தெளிவாகவும் இருக்க வேண்டும்."
            )
        else:
            return (
                "You are a mental health support expert. You are designed to console users with one assuring message, then  three to four sentences giving advice, then one appropriate follow up question. You can only answer queries regarding the user's mental health. The response must be concise and articulate"
            )

    def truncate_response(self, text, max_words=100):
        """Truncate response to maximum word count"""
        words = text.split()
        return " ".join(words[:max_words]) if len(words) > max_words else text

    def get_llm_response(self, user_input):
        """Get response from LLM using Transformers"""
        if not user_input.strip():
            return "Please enter a question."

        print(f"Processing with LLM...")
        
        language = self.detect_language(user_input)
        system_prompt = self.build_system_prompt(language)

        # Format as chat messages for Transformers
        messages = [
            {
                "role": "user",
                "content": f"{system_prompt} பயனர்: {user_input}" if language == "tamil" 
                        else f"{system_prompt} User: {user_input}"
            }
        ]

        try:
            # Apply chat template
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize and generate
            inputs = self.llm_tokenizer([text], return_tensors="pt").to("cpu")
            
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.5,
                top_p=0.95,
                top_k=64,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )

            # Decode response
            full_response = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract only the new generated part (remove input prompt)
            response = full_response[len(text):].strip()
            
            response = self.truncate_response(response)
            print(f"LLM Response: {response}")
            return response
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return f"Error generating response: {e}"


    def text_to_speech(self, text, output_path="output_audio.wav"):
        """Convert text to speech using Meta MMS-TTS Tamil"""
        print(f"Converting text to speech...")
        print(text)
        
        # Tokenize text for MMS-TTS
        inputs = self.tts_tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate speech
        with torch.no_grad():
            output = self.tts_model(input_ids)
        
        # Extract audio waveform
        audio_arr = output.waveform.squeeze().cpu().numpy()
        
        # Save audio file (MMS-TTS uses 16kHz sample rate)
        sf.write(output_path, audio_arr, 16000)
        print(f"Audio saved to: {output_path}")
        return output_path

    def process_audio_pipeline(self, input_audio_path, output_audio_path="response_audio.wav"):
        """Complete pipeline: Audio -> Text -> LLM -> Audio"""
        print("=" * 50)
        print("VOICE ASSISTANT PIPELINE")
        print("=" * 50)
        
        try:
            # Step 1: ASR - Audio to Text
            transcription = self.transcribe_audio(input_audio_path)
            
            # Step 2: LLM - Get response
            llm_response = self.get_llm_response(transcription)
            
            # Step 3: TTS - Text to Audio
            output_path = self.text_to_speech(llm_response, output_audio_path)
            
            print("\n" + "=" * 50)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"Input: {input_audio_path}")
            print(f"Transcription: {transcription}")
            print(f"Response: {llm_response}")
            print(f"Output Audio: {output_path}")
            print("=" * 50)
            
            return {
                "transcription": transcription,
                "response": llm_response,
                "output_audio": output_path
            }
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            return None


def main():
    """Main function to run the voice assistant"""
    # Initialize the pipeline
    try:
        assistant = VoiceAssistantPipeline("vamsss/gemma-finetune-gguf")
    except Exception as e:
        print(f"Error initializing models: {e}")
        print("Please ensure all model files are available and paths are correct.")
        return

    print("Voice Assistant ready!")
    print("Commands:")
    print("- Enter audio file path to process")
    print("- Type 'quit' to exit")
    print()

    while True:
        user_input = input("Enter audio file path (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        if not os.path.exists(user_input):
            print(f"Error: File '{user_input}' not found.")
            continue
            
        # Process the audio file
        result = assistant.process_audio_pipeline(user_input)
        
        if result:
            print(f"\nProcessing complete! Check '{result['output_audio']}' for the response.\n")
        else:
            print("Processing failed. Please try again.\n")


if __name__ == "__main__":
    main()
