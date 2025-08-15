import whisper
# Load base Whisper small model
model = whisper.load_model("small")

def transcribe_tamil_with_base_whisper(audio_path):
    # Transcribe with Tamil language specified
    result = model.transcribe(
        audio_path, 
        language="ta",  # Tamil language code
        task="transcribe"
    )
    return result["text"]

# Example usage
transcription = transcribe_tamil_with_base_whisper("sample_input_recording.wav")
print(f"Tamil Transcription: {transcription}")