import streamlit as st
import speech_recognition as sr
import torch
import io
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class VoiceInputHandler:
    def __init__(self):
        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()
        
        # Optional: Load a more advanced STT model
        try:
            self.stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            self.stt_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        except Exception as e:
            st.warning(f"Advanced STT model loading failed: {e}")
            self.stt_model = None
            self.stt_processor = None

    def transcribe_audio(self, audio_data, sample_rate=16000):
        """
        Transcribe audio using multiple methods for robustness.
        """
        # Convert numpy array to wav file
        wav_file = 'temp_audio.wav'
        sf.write(wav_file, audio_data, sample_rate)

        # Method 1: SpeechRecognition (Google)
        try:
            with sr.AudioFile(wav_file) as source:
                audio_data = self.recognizer.record(source)
                google_transcript = self.recognizer.recognize_google(audio_data)
                if google_transcript:
                    return google_transcript
        except Exception as e:
            st.warning(f"Google Speech Recognition failed: {e}")

        # Method 2: Wav2Vec2 (Offline)
        if self.stt_model and self.stt_processor:
            try:
                input_values = self.stt_processor(
                    audio_data, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                ).input_values

                with torch.no_grad():
                    logits = self.stt_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                wav2vec_transcript = self.stt_processor.batch_decode(predicted_ids)[0]
                
                return wav2vec_transcript
            except Exception as e:
                st.warning(f"Wav2Vec2 transcription failed: {e}")

        st.error("Speech-to-Text transcription unsuccessful.")
        return ""
            
    def process_voice_query(self, audio_data):
        with st.spinner("Transcribing audio..."):
            transcript = self.transcribe_audio(audio_data)
        return transcript
        

