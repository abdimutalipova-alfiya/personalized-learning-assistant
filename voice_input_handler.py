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
        
    def transcribe_audio(self, audio_data):
        try:
            wav_file = io.BytesIO(audio_data)
            with sr.AudioFile(wav_file) as source:
                audio = self.recognizer.record(source)
                transcript = self.recognizer.recognize_google(audio)
                return transcript
        except Exception as e:
            st.warning(f"Transcription failed: {e}")
            return None
        
    def process_voice_query(self, audio_data):
        if audio_data is not None:
            with st.spinner("Transcribing audio..."):
                transcript = self.transcribe_audio(audio_data)
            return transcript
        return None
        

