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
        
    def transcribe_audio(self, wav_file):
        """
        Transcribe audio using multiple methods for robustness.
        """

        # Method 1: SpeechRecognition (Google)
        try:
            with sr.AudioFile(wav_file) as source:
                audio_data = self.recognizer.record(source)
                google_transcript = self.recognizer.recognize_google(audio_data)
                if google_transcript:
                    return google_transcript
        except Exception as e:
            st.warning(f"Google Speech Recognition failed: {e}")

    def process_voice_query(self, audio_data):
        # Transcribe
        with st.spinner("Transcribing audio..."):
            transcript = self.transcribe_audio(audio_data)

        return transcript
        

