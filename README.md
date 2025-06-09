# 🎈 Personalized Learning Assistant 

An advanced AI learning assistant that leverages NLP techniques to answer queries, summarize course material, and generate quizzes using Retrieval-Augmented Generation (RAG) architecture with multiple LLM integrations.

## 🌟 Key Features

- **Multi-source content ingestion** (PDFs, YouTube transcripts)
- **Voice interaction** with dual transcription (Google STT & Wav2Vec2)
- **Multi-LLM integration** (Google Gemini, Groq's Llama 3)
- **Context-aware responses** using FAISS vector search
- **Reference tracking** for all generated answers
- **Cheat sheet generator** for quick document summarization
- **Streamlit-based** user-friendly interface
  

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run Home.py
   ```
