import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Global variables for document storage
class DocumentContext:
    faiss_indexes = None
    documents = None
    document_sources = None  # To store references (e.g., file names or URLs)
    tokenizer = None
    model = None

# Initialize tokenizer and model
DocumentContext.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
DocumentContext.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

class DocumentProcessor:
    @staticmethod
    def generate_embeddings(text):
        """Generate embeddings for a given text with consistent dimensions."""
        max_length = 512
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding='max_length', 
            max_length=max_length,
            return_attention_mask=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return embeddings.numpy()

    @classmethod
    def chunk_and_embed_documents(cls, uploaded_files, youtube_links=None):
        all_faiss_indexes = []
        all_documents = []
        all_sources = []
        
        # Processing uploaded PDFs
        for uploaded_file in uploaded_files:
            text = ""
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            chunks = []
            words = text.split()
            for i in range(0, len(words), 100):
                chunk = ' '.join(words[i:i+100])
                chunks.append(chunk)
            
            embeddings = []
            valid_chunks = []
            for chunk in chunks:
                if chunk.strip():
                    try:
                        embedding = cls.generate_embeddings(chunk)
                        embeddings.append(embedding)
                        valid_chunks.append(chunk)
                    except Exception as e:
                        st.warning(f"Could not process chunk: {e}")
            
            if embeddings:
                embeddings = np.vstack(embeddings)
                faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
                faiss_index.add(embeddings)
                
                all_faiss_indexes.append(faiss_index)
                all_documents.append(valid_chunks)
                all_sources.append(uploaded_file.name)  
            else:
                st.error("No valid document chunks could be processed.")

        # Processing YouTube links
        if youtube_links:
            for link in youtube_links:
                video_id = link.split("v=")[-1]
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    text = "\n".join([entry['text'] for entry in transcript])
                    
                    chunks = []
                    words = text.split()
                    for i in range(0, len(words), 100):
                        chunk = ' '.join(words[i:i+100])
                        chunks.append(chunk)
                    
                    embeddings = []
                    valid_chunks = []
                    for chunk in chunks:
                        if chunk.strip():
                            try:
                                embedding = cls.generate_embeddings(chunk)
                                embeddings.append(embedding)
                                valid_chunks.append(chunk)
                            except Exception as e:
                                st.warning(f"Could not process chunk: {e}")
                    
                    if embeddings:
                        embeddings = np.vstack(embeddings)
                        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
                        faiss_index.add(embeddings)
                        
                        all_faiss_indexes.append(faiss_index)
                        all_documents.append(valid_chunks)
                        all_sources.append(f"Video: {link}")  
                    else:
                        st.error(f"No valid chunks could be processed from video {link}.")
                except Exception as e:
                    st.error(f"Error fetching transcript for {link}: {e}")

        return all_faiss_indexes, all_documents, all_sources