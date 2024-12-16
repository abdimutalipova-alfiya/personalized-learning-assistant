import streamlit as st
import os
from crewai import Crew, Task, Agent, LLM
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
from PyPDF2 import PdfReader  # Make sure to import PdfReader from PyPDF2
from crewai import LLM

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the tokenizer and model for embeddings (should be done once)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Helper functions for document processing
def chunk_text_by_sentences(text, max_length=1000):
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def generate_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1000)
    with torch.no_grad():
        output = model(**inputs)
        embeddings = output.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Function to process PDF with chunks and generate embeddings
def process_pdf(uploaded_files):
    all_faiss_indexes = []  # To store FAISS indexes for all files
    all_documents = []  # To store all document chunks
    
    for uploaded_file in uploaded_files:
        text = ""
        pdf_reader = PdfReader(uploaded_file)  # Initialize PdfReader for PDF processing
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Chunk the document text
        chunks = chunk_text_by_sentences(text)
        embeddings = []
        
        for chunk in chunks:
            embedding = generate_embeddings(chunk, tokenizer, model)
            embeddings.append(embedding)
        
        embeddings = np.vstack(embeddings)

        # Add embeddings to FAISS index
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)
        
        all_faiss_indexes.append(faiss_index)
        all_documents.append(chunks)

    return all_faiss_indexes, all_documents

