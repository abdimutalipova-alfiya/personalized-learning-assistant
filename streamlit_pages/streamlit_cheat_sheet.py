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


def setup_cheatsheet_crew(topic, context, llm):
    if context:
        # If context is provided, generate a summary based on the provided context
        goal = f"""Generate a concise, one-page summary of the provided context: {context}. The summary should capture the most essential points, highlighting key takeaways, facts, and actionable insights."""
        backstory = "You are an expert in distilling large volumes of information into clear and compact summaries. Your task is to focus on the most important details of the provided context and organize them in a way that is easy for users to review and apply."
    else:
        # If no context is provided, generate a general summary based on the topic
        goal = f"""Generate a concise, one-page summary based on the topic: {topic}. Use general knowledge and insights to capture the most important points and provide actionable takeaways."""
        backstory = "You are an expert in summarizing broad topics. If no specific context is provided, use general knowledge to create a high-level summary of the topic. Ensure clarity, conciseness, and logical flow."

    cheatsheet_agent = Agent(
        role="Summarization Specialist",
        goal=goal,
        backstory=backstory,
        verbose=True,
        llm=llm
    )

    cheatsheet_task = Task(
        description="Your objective is to create a one-page summary. If context is provided, focus on that specific content. If not, create a general summary based on the topic. Avoid redundancy, ensure clarity, and keep the summary concise.",
        agent=cheatsheet_agent,
        expected_output="A concise, well-structured summary capturing the core ideas of the provided content or topic. The output should be clear, actionable, and formatted for easy review on a single page."
    )

    crew = Crew(
        agents=[cheatsheet_agent],
        tasks=[cheatsheet_task],
        verbose=True
    )

    crew_result = crew.kickoff()
    return crew_result.raw


# Streamlit interface for file upload and querying
st.title("📚 Personalized Learning Chat Assistant")

if st.session_state.selected_llm == "Groq API":
    llm = LLM(
    model="groq/llama-3.1-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY")
)
elif st.session_state.selected_llm == "OpenAI":
    llm = LLM(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)
elif st.session_state.selected_llm == "Other":
    st.warning("Other LLMs are not yet configured.")
    st.stop()
else:
    st.error("Invalid LLM selection.")
    st.stop()


# Initialize session state for chat history if not already present
if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display chat history
for entry in st.session_state.history:
    st.write(f"**User**: {entry['question']}")
    st.write(f"**Answer**: {entry['answer']}")
    st.write("---")

# Ask user to enter a question
if topic := st.chat_input("What is your topic:"):
    st.session_state.messages.append({"role": "user", "content": topic})
    with st.chat_message("user"):
        st.markdown(topic)


if topic:
    with st.spinner("Generating cheatsheet..."):
        # Get relevant document chunks and generate answer using Qrog
        query_embedding = generate_embeddings(topic, tokenizer, model)
        
        # Search the FAISS index for top relevant chunks
        k = 5  # Retrieve top 5 relevant chunks
        faiss_index = st.session_state["faiss_indexes"][0]  # For simplicity, using the first file's index (extend as needed)
        distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k)
        
        relevant_chunks = [st.session_state["documents"][0][i] for i in indices[0]]
        context = " ".join(relevant_chunks)

        # Answer the question using Qrog
        with st.chat_message("assistant"):
            answer = setup_cheatsheet_crew(topic, context, llm)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        
        # Save the interaction to the history
        st.session_state.history.append({"question": topic, "answer": answer})
        
        # Display the answer
        st.write(answer)
