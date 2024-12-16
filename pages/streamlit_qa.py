import streamlit as st
import os
from crewai import Crew, Task, Agent, LLM
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import requests
from streamlit_extras.bottom_container import bottom 
from document_processor import DocumentProcessor
from streamlit_app import configure_llm


st.set_page_config(page_title="Question-Answer Tool", page_icon="📈")
st.sidebar.header("Question-Answer Tool")

# Initialize session_state keys if not present
if "faiss_indexes" not in st.session_state:
    st.session_state["faiss_indexes"] = None
if "uploaded_files_cheatsheet" not in st.session_state:
    st.session_state["uploaded_files_cheatsheet"] = None
if "documents" not in st.session_state:
    st.session_state["documents"] = None
if "document_sources" not in st.session_state:
    st.session_state["document_sources"] = None
if "selected_llm" not in st.session_state:
    st.session_state["selected_llm"] = "Groq API"
if "llm" not in st.session_state:
    st.session_state["llm"] = None

selected_llm = st.sidebar.selectbox(
    "Select LLM", 
    ["Groq API", "Gemini"],
    help="Choose the Language Model for your queries"
)
st.session_state["selected_llm"] = selected_llm
st.session_state["llm"] = configure_llm(st.session_state["selected_llm"])

    # Upload PDFs and YouTube links
uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf", key="uploaded_files")
youtube_links = st.sidebar.text_area("Enter YouTube Links (comma separated)")
# Handle voice input
if uploaded_files or youtube_links:
    with st.spinner("Processing documents and YouTube links..."):
        youtube_links = youtube_links.split(",") if youtube_links else []
        faiss_indexes, documents, document_sources = DocumentProcessor.chunk_and_embed_documents(uploaded_files, youtube_links)
        st.session_state["faiss_indexes"] = faiss_indexes
        st.session_state["documents"] = documents
        st.session_state["document_sources"] = document_sources
        st.sidebar.success("Documents and YouTube links processed successfully!")


# Load the tokenizer and model for generating embeddings (used for matching queries with documents)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

from voice_input_handler import VoiceInputHandler

# Initialize the voice input handler for processing voice queries
voice_input_handler = VoiceInputHandler()

def fetch_general_references(question):
    """
    Fetch general references for a question using SerpAPI.
    Args:
        question (str): The user's query.
    Returns:
        list: A list of reference links or an error message.
    """
    serp_api_key = st.secrets["SERPAPI_API_KEY"] # Ensure SerpAPI key is set
    if not serp_api_key:
        return ["SerpAPI key not found. Please set SERPAPI_API_KEY environment variable."]

    # Parameters for the SerpAPI search request
    params = {
        "q": question,  # User's query
        "hl": "en",    # Language preference
        "gl": "us",   # Country preference
        "num": 3,       # Limit to 3 references
        "api_key": serp_api_key
    }

    try:
        # Send the request to SerpAPI
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        search_results = response.json()
        # Extract links from the search results
        references = [
            result["link"] for result in search_results.get("organic_results", [])
            if "link" in result
        ]
        return references if references else ["No references found."]
    except Exception as e:
        return [f"Error fetching references: {e}"]

def generate_embeddings(text, tokenizer, model):
    """
    Generate embeddings for a given text using the transformer model.
    Args:
        text (str): Input text.
    Returns:
        np.ndarray: Embedding vector.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1000)
    with torch.no_grad():
        output = model(**inputs)
        embeddings = output.last_hidden_state.mean(dim=1).numpy()  # Average the embeddings
    return embeddings

def setup_qa_crew(question, context, llm, sources):
    """
    Configure the agents and tasks for answering a question.
    Args:
        question (str): The user's query.
        context (str): Relevant document context.
        llm (LLM): Language model.
        sources (list): References.
    Returns:
        dict: The generated answer and references.
    """
    if not context or len(context.strip()) < 50:
        qa_agent = Agent(
            role="General Knowledge Assistant",
            goal="Answer user queries using general knowledge.",
            backstory="An AI assistant answering questions without document context.",
            verbose=True,
            llm=llm
        )
        qa_task = Task(
            description=f"Answer the question: '{question}'.",
            agent=qa_agent,
            expected_output="A detailed and helpful answer."
        )
        general_references = fetch_general_references(question)
        crew = Crew(agents=[qa_agent], tasks=[qa_task], verbose=True)
        crew_result = crew.kickoff()
        return {"answer": crew_result.raw, "references": general_references}
    else:
        qa_agent = Agent(
            role="Document Knowledge Assistant",
            goal="Answer user queries using provided document context.",
            backstory="An AI assistant trained to extract answers from documents.",
            verbose=True,
            llm=llm
        )
        qa_task = Task(
            description=f"Use the following context to answer '{question}': {context}",
            agent=qa_agent,
            expected_output="A detailed answer with references."
        )
        crew = Crew(agents=[qa_agent], tasks=[qa_task], verbose=True)
        crew_result = crew.kickoff()
        return {"answer": crew_result.raw, "references": sources}

# Streamlit interface setup
st.title("📚 Personalized Learning Chat Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize message history

if "document_sources" not in st.session_state:
    st.session_state["document_sources"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

spinner_placeholder = st.empty()
with bottom():

    cols = st.columns([6, 1])  # Divide layout into two columns
    prompt = None
# Button stays in the smaller column
    with cols[1]:
        if st.button("🎤"):
            with spinner_placeholder:
                voice_query = voice_input_handler.process_voice_query()
                if voice_query:
                    prompt = voice_query  # Set voice query as prompt

# Input field in the larger column
        with cols[0]:
            text_input = st.chat_input("Ask a question:", key="input-css")
            if text_input:
                prompt = text_input  # Set manual input as prompt

if prompt:
    spinner_placeholder.empty()
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Generating answer..."):
        context = ""
        sources = []

        # Document-based retrieval
        if (st.session_state.get("faiss_indexes") and st.session_state.get("documents")):
            query_embedding = generate_embeddings(prompt, tokenizer, model)
            k = 5
            faiss_index = st.session_state["faiss_indexes"][0]
            distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k)

            relevant_chunks = []
            relevant_sources = []

            for i in indices[0]:
                if i < len(st.session_state["documents"][0]):
                    relevant_chunks.append(st.session_state["documents"][0][i])
                    relevant_sources.append(st.session_state["document_sources"][0])

            unique_sources = list(set(relevant_sources))
            context = " ".join(relevant_chunks)
            sources = unique_sources

        # General fallback
        if not context.strip():
            sources = fetch_general_references(prompt)

        # Generate answer
        result = setup_qa_crew(prompt, context, st.session_state.get("llm"), sources)
        answer = result["answer"]
        references = result["references"]

        # Append assistant response
        references_markdown = "\n".join(["- " + ref for ref in references])
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{answer}\n\n**References:**\n{references_markdown}"
        })

    st.rerun()
