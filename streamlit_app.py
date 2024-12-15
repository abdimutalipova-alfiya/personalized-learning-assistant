import streamlit as st
from document_processor import DocumentProcessor, DocumentContext
import os
from crewai import LLM

# Import the new LLM providers
from huggingface_llm import HuggingFaceLLMProvider
from gemini_llm import GeminiLLMProvider

# Ensure environment variables are set
groq_api_key = os.getenv("GROQ_API_KEY")
serp_api_key = os.getenv("SERPAPI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # For huggingface models
gemini_api_key = os.getenv("GEMINI_API_KEY")



if not groq_api_key or not serp_api_key:
    st.error("Please set GROQ_API_KEY and SERPAPI_API_KEY environment variables.")
    st.stop()

# Initialize session_state keys if not present
if "faiss_indexes" not in st.session_state:
    st.session_state["faiss_indexes"] = None
if "documents" not in st.session_state:
    st.session_state["documents"] = None
if "document_sources" not in st.session_state:
    st.session_state["document_sources"] = None
if "selected_llm" not in st.session_state:
    st.session_state["selected_llm"] = "Groq API"
if "llm" not in st.session_state:
    st.session_state["llm"] = None

# Enhanced LLM Selection with Validation
st.session_state["selected_llm"] = st.sidebar.selectbox(
    "Select LLM", 
    ["Groq API", "OpenAI", "HuggingFace", "Gemini"],
    help="Choose the Language Model for your queries"
)



# Comprehensive LLM Configuration
def configure_llm(llm_name):
    """
    Configure and return the appropriate LLM based on the selection.
    """
    try:
        if llm_name == "Groq API":
            return LLM(
                model="groq/llama-3.1-70b-versatile", 
                api_key=groq_api_key
            )
        
        elif llm_name == "OpenAI":
            if not openai_api_key:
                st.warning("OPENAI_API_KEY not found. Falling back to Groq LLM.")
                return LLM(
                    model="groq/llama-3.1-70b-versatile",
                    api_key=groq_api_key
                )
            return LLM(
                model="gpt-4",
                api_key=openai_api_key
            )
        
        elif llm_name == "HuggingFace":
            if not hf_token:
                st.warning("HUGGINGFACEHUB_API_TOKEN not found. Falling back to Groq LLM.")
                return LLM(
                    model="groq/llama-3.1-70b-versatile",
                    api_key=groq_api_key
                )
            return LLM(
                provider="huggingface",
                model="huggingface/EleutherAI/gpt-neox-20b",
                api_key=hf_token
            )
        
        elif llm_name == "Gemini":
            if not gemini_api_key:
                st.warning("GEMINI_API_KEY not found. Falling back to Groq LLM.")
                return LLM(
                    model="groq/llama-3.1-70b-versatile",
                    api_key=groq_api_key
                )
            return LLM(
                provider="gemini",
                model="gemini/gemini-pro",
                api_key=gemini_api_key
            )
    
    except Exception as e:
        st.error(f"Error configuring {llm_name}: {e}")
        return LLM(
            model="groq/llama-3.1-70b-versatile",
            api_key=groq_api_key
        )
# Set the LLM in session state
st.session_state["llm"] = configure_llm(st.session_state["selected_llm"])


# Upload PDFs and YouTube links
uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf", key="uploaded_files")
youtube_links = st.sidebar.text_area("Enter YouTube Links (comma separated)")

if uploaded_files or youtube_links:
    with st.spinner("Processing documents and YouTube links..."):
        youtube_links = youtube_links.split(",") if youtube_links else []
        faiss_indexes, documents, document_sources = DocumentProcessor.chunk_and_embed_documents(uploaded_files, youtube_links)
        st.session_state["faiss_indexes"] = faiss_indexes
        st.session_state["documents"] = documents
        st.session_state["document_sources"] = document_sources

        if st.session_state["document_sources"] and len(st.session_state["document_sources"]) > 0:
            st.write(st.session_state["document_sources"])
        else:
            st.write("No documents or YouTube links processed.")
        st.sidebar.success("Documents and YouTube links processed successfully!")
else:
    st.write("Upload PDFs or provide YouTube links to begin.")

home = st.Page("streamlit_pages/streamlit_home.py", title="Home", icon="👋", )
qa = st.Page("streamlit_pages/streamlit_qa.py", title="Question Answering Tool", icon="👋")
cheatsheet= st.Page("streamlit_pages/streamlit_cheat_sheet.py", title="CheatSheet Tool", icon="👋")

pg = st.navigation(
            {
            "Home": [home],
            "Tools": [qa, cheatsheet]}, expanded=True
        ) 

pg.run()