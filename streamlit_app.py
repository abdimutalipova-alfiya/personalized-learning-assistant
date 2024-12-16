import streamlit as st
from document_processor import DocumentProcessor
import os
from crewai import LLM

def process_query(query):
    """
    This function will be called when a voice query is transcribed
    It should add the query to your messages and trigger a rerun
    """
    st.session_state.messages.append({"role": "user", "content": query})
    st.rerun()


# Ensure environment variables are set
groq_api_key = st.secrets["GROQ_API_KEY"]
serp_api_key = st.secrets["SERPAPI_API_KEY"]
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"] # For huggingface models
gemini_api_key = st.secrets["GEMINI_API_KEY"]



if not groq_api_key or not serp_api_key:
    st.error("Please set GROQ_API_KEY and SERPAPI_API_KEY environment variables.")
    st.stop()

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
        
        # elif llm_name == "HuggingFace":
        #     if not hf_token:
        #         st.warning("HUGGINGFACEHUB_API_TOKEN not found. Falling back to Groq LLM.")
        #         return LLM(
        #             model="groq/llama-3.1-70b-versatile",
        #             api_key=groq_api_key
        #         )
        #     return LLM(
        #         provider="huggingface",
        #         model="huggingface/EleutherAI/gpt-neox-20b",
        #         api_key=hf_token
        #     )
        
        elif llm_name == "Gemini":
            if not gemini_api_key:
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


home = st.Page("streamlit_pages/streamlit_home.py", title="Home", icon="👋", )
qa = st.Page("streamlit_pages/streamlit_qa.py", title="Question Answering Tool", icon="👋")
cheatsheet= st.Page("streamlit_pages/streamlit_cheat_sheet.py", title="CheatSheet Tool", icon="👋")


pg = st.navigation(
            {
            "Home": [home],
            "Tools": [qa, cheatsheet]}, expanded=True
        ) 



if pg==qa:
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

elif pg==cheatsheet:
    selected_llm = st.sidebar.selectbox(
    "Select LLM", 
    ["Groq API", "Gemini"],
    help="Choose the Language Model for your queries"
)
    st.session_state["selected_llm"] = selected_llm
    st.session_state["llm"] = configure_llm(st.session_state["selected_llm"])

    uploaded_files_cheatsheet = st.sidebar.file_uploader("Upload PDFs to Generate Cheatsheet", accept_multiple_files=True, type="pdf")
    if uploaded_files_cheatsheet:
        with st.spinner("Processing documents..."):
            st.session_state["uploaded_files_cheatsheet"]=uploaded_files_cheatsheet
            st.sidebar.success("Documents processed successfully!")


# Run the navigation
pg.run()

 
