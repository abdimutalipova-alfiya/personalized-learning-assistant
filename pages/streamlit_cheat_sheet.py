import streamlit as st
import os
from crewai import Crew, Task, Agent
from transformers import AutoTokenizer, AutoModel
import numpy as np
from PyPDF2 import PdfReader  # Ensure PyPDF2 is installed
from docx import Document  # To handle DOCX files
from io import BytesIO
from streamlit_app import configure_llm
from crewai import Crew, Task, Agent, LLM


st.set_page_config(page_title="Cheat Sheet Tool", page_icon="📊")
selected_llm = st.sidebar.selectbox(
    "Select LLM", 
    ["Groq API", "Gemini"],
    help="Choose the Language Model for your queries"
)


if "uploaded_files_cheatsheet" not in st.session_state:
    st.session_state["uploaded_files_cheatsheet"] = None
if "selected_llm" not in st.session_state:
    st.session_state["selected_llm"] = "Groq API"
if "llm" not in st.session_state:
    st.session_state["llm"] = None

st.session_state["selected_llm"] = selected_llm
st.session_state["llm"] = configure_llm(st.session_state["selected_llm"])

uploaded_files_cheatsheet = st.sidebar.file_uploader("Upload PDFs to Generate Cheatsheet", accept_multiple_files=True, type="pdf")
if uploaded_files_cheatsheet:
    with st.spinner("Processing documents..."):
        st.session_state["uploaded_files_cheatsheet"]=uploaded_files_cheatsheet
        st.sidebar.success("Documents processed successfully!")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the tokenizer and model for embeddings (once)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def create_docx_from_text(text):
    doc = Document()
    doc.add_paragraph(text)
    return doc

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


def setup_cheetsheet_crew(context, llm):
    goal = f"Generate a concise, one-page summary. Focus on the most important insights and actionable takeaways."
    cheatsheet_agent = Agent(
        role="Summarization Specialist",
        goal=goal,
        backstory="You are an expert at summarizing large texts into clear, concise, and actionable cheatsheets.",
        verbose=True,
        llm=llm
    )
    cheatsheet_task = Task(
        description=f"Summarize the following context into a single-page cheatsheet: {context}",
        agent=cheatsheet_agent,
        expected_output="A well-structured, easy-to-read cheatsheet capturing the key takeaways and insights."
    )

    crew = Crew(agents=[cheatsheet_agent], tasks=[cheatsheet_task], verbose=True)
    crew_result = crew.kickoff()
    return crew_result.raw


# Streamlit interface
st.title("📝 AI-Powered Cheatsheet Generator")

# Introductory text
st.write("""
Welcome to the Cheatsheet Generator! Upload a document (PDF file) and click the button to generate a concise, one-page cheatsheet summarizing its content.
""")

# Process session-based documents
if st.session_state.uploaded_files_cheatsheet is not None:
    documents = st.session_state.uploaded_files_cheatsheet

    text = ""
    for doc in documents:
        reader = PdfReader(doc)
        text += "\n".join([page.extract_text() for page in reader.pages])


    if st.button("Generate Cheatsheet"):
        with st.spinner("Generating cheatsheet..."):
            chunks = chunk_text_by_sentences(text)
            st.session_state.documents_cheatsheet = chunks

            context = " ".join(chunks[:5])
                # Generate cheatsheet using CrewAI
            try:
                cheatsheet = setup_cheetsheet_crew(context, st.session_state.get("llm"))
                st.subheader("📄 Your Cheatsheet")
                st.write(cheatsheet)

                docx_file = create_docx_from_text(cheatsheet)

                    # Save DOCX file to memory
                byte_io=BytesIO()
                docx_file.save(byte_io)
                byte_io.seek(0)


                    # Option to download the cheatsheet
                st.download_button(
                    label="Download Cheatsheet (docx)",
                    data=byte_io,
                    file_name="cheatsheet.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"An error occurred while generating the cheatsheet: {e}")
else:
    st.info("Please upload documents through the sidebar to get started.")
