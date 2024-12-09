import streamlit as st
from crewai import Agent, Task, Crew, Process
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from crewai import LLM
import os
import faiss
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



st.title("Personalized Learning Assistant")

# Sidebar for LLM selection and file upload
selected_llm = st.sidebar.selectbox("Select LLM", ["Groq API", "OpenAI", "Other"])
uploaded_files = st.sidebar.file_uploader("Upload PDF documents", accept_multiple_files=True, type="pdf")

# Main chat interface
chat_history = []
user_input = st.chat_input("Ask a question about your course materials")

class ContentIngestionAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Content Ingestion Specialist",
            goal="Process and store course materials efficiently",
            backstory="Expert in document processing and information retrieval",
            llm=llm
        )

    def ingest_documents(self, documents):
        parsed_docs = []
        llama_parser = LlamaParse(result_type="text", api_key=os.getenv("LLAMA_CLOUD_API_KEY"))
        # Use LlamaIndex to load and process documents
        for document in documents:
            doc_content = llama_parser.load_data(document.getvalue(), extra_info={"file_name": document.name})
            parsed_docs.extend([Document(text=page.text, extra_info={"file_name": document.name}) for page in doc_content])

        # Create FAISS index
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1536))
        Settings.vector_store = vector_store  
        d = 1536  # Dimension of embeddings
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        Settings.vector_store = vector_store  


        # Create VectorStoreIndex
        index = VectorStoreIndex.from_documents(parsed_docs)
        return index
    
class QuestionAnsweringAgent(Agent):
    def __init__(self, llm, index):
        super().__init__(
            role="Question Answering Specialist",
            goal="Provide accurate and contextually relevant answers",
            backstory="Expert in information retrieval and natural language processing",
            llm=llm
        )
        self.index = index

    def answer_question(self, question):
        query_engine = self.index.as_query_engine()
        response = query_engine.query(question)
        return response.response
    
def setup_crew(llm, index):
    ingestion_agent = ContentIngestionAgent(llm)

    ingest_task = Task(
        description="Process and store uploaded documents",
        agent=ingestion_agent
    )
    qa_agent = QuestionAnsweringAgent(llm, index)

    qa_task = Task(
        description="Answer user questions based on stored information",
        agent=qa_agent
    )

    crew = Crew(
        agents=[ingestion_agent, qa_agent],
        tasks=[ingest_task, qa_task],
        process=Process.sequential,
        verbose=True
    )

    return crew


if uploaded_files:
    with st.spinner("Processing uploaded documents..."):
        llm = LLM(
    model="groq/llama-3.1-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)  # Initialize Groq API client
        index = ContentIngestionAgent(llm).ingest_documents(uploaded_files)
        crew = setup_crew(llm, index)

    if user_input:
        with st.spinner("Generating answer..."):
            response = crew.kickoff(inputs={'question': user_input})
            st.chat_message("assistant").write(response)
            chat_history.append(("user", user_input))
            chat_history.append(("assistant", response))

    # Display chat history
    for role, message in chat_history:
        st.chat_message(role).write(message)
else:
    st.warning("Please upload course materials to begin.")

    