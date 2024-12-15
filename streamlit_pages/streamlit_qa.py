import streamlit as st
import os
from crewai import Crew, Task, Agent, LLM
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import requests
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Load the tokenizer and model for embeddings (should be done once)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def fetch_general_references(question):
    """
    Fetch references for general knowledge questions using SerpAPI.
    Returns a list of reference links.
    """
    serp_api_key = os.getenv("SERPAPI_API_KEY")  # Ensure this environment variable is set
    if not serp_api_key:
        return ["SerpAPI key not found. Please set SERPAPI_API_KEY environment variable."]
    
    search_url = "https://serpapi.com/search"
    params = {
        "q": question,
        "hl": "en",
        "gl": "us",
        "num": 3,  # Limit to 3 references
        "api_key": serp_api_key
    }

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        search_results = response.json()
        references = [
            result["link"] for result in search_results.get("organic_results", [])
            if "link" in result
        ]
        return references if references else ["No references found."]
    except Exception as e:
        return [f"Error fetching references: {e}"]

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

def setup_qa_crew(question, context, llm, sources):
    # If context is empty or very short, create a more general agent
    if not context or len(context.strip()) < 50:
        qa_agent = Agent(
            role="Knowledgeable General Assistant",
            goal="Provide comprehensive and helpful answers to user questions using general knowledge.",
            backstory="You are an AI assistant capable of answering a wide range of questions using your broad knowledge base.",
            verbose=True,
            llm=llm
        )

        

        qa_task = Task(
            description=f"Answer the user's question: '{question}' comprehensively and helpfully.",
            agent=qa_agent,
            expected_output="A detailed, informative answer drawing from general knowledge with a reference to the knowledge base used (e.g., 'LLM Model XYZ')"
        )

          # Fetch general references using SerpAPI
        general_references = fetch_general_references(question)

        crew = Crew(
            agents=[qa_agent],
            tasks=[qa_task],
            verbose=True
        )
        crew_result = crew.kickoff()

        return {
            "answer": crew_result.raw,
            "references": general_references
        }
    
    else:
        # Original implementation for document-based queries
        qa_agent = Agent(
            role="Knowledge Assistant",
            goal=f"Provide comprehensive and contextually relevant answers to the user's question: '{question}'.",
            backstory="You are a highly knowledgeable assistant trained to retrieve and synthesize information from provided documents.",
            verbose=True,
            llm=llm
        )

        qa_task = Task(
            description=f"Extract and provide relevant information for: '{question}'. Use the following context: {context}",
            agent=qa_agent,
            expected_output=f"Detailed answer with references to sources: {sources}"
        )
    crew = Crew(
        agents=[qa_agent],
        tasks=[qa_task],
        verbose=True
    )

    crew_result = crew.kickoff()
    
    return {
            "answer": crew_result.raw,
            "references": sources
        }


st.title("📚 Personalized Learning Chat Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ask user to enter a question
prompt = st.chat_input("Ask a question:")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

if prompt:
    with st.spinner("Generating answer..."):
        if (st.session_state.get("faiss_indexes") and 
            st.session_state["faiss_indexes"] is not None and
            len(st.session_state["faiss_indexes"]) > 0 and
            st.session_state.get("documents") and
            st.session_state["documents"] is not None and
            len(st.session_state["documents"]) > 0):

            # If documents are available, proceed with document-based QA
            query_embedding = generate_embeddings(prompt, tokenizer, model)
            # Retrieve top relevant chunks
            k = 5
            faiss_index = st.session_state["faiss_indexes"][0]
            distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k)

            relevant_chunks = [st.session_state["documents"][0][i] for i in indices[0]]
            relevant_sources = [st.session_state["document_sources"][0][i] for i in indices[0]]

            # Deduplicate references
            unique_sources = list(set(relevant_sources))
            context = " ".join(relevant_chunks)

            with st.chat_message("assistant"):
                # Run the QA crew to get final answer
                result = setup_qa_crew(prompt, context, st.session_state["llm"], unique_sources)
                answer = result["answer"]
                references = result["references"]
                st.write(f"**Answer:** {answer}")
                st.write(f"**References:** {', '.join(references)}")
                st.session_state.messages.append({"role": "assistant", "content": f"{answer}\n\nReferences: {', '.join(references)}"})
        else:
            # If no documents are available, answer the question using general knowledge
            context = ""
            sources = []
            with st.chat_message("assistant"):
                result = setup_qa_crew(prompt, context, st.session_state["llm"], sources)
                answer = result["answer"]
                references = result["references"]
                st.write(f"**Answer:** {answer}")
                st.write("**References:**")
                for ref in references:
                    st.markdown(f"- [{ref}]({ref})")
                st.session_state.messages.append({"role": "assistant", "content": f"{answer}\n\nReferences:\n" + "\n".join(references)})
                
