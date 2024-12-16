__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import  LLM

st.set_page_config(
        page_title="Welcome to My App",
        page_icon="🌟",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Entry page content
st.title("🌟 Welcome to Personalized Learning Assistant")
st.subheader("Your one-stop solution for personalized learning and AI-powered tools.")

st.markdown("""
    Welcome to **Personalized Learning Assistant**, where you can:
    - 🔍 Explore educational data and insights tailored to your learning journey.
    - 💡 Generate concise summaries of course materials and documents.
    - 🤖 Leverage AI-powered tools to answer questions, generate cheat sheets, and enhance your learning experience.

   Whether you're a student, educator, or lifelong learner, we’re here to help you make the most of your learning experience.
    Let's make your educational journey as seamless and productive as possible!
    """)


    # Footer
st.divider()
st.markdown("Made with ❤️ using Streamlit.")

# Ensure environment variables are set
groq_api_key = st.secrets["GROQ_API_KEY"]
serp_api_key = st.secrets["SERPAPI_API_KEY"]
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"] # For huggingface models
gemini_api_key = st.secrets["GEMINI_API_KEY"]


if not groq_api_key or not serp_api_key:
    st.error("Please set GROQ_API_KEY and SERPAPI_API_KEY environment variables.")
    st.stop()


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


