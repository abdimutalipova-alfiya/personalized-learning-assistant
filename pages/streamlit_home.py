import streamlit as st

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
