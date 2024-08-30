import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()  # Load environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=True)
        return response
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")

st.header("Chat Bot")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input = st.text_input("Input:")
submit = st.button("Ask the question")

if submit and input:
    with st.spinner("Generating response..."):
        response = get_gemini_response(input)
        if response:
            # Add user query and response to session state chat history
            st.session_state['chat_history'].append(("You", input))
            st.subheader("The Response is")
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("Bot", chunk.text))

st.subheader("Chat History")
with st.expander("Expand to see chat history"):
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")
