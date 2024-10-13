import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import json

load_dotenv()  # Load environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    try:
        response = chat.send_message(content=question, stream=True)
        return response
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

def load_chat_history(user_id):
    """Loads chat history from a JSON file."""
    chat_history_file = f"chat_history_{user_id}.json"
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r") as f:
            return json.load(f)
    else:
        return []

def save_chat_history(user_id, chat_history):
    """Saves chat history to a JSON file."""
    with open(f"chat_history_{user_id}.json", "w") as f:
        json.dump(chat_history, f)

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")

st.header("Gemini LLM Application")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_id = "user1"  # Replace with actual user identification
chat_history = load_chat_history(user_id)
st.session_state['chat_history'] = chat_history

understanding = ["High", "Middle", "Low"]
level_of_understanding = st.selectbox("Select your Level of understanding :", understanding)

experience = ["new", "old"]
ex = st.selectbox("Select your  :", experience)

input = st.text_input("Input:")
submit = st.button("Ask the question")

if submit and input:
    with st.spinner("Generating response..."):
        response = get_gemini_response(''' this is my level of understanding {level_of_understanding} and i am {experience} to this topic i had a doubt
                                                                so answer the question on Basis of my level of understanding and my experience and my question is '''+input)
        if response:
            # Add user query and response to session state chat history
            st.session_state['chat_history'].append(("You", input))
            st.subheader("The Response is")
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("Bot", chunk.text))

        # Save chat history after each interaction
        save_chat_history(user_id, st.session_state['chat_history'])

st.subheader("Chat History")
with st.expander("Expand to see chat history"):
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")
