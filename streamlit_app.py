import streamlit as st
from groq import Groq

# Initialize the Groq client using the API key stored in Streamlit's secrets
client = Groq(api_key="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")

# Set the system prompt
system_prompt = {
    "role": "system",
    "content": "You are a helpful assistant. You reply with very short answers."
}

# Initialize or load the chat history in Streamlit's session state (for the backend only)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [system_prompt]

# Initialize the input field value in session state to track clearing
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Title for the Streamlit app
st.title("Chat with Groq Assistant")

def clear_text():
    st.session_state.user_input = "" 

# Text input for user query
user_input = st.text_input("You:", st.session_state.user_input, placeholder="Type your message here...", on_change=clear_text)

# Function to generate a response from Groq and update the chat history
def get_groq_response(input_text):
    # Add the user's message to the chat history
    st.session_state.chat_history.append({"role": "user", "content": input_text})
    
    # Generate the response using the Groq client
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=st.session_state.chat_history,
        max_tokens=100,
        temperature=1.2
    )
    
    # Extract the assistant's response from the API result
    assistant_message = response.choices[0].message.content
    
    # Add the assistant's response to the chat history (in session state only)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message

# If the user submits a message, get the assistant's response
if user_input:
    assistant_response = get_groq_response(user_input)
    
    # Display only the latest user input and assistant response
    st.write("You:", user_input)
    st.write("Assistant:", assistant_response)

