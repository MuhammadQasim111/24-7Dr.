# app.py

import streamlit as st
from transformers import pipeline

# Initialize the BioGPT model using the Hugging Face pipeline
generator = pipeline("text-generation", model="microsoft/BioGPT")

# Streamlit app title and description
st.title("24/7Dr. Health Chatbot")
st.markdown("""
    This is a health chatbot that can provide responses based on the symptoms you describe.
    It uses a medical GPT model to generate responses and help guide your understanding.
    """)

# Initialize session state for conversation history if it does not exist
if 'history' not in st.session_state:
    st.session_state.history = []

# Function to generate chatbot responses using BioGPT
def generate_medical_response(user_input):
    """
    Generates a response using BioGPT model based on user input (symptoms).
    
    Args:
        user_input (str): The symptoms or health-related query from the user.
        
    Returns:
        str: The generated response from the BioGPT model.
    """
    response = generator(user_input,
                         max_length=150,
                         num_return_sequences=1,
                         pad_token_id=50256,
                         truncation=True,
                         temperature=0.7,
                         top_k=50,
                         top_p=0.95)
    return response[0]['generated_text']

def display_conversation_history():
    """Display the conversation history in the app."""
    if st.session_state.history:
        st.subheader("Conversation History")
        for message in st.session_state.history:
            st.write(message)

def main():
    """Main function to run the Streamlit app."""
    
    # Input box for user to describe symptoms
    user_input = st.text_input("Describe your symptoms:")

    # When the 'Ask' button is pressed
    if st.button("Ask"):
        if user_input:  # Check if user input is not empty
            # Store the user's input in the conversation history
            st.session_state.history.append(f"You: {user_input}")

            # Generate the chatbot's response using BioGPT
            bot_response = generate_medical_response(user_input)

            # Store the chatbot's response in the conversation history
            st.session_state.history.append(f"Bot: {bot_response}")

            # Clear the input box after submission (optional for improved UX)
            st.text_input("Describe your symptoms:", "", key="clear_input")

    # Display the conversation history on the Streamlit app
    display_conversation_history()

if __name__ == "__main__":
    main()
