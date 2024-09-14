import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

# Initialize Hugging Face model
qa_pipeline = pipeline("question-answering")

# Function to fetch data from a website
def fetch_web_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        return str(e)

# Function to get response from Hugging Face model
def get_hf_response(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Create the Streamlit app
st.title('Web Data-Based Conversational Agent')

url = st.text_input("Enter the URL of the webpage:")
user_input = st.text_input("Ask a question:")

if st.button('Submit'):
    if url and user_input:
        # Fetch data from the web
        context = fetch_web_data(url)
        
        # Get response from Hugging Face
        hf_response = get_hf_response(user_input, context)

        st.write("**Fetched Data (excerpt):**")
        st.write(context[:1000])  # Display a snippet of the fetched data
        
        st.write("**AI Response:**")
        st.write(hf_response)
    else:
        st.write("Please enter a URL and a question.")