import streamlit as st
from PIL import Image
import model1  # CLIP functionality
import model2  # RAG functionality
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Import the text-to-text chatbot setup from model1.py
from model1 import ask_text_chatbot

st.title("Medical Image Analysis Chatbot")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Text-to-Text Chatbot")
    text_question = st.text_area("Enter your question:")
    if st.button("Ask", key="text-chatbot"):
        text_response = ask_text_chatbot(text_question)
        st.write("Response:", text_response)

with col2:
    st.header("Image-to-Text Chatbot")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        text_input = st.text_area("Enter text input (comma-separated for list of sentences or a single sentence):")

        if st.button("Analyze with CLIP", key="clip-analyze"):
            if "," in text_input:
                # Image + List of sentences (CLIP)
                sentences = [sentence.strip() for sentence in text_input.split(",")]
                logits, probs = model1.image_text_similarity(image, sentences)
                st.write("Image-text similarity scores (logits_per_image):", logits)
                st.write("Probabilities (softmaxed):", probs)
            else:
                st.write("Please enter a list of sentences separated by commas for CLIP analysis.")

        if st.button("Analyze with RAG", key="rag-analyze"):
            if "," not in text_input:
                # Image + Single sentence (RAG)
                rag_response = model2.analyze_image_and_text_rag(image, text_input)
                st.write("RAG Model Response:", rag_response)
            else:
                st.write("Please enter a single sentence for RAG analysis.")
