from transformers import CLIPProcessor, CLIPModel, RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch
from PIL import Image
import os

# CLIP model 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# RAG model 
rag_model_name = "facebook/rag-sequence-nq"
rag_tokenizer = RagTokenizer.from_pretrained(rag_model_name)
retriever = RagRetriever.from_pretrained(
    rag_model_name, 
    index_name="legacy",
    passages_path="data/documents",  # Path to the documents
    use_dummy_dataset=True
)
rag_model = RagSequenceForGeneration.from_pretrained(rag_model_name, retriever=retriever)

def load_image(image_path):
    return Image.open(image_path)


def image_text_similarity(image, texts):
    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return logits_per_image, probs


def analyze_image_and_text_rag(image_path, question):

    image = load_image(image_path)
    inputs = rag_tokenizer(question, return_tensors="pt")
    input_ids = inputs['input_ids']
    generated = rag_model.generate(input_ids=input_ids)
    response = rag_tokenizer.decode(generated[0], skip_special_tokens=True)
    
    return response


def handle_user_input(image_path, text_input):
    image = load_image(image_path)
    
    if isinstance(text_input, list):
        # Option 1: Image + List of sentences
        logits, probs = image_text_similarity(image, text_input)
        print("Image-text similarity scores (logits_per_image):", logits)
        print("Probabilities (softmaxed):", probs)
    elif isinstance(text_input, str):
        # Option 2: Image + Single sentence
        response = analyze_image_and_text_rag(image_path, text_input)
        print("Generated Description:", response)
    else:
        raise ValueError("Invalid input: text_input must be either a list of sentences or a single sentence string.")


if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  
    text_input_list = ["a rash", "a lesion", "a cut"]
    handle_user_input(image_path, text_input_list)

    text_input_sentence = "Describe the condition and suggest if a doctor visit is necessary."
    handle_user_input(image_path, text_input_sentence)
