
import torch
from transformers import AutoModelForCausalLM
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoProcessor

# pipeline = pipeline("text-classification", model="utkarshiitr/medicalchatbot_2")
# data=pipeline("I feel scratchy in my throat. What should i do??")
# print(data)
# tokenizer = AutoTokenizer.from_pretrained("utkarshiitr/medicalchatbot")
# model = AutoModelForSequenceClassification.from_pretrained("utkarshiitr/medicalchatbot")

# inputs = tokenizer("fever,cough", return_tensors="pt")
# outputs = model(**inputs)
# print(outputs)
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image from URL using PIL
image = Image.open("lesion.jpg")

# Text descriptions
texts = ["a rash","a lesion",  "a cut"]

# Process text and image inputs
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# Model inference
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # Image-text similarity scores
probs = logits_per_image.softmax(dim=1)      # Softmax to get label probabilities

print("Image-text similarity scores (logits_per_image):", logits_per_image)
print("Probabilities (softmaxed):", probs)

