import torch
from transformers import AutoModelForCausalLM
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoProcessor
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image


processor = AutoProcessor.from_pretrained("sezenkarakus/image-GIT-description-model-v3")
model = AutoModelForCausalLM.from_pretrained("sezenkarakus/image-GIT-description-model-v3")

image2 = Image.open("lesion.jpg")

inputs = processor(images=image2, return_tensors="pt", padding=True)
outputs = model.generate(**inputs,max_new_tokens=200)
generated_descriptions = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated Description:", generated_descriptions)