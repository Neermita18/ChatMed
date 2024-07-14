
import torch
from transformers import AutoModelForCausalLM
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# pipeline = pipeline("text-classification", model="utkarshiitr/medicalchatbot_2")
# data=pipeline("I feel scratchy in my throat. What should i do??")
# print(data)
tokenizer = AutoTokenizer.from_pretrained("utkarshiitr/medicalchatbot")
model = AutoModelForSequenceClassification.from_pretrained("utkarshiitr/medicalchatbot")

inputs = tokenizer("fever,cough", return_tensors="pt")
outputs = model(**inputs)
print(outputs)

