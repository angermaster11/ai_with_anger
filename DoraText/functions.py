from transformers import pipeline
import torch 

import torch

device = 0 if torch.cuda.is_available() else -1


model = pipeline("text-generation")
def generate_text(prompt,max_length=50):
    result = model(prompt,max_length=max_length)
    return result[0]['generated_text']

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)

def generate_summary(text,min_length,max_length):
    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=True,
        truncation=True)
    return result[0]['summary_text']

ner_pipeline = pipeline("ner", grouped_entities=True, device=device)
def generate_entities(text):
    entities = ner_pipeline(text)
    return entities