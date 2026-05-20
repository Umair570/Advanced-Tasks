import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./fine_tuned_bert_news"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    return {model.config.id2label[i]: float(probabilities[i]) for i in range(len(probabilities))}

interface = gr.Interface(fn=classify_news, inputs="textbox", outputs="label")

# Launching with a public tunnel for cloud access
interface.launch(share=True)
