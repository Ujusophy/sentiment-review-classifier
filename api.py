from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load tokenizer and model
model_path = "sentiment_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Setup API
app = FastAPI(title="Sentiment Review Classifier")

# Define input schema
class Review(BaseModel):
    text: str

# Define prediction endpoint
@app.post("/predict")
def predict_sentiment(review: Review):
    inputs = tokenizer(review.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
        label = "Positive" if predicted_class == 1 else "Negative"

    return {
        "text": review.text,
        "predicted_label": label,
        "class_id": predicted_class
    }
