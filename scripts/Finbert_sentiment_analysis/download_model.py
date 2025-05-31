# download_model.py
from transformers import BertTokenizer, BertForSequenceClassification
import os

def download_finbert_model():
    model_name = "ProsusAI/finbert"
    model_path = os.path.join(os.path.dirname(__file__), "finbert_model")

    os.makedirs(model_path, exist_ok=True)

    print(f"🔽 Downloading model from: {model_name}")
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    print("💾 Saving model locally with safe serialization...")
    model.save_pretrained(model_path, safe_serialization=True)
    tokenizer.save_pretrained(model_path)

    print(f"✅ Model saved to: {model_path}")

if __name__ == "__main__":
    download_finbert_model()
