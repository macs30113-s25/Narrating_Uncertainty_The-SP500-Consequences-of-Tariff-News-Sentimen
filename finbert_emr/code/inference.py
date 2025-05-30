import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def model_fn(model_dir):

    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    return {"model": model, "tokenizer": tokenizer, "device": device}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data["text"]
    else:
        raise ValueError(f"不支持的 content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]
    
    inputs = tokenizer(
        input_data,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    return {
        "negative": float(probs[0]),
        "neutral": float(probs[1]),
        "positive": float(probs[2])
    }

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"不支持的 content type: {response_content_type}")