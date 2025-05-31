import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib  # For loading the saved scaler

# Model definition, input_size=6, consistent with training
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def model_fn(model_dir):
    # 1. Load the model
    model = LSTMModel(input_size=6)
    model_path = os.path.join(model_dir, "lstm_sp500_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 2. Load the normalization parameters used during training
    scaler_path = os.path.join(model_dir, "feature_scaler.save")
    if os.path.exists(scaler_path):
        feature_scaler = joblib.load(scaler_path)
    else:
        raise RuntimeError("Feature scaler file not found!")

    # Return both model and scaler
    return {"model": model, "scaler": feature_scaler}


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        inputs = np.array(data['inputs'], dtype=np.float32)

        # Input must be (batch_size, seq_len, 6)
        if inputs.ndim == 2 and inputs.shape[1] == 6:
            inputs = inputs.reshape(inputs.shape[0], 1, 6)  # seq_len=1 example
        else:
            raise ValueError(f"Input shape must be (batch_size, 6), got {inputs.shape}")
        return inputs
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_and_scaler):
    model = model_and_scaler['model']
    scaler = model_and_scaler['scaler']

    # Normalize input
    batch_size, seq_len, feature_dim = input_data.shape
    reshaped = input_data.reshape(-1, feature_dim)  # (batch_size * seq_len, 6)
    scaled = scaler.transform(reshaped)
    scaled = scaled.reshape(batch_size, seq_len, feature_dim)

    # Convert to tensor
    inputs = torch.tensor(scaled, dtype=torch.float32)

    # Model inference
    with torch.no_grad():
        outputs = model(inputs)

    # Return as numpy array
    return outputs.numpy().tolist()


def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return json.dumps({"predictions": prediction})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
