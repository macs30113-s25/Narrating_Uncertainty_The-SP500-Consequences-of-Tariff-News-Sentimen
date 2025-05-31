import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter

# Custom Dataset Class
class SP500Dataset(Dataset):
    def __init__(self, data_path, seq_len=1):
        df = pd.read_csv(data_path)

        # Extract date column
        if 'date' in df.columns:
            self.dates = pd.to_datetime(df['date'].values)
            df = df.drop(columns=['date'])
        else:
            self.dates = None

        self.seq_len = seq_len
        self.feature_scaler = StandardScaler()
        self.label_scaler = StandardScaler()

        # Feature columns and target column
        feature_cols = ['open', 'high', 'low', 'adj_close', 'volume', 'score']
        label_col = ['close']

        features = df[feature_cols].values
        labels = df[label_col].values

        # Standardize features and labels
        self.features = self.feature_scaler.fit_transform(features)
        self.labels = self.label_scaler.fit_transform(labels)

        self.label_scaler_ = self.label_scaler  # Save scaler for inverse transform

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use output from last time step
        return out


# Main function
def train():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)

    # Local training: specify data path
    parser.add_argument('--data-path', type=str, default='sp500_data.csv')

    # Local training: model save path
    parser.add_argument('--model-dir', type=str, default='./model_output')

    args = parser.parse_args()

    # Create model save directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Load data
    dataset = SP500Dataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = LSTMModel(input_size=dataset.features.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # Save model to specified path
    model_path = os.path.join(args.model_dir, "lstm_sp500_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluation
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            pred = model(X_batch)
            all_preds.extend(pred.numpy().flatten())
            all_true.extend(y_batch.numpy().flatten())

    # Inverse transform
    all_preds_unscaled = dataset.label_scaler_.inverse_transform(np.array(all_preds).reshape(-1, 1))
    all_true_unscaled = dataset.label_scaler_.inverse_transform(np.array(all_true).reshape(-1, 1))

    # Calculate evaluation metrics
    mse = mean_squared_error(all_true_unscaled, all_preds_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_true_unscaled, all_preds_unscaled)
    r2 = r2_score(all_true_unscaled, all_preds_unscaled)

    print(f"Evaluation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Get corresponding dates (only valid prediction dates)
    if dataset.dates is not None:
        plot_dates = dataset.dates[dataset.seq_len:]
    else:
        plot_dates = np.arange(len(all_true_unscaled))

    # -----------------------------
    # Plot: True vs Predicted Values
    # -----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(plot_dates, all_true_unscaled, label='True Value', color='blue')
    plt.plot(plot_dates, all_preds_unscaled, label='Predicted Value', color='red', linestyle='--')

    plt.title('True vs Predicted Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    # plt.grid(True)

    plt.savefig('True-vs-Predicted-Values-Over-Time2024.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot error distribution
    errors = all_true_unscaled.flatten() - all_preds_unscaled.flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, color='blue', alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error (True - Predicted)')
    plt.ylabel('Frequency')
    # plt.grid(True)
    plt.savefig('Error-Distribution2024.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Residual plot over time
    residuals = all_true_unscaled.flatten() - all_preds_unscaled.flatten()
    plt.figure(figsize=(14, 6))
    plt.scatter(plot_dates, residuals, color='purple', alpha=0.6, label='Residuals')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error Line')

    plt.title('Residual Plot Over Time')
    plt.xlabel('Date')
    plt.ylabel('Residual (Error)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    # Set date format
    locator = AutoDateLocator()
    formatter = DateFormatter('%Y-%m-%d')
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.savefig('Residual_Plot_Over_Time2024.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    train()