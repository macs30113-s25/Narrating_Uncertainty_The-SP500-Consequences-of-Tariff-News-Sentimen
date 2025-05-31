import torch
from model import LSTMModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)

model_path = 'lstm_sp500_model.pth'
model = load_model(model_path)

# Data preprocessing function (keep date)
def preprocess_data(data_path):
    df = pd.read_csv(data_path)

    # Save date column for plotting
    dates = df['date'].values if 'date' in df.columns else None

    # Drop date column (if exists)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    # Make sure column names exist (case sensitive! Modify according to your CSV)
    feature_cols = ['open', 'high', 'low', 'adj_close', 'volume', 'score']
    label_col = ['close']

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    features = df[feature_cols].values
    labels = df[label_col].values

    # Standardization
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()

    features_scaled = feature_scaler.fit_transform(features)
    labels_scaled = label_scaler.fit_transform(labels)

    return features_scaled, labels_scaled, label_scaler, dates

# Build time series data
def create_sequences(data_with_dates):
    features_scaled, labels_scaled, dates = data_with_dates

    # Convert string dates to datetime objects
    dates = pd.to_datetime(dates)

    xs, ys, date_list = [], [], []
    seq_len = 1
    for i in range(len(features_scaled) - seq_len):
        x = features_scaled[i:i + seq_len]
        y = labels_scaled[i + seq_len]
        xs.append(x)
        ys.append(y)
        date_list.append(dates[i + seq_len]) 
    return np.array(xs), np.array(ys), np.array(date_list)

# Load data
data_path = 'sp500_2020_q1_news.csv'

# Note: preprocess_data returns dates as well
features_scaled, labels_scaled, label_scaler, dates = preprocess_data(data_path)

# Build sequences and get corresponding dates
X_seq, y_true_scaled, date_list = create_sequences((features_scaled, labels_scaled, dates))

X_tensor = torch.tensor(X_seq, dtype=torch.float32)

# Model prediction
model.eval()
with torch.no_grad():
    predictions_scaled = model(X_tensor).numpy()


# Inverse standardization
predictions_unscaled = label_scaler.inverse_transform(predictions_scaled)
y_true_unscaled = label_scaler.inverse_transform(y_true_scaled.reshape(-1, 1))

# -----------------------------
# Model evaluation
mse = mean_squared_error(y_true_unscaled, predictions_unscaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_unscaled, predictions_unscaled)
r2 = r2_score(y_true_unscaled, predictions_unscaled)

print(f"Evaluation on sp500_2020_q1_news.csv:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Calculate errors
errors = y_true_unscaled - predictions_unscaled

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, color='blue', alpha=0.7)
plt.title('Error Distribution')
plt.xlabel('Prediction Error (True - Predicted)')
plt.ylabel('Frequency')

# Save image
plt.savefig('Error-Distribution2020.png', dpi=300, bbox_inches='tight')  # dpi controls resolution, bbox_inches='tight' prevents label cutoff
plt.close()  # Close to free memory

# Plot: True vs Predicted values
plt.figure(figsize=(14, 6))
plt.plot(date_list, y_true_unscaled, label='True Value', color='blue')
plt.plot(date_list, predictions_unscaled, label='Predicted Value', color='red', linestyle='--')

plt.title('True vs Predicted Values Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)  # Rotate dates to avoid overlap
plt.tight_layout()
plt.legend()

# Save image
plt.savefig('True-vs-Predicted-Values-Over-Time2020.png', dpi=300, bbox_inches='tight')
plt.close()  # Close to free memory

# Residual plot by date
residuals = y_true_unscaled.flatten() - predictions_unscaled.flatten()

plt.figure(figsize=(14, 6))
plt.scatter(date_list, residuals, color='purple', alpha=0.6, label='Residuals')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error Line')

plt.title('Residual Plot Over Time')
plt.xlabel('Date')
plt.ylabel('Residual (Error)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.grid(True)

# Set date format
from matplotlib.dates import AutoDateLocator, DateFormatter
locator = AutoDateLocator()
formatter = DateFormatter('%Y-%m-%d')
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)

# Save image
plt.savefig('Residual_Plot_Over_Time2020.png', dpi=300, bbox_inches='tight')
plt.close()
