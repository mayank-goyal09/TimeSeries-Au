import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import joblib
import os

# Configuration
WINDOW_SIZE = 60
HORIZON = 30
EPOCHS = 50
BATCH_SIZE = 32
MODEL_PATH = "gold_lstm_multioutput.keras"
SCALER_PATH = "price_scaler.pkl"
DATA_FILE = "Gold Price.csv"

def train_model():
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    prices = df['Price'].values.reshape(-1, 1)
    
    # Scale Data
    print("Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # Create Sequences
    X, y = [], []
    for i in range(len(scaled_prices) - WINDOW_SIZE - HORIZON + 1):
        X.append(scaled_prices[i : i + WINDOW_SIZE])
        y.append(scaled_prices[i + WINDOW_SIZE : i + WINDOW_SIZE + HORIZON])
        
    X = np.array(X)
    y = np.array(y)
    
    # Reshape y for Dense output (samples, horizon)
    y = y.reshape(y.shape[0], HORIZON)
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Build Model
    print("Building LSTM model...")
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=False),
        Dense(HORIZON)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    print(f"Training for {EPOCHS} epochs...")
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    
    # Save
    print("Saving model and scaler...")
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print("Training complete! New model saved.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")

if __name__ == "__main__":
    train_model()
