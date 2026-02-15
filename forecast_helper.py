"""
Forecast helper functions for Gold Price Prediction
Copy this code into a new cell in your notebook and run it!
"""
import numpy as np
import pandas as pd

def forecast_next_n_days(model, df, scaler, window_size=60, n_days=30):
    """
    Forecast next n days using recursive single-step prediction.
    Use this for single-output models (Dense(1)).
    """
    prices = df["Price"].values
    predictions = []
    
    # Get last window
    current_window = scaler.transform(prices[-window_size:].reshape(-1, 1)).flatten()
    
    for _ in range(n_days):
        # Reshape for LSTM: (1, window_size, 1)
        X = current_window.reshape(1, window_size, 1)
        
        # Predict next value (scaled)
        pred_scaled = model.predict(X, verbose=0)[0, 0]
        predictions.append(pred_scaled)
        
        # Slide window: remove first, add prediction
        current_window = np.append(current_window[1:], pred_scaled)
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    pred_prices = scaler.inverse_transform(predictions).flatten()
    
    # Create future dates
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=n_days,
        freq="D"
    )
    
    future_df = pd.DataFrame({"Forecast_Price": pred_prices}, index=future_dates)
    future_df.index.name = "Date"
    
    return future_df


def forecast_next_n_days_direct(model, df, scaler, window_size=60, horizon=30):
    """
    Forecast using multi-output model (Dense(30) or Seq2Seq).
    Returns all predictions in one shot.
    """
    prices = df[["Price"]].values
    scaled = scaler.transform(prices)
    
    last_window = scaled[-window_size:]
    X = last_window.reshape(1, window_size, 1)
    
    # Predict (handles both Dense and Seq2Seq output shapes)
    preds_scaled = model.predict(X, verbose=0)
    preds_scaled = preds_scaled.reshape(-1, 1)
    
    # Inverse scale
    preds = scaler.inverse_transform(preds_scaled).flatten()
    
    # Use actual prediction length
    actual_horizon = min(len(preds), horizon)
    preds = preds[:actual_horizon]
    
    # Future dates
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=actual_horizon,
        freq="D"
    )
    
    future_df = pd.DataFrame({"Forecast_Price": preds}, index=future_dates)
    future_df.index.name = "Date"
    
    return future_df
