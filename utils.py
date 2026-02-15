import numpy as np
import pandas as pd

def forecast_next_n_days_direct(model, df, scaler, window_size=60, horizon=30):
    """
    Forecast future gold prices using the LSTM model.
    Works with both Dense output and Seq2Seq (TimeDistributed) output.
    """
    # Extract price column
    prices = df[["Price"]].values

    # Scale
    scaled = scaler.transform(prices)

    # Last window
    last_window = scaled[-window_size:]
    X = last_window.reshape(1, window_size, 1)

    # Predict (multi-output model)
    preds_scaled = model.predict(X, verbose=0)
    
    # Handle different output shapes
    # Seq2Seq: (1, horizon, 1) -> Dense: (1, horizon)
    preds_scaled = preds_scaled.reshape(-1, 1)

    # Inverse scale
    preds = scaler.inverse_transform(preds_scaled).flatten()

    # Use the actual number of predictions from the model
    actual_horizon = len(preds)
    
    # If requested horizon is less than model output, truncate predictions
    if horizon < actual_horizon:
        preds = preds[:horizon]
        actual_horizon = horizon

    # Future dates
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=actual_horizon,
        freq="D"
    )

    future_df = pd.DataFrame(
        {"Forecast_Price": preds},
        index=future_dates
    )
    future_df.index.name = "Date"

    return future_df