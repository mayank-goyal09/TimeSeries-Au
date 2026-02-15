import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import timedelta

# Configuration
CSV_PATH = "Gold Price.csv"
WINDOW_SIZE = 730  # Keep exactly 730 days
TARGET_COLUMNS = ["Date", "Price", "Open", "High", "Low", "Volume", "Chg%"]

def update_database():
    print(f"Loading {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print("Error: CSV file not found.")
        return

    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    
    # Get the last date in the current database
    last_date = df.iloc[-1]['Date']
    print(f"Current Database Last Date: {last_date.date()}")
    
    # Check if we need to update
    today = pd.Timestamp.now().normalize()
    if last_date >= today:
        print("Database is already up to date.")
        return df

    # We need to fetch data from the day AFTER the last date
    start_fetch_date = last_date
    
    print(f"Fetching new data starting from {start_fetch_date.date()}...")
    
    # Fetch Gold Futures (USD) and USD/INR pair
    # We fetch a bit of overlap to calculate the conversion ratio dynamically
    gold_futures = yf.download("GC=F", start=start_fetch_date, progress=False)
    usd_inr = yf.download("INR=X", start=start_fetch_date, progress=False)

    if len(gold_futures) < 2:
        print("No new data available yet from Yahoo Finance.")
        return df

    # Prepare conversion factor based on the very last row of OUR data
    # (This ensures the new data 'glues' perfectly to the old data scale)
    # 1 Troy Oz = 31.1035 grams. We want price per 10 grams.
    # Theoretical Price = (USD_Gold / 31.1035) * 10 * USD_INR
    
    # Get vectors for calculation
    # Resample to ensure dates match (inner join on index)
    market_data = pd.DataFrame({
        'USD_Gold_Close': gold_futures['Close']['GC=F'],
        'USD_Gold_Open': gold_futures['Open']['GC=F'],
        'USD_Gold_High': gold_futures['High']['GC=F'],
        'USD_Gold_Low': gold_futures['Low']['GC=F'],
        'USD_INR': usd_inr['Close']['INR=X']
    }).dropna()

    if market_data.empty:
        print("Could not align Gold and USD/INR data.")
        return df

    # Calculate the 'Premium Factor' using the last known date in our CSV
    # matching the market data.
    
    try:
        # We need to find the latest date that exists in BOTH datasets to calculate the ratio
        common_dates = df['Date'].isin(market_data.index)
        if not common_dates.any():
             print("No date overlap found. Using last CSV price vs first Market price for ratio.")
             last_csv_price = df.iloc[-1]['Price']
             market_row = market_data.iloc[0]
             
             theoretical_inr = (market_row['USD_Gold_Close'] / 31.1035) * 10 * market_row['USD_INR']
             premium_ratio = last_csv_price / theoretical_inr
        else:
             # Use the latest common date for precision
             latest_common_date = df[df['Date'].isin(market_data.index)].iloc[-1]['Date']
             last_csv_price = df[df['Date'] == latest_common_date].iloc[0]['Price']
             market_row = market_data.loc[latest_common_date]
             
             theoretical_inr = (market_row['USD_Gold_Close'] / 31.1035) * 10 * market_row['USD_INR']
             premium_ratio = last_csv_price / theoretical_inr
        
        print(f"Calculated Locality Premium Ratio: {premium_ratio:.4f}")
        
    except Exception as e:
        print(f"Error calculating ratio: {e}")
        premium_ratio = 1.15 # Fallback

    # --- PROCESS NEW DAYS ---
    new_rows = []
    
    # Iterate through market data, skipping dates we already have
    for date, row in market_data.iterrows():
        if date <= last_date:
            continue
            
        # Calculate Scaled INR Prices using the specific row's USD values
        def convert(usd_val):
            # Formula: (USD / 31.1035) * 10 * INR * Premium
            # Ensure scalar float values
            u = float(usd_val)
            i = float(row['USD_INR'])
            base = (u / 31.1035) * 10 * i
            final = base * premium_ratio
            return final

        price = int(convert(row['USD_Gold_Close']))
        open_p = int(convert(row['USD_Gold_Open']))
        high_p = int(convert(row['USD_Gold_High']))
        low_p = int(convert(row['USD_Gold_Low']))
        
        # Previous price for Chg% calculation
        prev_price = df.iloc[-1]['Price'] if not new_rows else new_rows[-1]['Price']
        chg_pct = round(((price - prev_price) / prev_price) * 100, 2)
        
        volume = 0 # Volume data for futures is often weird, keeping 0 or mimicking avg
        
        new_row = {
            "Date": date,
            "Price": price,
            "Open": open_p,
            "High": high_p,
            "Low": low_p,
            "Volume": volume,
            "Chg%": chg_pct
        }
        new_rows.append(new_row)

    if new_rows:
        print(f"Adding {len(new_rows)} new days of data...")
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        
        # --- ROLLING WINDOW (Delete Old) ---
        if len(df) > WINDOW_SIZE:
            rows_to_remove = len(df) - WINDOW_SIZE
            print(f"Rolling Window: Removing oldest {rows_to_remove} rows to keep {WINDOW_SIZE} days.")
            df = df.iloc[rows_to_remove:].reset_index(drop=True)
        
        # Format Date back to string if needed, but CSV read/write handles ISO mostly.
        # Ensure column order
        df = df[TARGET_COLUMNS]
        
        # Save
        df.to_csv(CSV_PATH, index=False)
        print("Database updated successfully!")
    else:
        print("No new rows to add after alignment.")

    return df

if __name__ == "__main__":
    update_database()
