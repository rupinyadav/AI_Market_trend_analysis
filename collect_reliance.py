# collect_reliance.py  -- robust single-ticker download and save
import yfinance as yf
import pandas as pd
import os

def download_reliance_data(start_date="2015-01-01", end_date=None):
    ticker = "RELIANCE.NS"  # NSE Reliance

    print(f"\nDownloading {ticker} history via Ticker.history() ...")

    # Use Ticker.history which returns a clean DataFrame for a single ticker
    t = yf.Ticker(ticker)
    # history returns index = Date, columns like Open, High, Low, Close, Volume, Dividends, Stock Splits
    df = t.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)

    if df.empty:
        print("WARNING: downloaded DataFrame is empty. Check ticker and network.")
        return df

    # Reset index so Date becomes a column (easier to inspect and save)
    df = df.reset_index()

    # Normalize column names (some versions may use 'Close' vs 'Adj Close' etc.)
    # Keep the canonical order we want: Date, Open, High, Low, Close, Adj Close (if present), Volume
    cols = list(df.columns)
    # Build an output frame selecting columns if they exist:
    wanted = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    out_cols = [c for c in wanted if c in cols]
    df_out = df[out_cols].copy()

    # Ensure 'Date' column exists and is datetime
    df_out['Date'] = pd.to_datetime(df_out['Date'])

    # Create data folder if missing
    os.makedirs("data", exist_ok=True)

    filepath = "data/RELIANCE.csv"
    df_out.to_csv(filepath, index=False)
    print(f"Saved clean CSV: {filepath}")

    # Print verification
    print("\nSample rows (head):")
    print(df_out.head().to_string(index=False))
    print("\nColumns saved:", df_out.columns.tolist())
    return df_out

if __name__ == "__main__":
    download_reliance_data()

