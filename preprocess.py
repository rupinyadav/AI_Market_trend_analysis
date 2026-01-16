import pandas as pd
import numpy as np
import os

def load_data(filepath="data/RELIANCE.csv"):
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    print("Data loaded. Shape:", df.shape)
    return df

def clean_data(df):
    df = df.ffill().bfill()
    print("Missing values cleaned.")
    return df

def add_features(df):
    # Moving Averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # Daily Returns
    df["Daily_Return"] = df["Close"].pct_change()

    # Volatility (20-day STD)
    df["Volatility"] = df["Daily_Return"].rolling(20).std()

    # Momentum
    df["Momentum"] = df["Close"] - df["Close"].rolling(10).mean()

    # Trend labeling
    df["Trend"] = np.where(df["MA20"] > df["MA50"], "Uptrend", "Downtrend")
    df["Day"] = df.index.day
    df["Month"] = df.index.month
    df["Year"] = df.index.year
    df["DayOfWeek"] = df.index.dayofweek
    df["IsMonthStart"] = df.index.is_month_start.astype(int)
    df["IsMonthEnd"] = df.index.is_month_end.astype(int)
    print("Feature engineering completed.including time features")
    return df

def save_cleaned_data(df, filepath="data/RELIANCE_cleaned.csv"):
    df.to_csv(filepath)
    print(f"Cleaned data saved to {filepath}")

if __name__ == "__main__":
    print("STEP 2: Preprocessing Started...\n")
    df = load_data()
    df = clean_data(df)
    df = add_features(df)
    save_cleaned_data(df)
    print("\nSTEP 2 Completed Successfully ✔")

    
