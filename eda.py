#import pandas as pd

#df = pd.read_csv("data/RELIANCE_cleaned.csv", parse_dates=["Date"])
#df.set_index("Date", inplace=True)

#print("NaN values per column:")
#print(df.isna().sum())



import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/RELIANCE_cleaned.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

print("Dataset loaded:", df.shape)
print("\nSummary statistics:\n", df[["Close","Daily_Return","Volatility"]].describe())

# -----------------------------
# 1. PRICE + MOVING AVERAGES
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(df["Close"], label="Close Price")
plt.plot(df["MA20"], label="MA20")
plt.plot(df["MA50"], label="MA50")
plt.plot(df["MA200"], label="MA200")
plt.title("Reliance Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig("data/eda_price_ma.png")
plt.show()

# -----------------------------
# 2. VOLATILITY OVER TIME
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(df["Volatility"], color="orange")
plt.title("20-Day Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid()
plt.savefig("data/eda_volatility.png")
plt.show()

# -----------------------------
# 3. DAILY RETURNS
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(df["Daily_Return"], color="purple")
plt.title("Daily Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.grid()
plt.savefig("data/eda_daily_returns.png")
plt.show()

# -----------------------------
# 4. HISTOGRAM OF DAILY RETURNS
# -----------------------------
plt.figure(figsize=(8, 4))
plt.hist(df["Daily_Return"].dropna(), bins=60, color="skyblue")
plt.title("Distribution of Daily Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.grid()
plt.savefig("data/eda_returns_hist.png")
plt.show()

# -----------------------------
# 5. CORRELATION HEATMAP (OPTIONAL)
# -----------------------------
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df[["Close","MA20","MA50","MA200","Volatility","Momentum","Daily_Return"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Key Features")
plt.savefig("data/eda_corr_heatmap.png")
plt.show()

print("\nEDA Completed. All plots saved inside the data/ folder.")


# -----------------------------
# OUTLIER DETECTION USING IQR
# -----------------------------
Q1 = df["Daily_Return"].quantile(0.25)
Q3 = df["Daily_Return"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Daily_Return"] < lower_bound) |
    (df["Daily_Return"] > upper_bound)
]

print("Number of outlier days (IQR method):", len(outliers))
print(outliers[["Close", "Daily_Return"]].head())

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(df["Close"], label="Close Price")
plt.scatter(outliers.index, outliers["Close"], color="red", label="Outliers")
plt.title("Outlier Days Identified Using IQR Method")
plt.legend()
plt.grid()
plt.savefig("data/eda_outliers.png")
plt.show()
