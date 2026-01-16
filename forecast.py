import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/RELIANCE_cleaned.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Use only Close price and drop NaNs
series = df["Close"].dropna() 
series = series.asfreq('B')

print("Number of data points used for training:", len(series))

# -----------------------------
# TRAIN ARIMA MODEL
# -----------------------------
# ARIMA(p, d, q)
# p = autoregressive terms
# d = differencing (to make series stationary)
# q = moving average terms
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit()

print("\nARIMA Model Summary:\n")
print(model_fit.summary())

# -----------------------------
# FORECAST NEXT 30 DAYS
# -----------------------------
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Create date index for forecast
forecast_index = pd.date_range(
    start=series.index[-1],
    periods=forecast_steps + 1,
    freq="B"  # Business days
)[1:]

forecast_series = pd.Series(forecast.values, index=forecast_index)

# -----------------------------
# PLOT RESULTS
# -----------------------------
# ZOOM: last 2 years of data
# -----------------------------
# Fill missing business days ONLY for plotting
series_plot = series.ffill()

zoom_start_date = series_plot.index[-1] - pd.DateOffset(years=2)
series_zoom = series_plot[series_plot.index >= zoom_start_date]


plt.figure(figsize=(12, 5))
plt.plot(series_zoom, label="Historical Close Price (last 2 year)")
plt.plot(forecast_series, label="Forecast (Next 30 days)", color="red")
plt.title("Reliance Stock Price Forecast using ARIMA")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig("data/arima_forecast.png")
plt.show()

print("\nForecast completed. Plot saved as data/arima_forecast.png")
