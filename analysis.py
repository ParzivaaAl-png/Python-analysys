import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DATA_PATH = Path("web_traffic.csv")

def load_clean():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.drop_duplicates()
    return df

def daily_series(df):
    return df.groupby("date")["views"].sum().reset_index()

def forecast(series, days=14):
    s = series.set_index("date").asfreq("D")["views"].fillna(method="ffill")
    model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=7)
    fit = model.fit()
    return fit.forecast(days)

if __name__ == "__main__":
    df = load_clean()
    ds = daily_series(df)
    print("Forecast:\n", forecast(ds).head())
