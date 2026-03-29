import yfinance as yf
import pandas as pd

def load_data():
    stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    data = {}

    for stock in stocks:
        df = yf.download(stock, start="2018-01-01", end="2024-01-01")
        data[stock] = df["Close"]

    df_all = pd.concat(data.values(), axis=1)
    df_all.columns = stocks
    df_all.dropna(inplace=True)

    df_all.to_csv("data/raw_data.csv")
    return df_all