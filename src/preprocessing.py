from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def preprocess():
    df = pd.read_csv("data/raw_data.csv", index_col=0)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(scaled, columns=df.columns)
    df_scaled.to_csv("data/processed_data.csv")

    return df_scaled