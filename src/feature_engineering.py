import pandas as pd

def prepare_features(df):
    df = df[["match_id", "radiant_name", "dire_name", "radiant_win"]]
    df = df.dropna()
    df["target"] = df["radiant_win"].astype(int)
    X = pd.get_dummies(df[["radiant_name", "dire_name"]])
    y = df["target"]
    return X, y
