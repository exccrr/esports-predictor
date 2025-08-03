import joblib
import pandas as pd
from src.feature_engineering import prepare_features

model = joblib.load("models/model.pkl")

def predict_winner(radiant_team, dire_team):
    all_teams = pd.read_csv("data/matches.csv")[["radiant_name","dire_name"]].dropna()
    dummy = pd.get_dummies(all_teams)
    X = pd.DataFrame(columns=dummy.columns)
    X.loc[0] = 0
    if f"radiant_name_{radiant_team}" in X.columns:
        X.loc[0,f"radiant_name_{radiant_team}"] = 1
    if f"dire_name_{dire_team}" in X.columns:
        X.loc[0,f"dire_name_{dire_team}"] = 1

    prob = model.predict_proba(X)[0]
    radiant_pct = float(prob[1]) * 100
    dire_pct = float(prob[0]) * 100

    return radiant_pct, dire_pct

if __name__ == "__main__":
    radiant_team = "Team Spirit"
    dire_team = "PSG.LGD"
    radiant_pct, dire_pct = predict_winner(radiant_team, dire_team)
    print(f"  Prediction for {radiant_team} vs {dire_team}:")
    print(f"   {radiant_team} win chance: {radiant_pct:.2f}%")
    print(f"   {dire_team} win chance: {dire_pct:.2f}%")
