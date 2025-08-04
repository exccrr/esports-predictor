import os
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
MATCHES_PATH = BASE_DIR / "data" / "matches.csv"

model = joblib.load(MODEL_PATH)
matches = pd.read_csv(MATCHES_PATH)
matches = matches.dropna(subset=["radiant_name", "dire_name"])

winrate, form_wr, h2h = {}, {}, {}

for team in pd.concat([matches["radiant_name"], matches["dire_name"]]).dropna().unique():
    recent = matches[(matches["radiant_name"] == team) | (matches["dire_name"] == team)].head(30)
    wins = ((recent["radiant_name"] == team) & (recent["radiant_win"] == True)).sum()
    winrate[team] = (wins + 5) / (len(recent) + 10)

    form_matches = recent.head(5)
    form_wins = ((form_matches["radiant_name"] == team) & (form_matches["radiant_win"] == True)).sum()
    form_wr[team] = (form_wins + 2) / (len(form_matches) + 4)

for _, row in matches.iterrows():
    pair = tuple(sorted([row["radiant_name"], row["dire_name"]]))
    if pair not in h2h:
        h2h[pair] = {"games": 0, "wins": 0}
    h2h[pair]["games"] += 1
    if row["radiant_win"]:
        h2h[pair]["wins"] += 1

def get_h2h(r, d):
    pair = tuple(sorted([r, d]))
    if pair in h2h:
        g, w = h2h[pair]["games"], h2h[pair]["wins"]
        return (w + 2) / (g + 4)
    return 0.5

def get_confidence_level(team1, team2):
    c1 = len(matches[(matches["radiant_name"] == team1) | (matches["dire_name"] == team1)])
    c2 = len(matches[(matches["radiant_name"] == team2) | (matches["dire_name"] == team2)])
    if min(c1, c2) >= 20:
        return "HIGH"
    elif min(c1, c2) >= 10:
        return "MEDIUM"
    else:
        return "LOW"

def adjust_prediction(pred, confidence):
    if confidence == "LOW":
        return 0.5 + (pred - 0.5) * 0.5
    elif confidence == "MEDIUM":
        return 0.5 + (pred - 0.5) * 0.8
    return pred

def predict_winner(radiant_team, dire_team, model_name="xgb"):
    radiant_wr = winrate.get(radiant_team, 0.5)
    dire_wr = winrate.get(dire_team, 0.5)
    radiant_form = form_wr.get(radiant_team, 0.5)
    dire_form = form_wr.get(dire_team, 0.5)
    h2h_wr = get_h2h(radiant_team, dire_team)

    if all(abs(x - 0.5) < 1e-6 for x in [radiant_wr, dire_wr, radiant_form, dire_form, h2h_wr]):
        print("\n No historical data for these teams. Returning neutral 50/50 prediction.\n")
        return 0.5, 0.5

    dummy = pd.get_dummies(matches[["radiant_name", "dire_name"]].dropna())
    X = pd.DataFrame(columns=dummy.columns)
    X.loc[0] = 0
    if f"radiant_name_{radiant_team}" in X.columns:
        X.loc[0, f"radiant_name_{radiant_team}"] = 1
    if f"dire_name_{dire_team}" in X.columns:
        X.loc[0, f"dire_name_{dire_team}"] = 1

    X["radiant_winrate"] = radiant_wr
    X["dire_winrate"] = dire_wr
    X["radiant_form"] = radiant_form
    X["dire_form"] = dire_form
    X["h2h_wr"] = h2h_wr

    prob = model.predict_proba(X)[0]
    radiant_pct = float(prob[1])
    dire_pct = float(prob[0])
    return radiant_pct, dire_pct

def predict_all_models(radiant_team, dire_team):
    models = {
        "xgb": joblib.load("models/xgb_model.pkl"),
        "catboost": joblib.load("models/catboost_model.pkl"),
        "logistic": joblib.load("models/logistic_model.pkl")
    }

    radiant_wr = winrate.get(radiant_team, 0.5)
    dire_wr = winrate.get(dire_team, 0.5)
    radiant_form = form_wr.get(radiant_team, 0.5)
    dire_form = form_wr.get(dire_team, 0.5)
    h2h_wr = get_h2h(radiant_team, dire_team)

    if all(abs(x - 0.5) < 1e-6 for x in [radiant_wr, dire_wr, radiant_form, dire_form, h2h_wr]):
        return {name: (50.0, 50.0) for name in models}

    all_teams = matches[["radiant_name","dire_name"]].dropna()
    dummy = pd.get_dummies(all_teams)
    X = pd.DataFrame(columns=dummy.columns)
    X.loc[0] = 0

    if f"radiant_name_{radiant_team}" in X.columns:
        X.loc[0,f"radiant_name_{radiant_team}"] = 1
    if f"dire_name_{dire_team}" in X.columns:
        X.loc[0,f"dire_name_{dire_team}"] = 1

    X["radiant_winrate"] = radiant_wr
    X["dire_winrate"] = dire_wr
    X["radiant_form"] = radiant_form
    X["dire_form"] = dire_form
    X["h2h_wr"] = h2h_wr

    results = {}
    for name, model in models.items():
        prob = model.predict_proba(X)[0]
        total = prob[0] + prob[1]
        radiant_pct = prob[1] / total 
        dire_pct = prob[0] / total 
        results[name] = (radiant_pct, dire_pct)

    return results

if __name__ == "__main__":
    radiant = "Team Spirit"
    dire = "PSG.LGD"
    r_pct, d_pct = predict_winner(radiant, dire)
    confidence = get_confidence_level(radiant, dire)
    r_pct = adjust_prediction(r_pct, confidence)
    d_pct = 1.0 - r_pct

    print(f"\n  Prediction for {radiant} vs {dire}:")
    print(f"   {radiant} win chance: {r_pct * 100:.2f}%")
    print(f"   {dire} win chance: {d_pct * 100:.2f}%")
    print(f"Confidence level: {confidence}")
