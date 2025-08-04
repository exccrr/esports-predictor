import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
matches = pd.read_csv("data/matches.csv")
matches = matches.dropna(subset=["radiant_name", "dire_name"])

winrate = {}
for team in pd.concat([matches["radiant_name"], matches["dire_name"]]).dropna().unique():
    team_matches = matches[(matches["radiant_name"] == team) | (matches["dire_name"] == team)].head(30)
    wins = ((team_matches["radiant_name"] == team) & (team_matches["radiant_win"] == True)).sum()
    winrate[team] = (wins + 5) / (len(team_matches) + 10)

form_wr = {}
for team in winrate.keys():
    team_matches = matches[(matches["radiant_name"] == team) | (matches["dire_name"] == team)].head(5)
    wins = ((team_matches["radiant_name"] == team) & (team_matches["radiant_win"] == True)).sum()
    form_wr[team] = (wins + 2) / (len(team_matches) + 4)

h2h = {}
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
        games = h2h[pair]["games"]
        wins = h2h[pair]["wins"]
        return (wins + 2) / (games + 4)
    return 0.5

def predict_winner(radiant_team, dire_team, model_name="xgb"):
    radiant_wr = winrate.get(radiant_team, 0.5)
    dire_wr = winrate.get(dire_team, 0.5)
    radiant_form = form_wr.get(radiant_team, 0.5)
    dire_form = form_wr.get(dire_team, 0.5)
    h2h_wr = get_h2h(radiant_team, dire_team)

    if all(abs(x - 0.5) < 1e-6 for x in [radiant_wr, dire_wr, radiant_form, dire_form, h2h_wr]):
        print("\n No historical data for these teams. Returning neutral 50/50 prediction.\n")
        return 50.0, 50.0

    all_teams = matches[["radiant_name", "dire_name"]].dropna()
    dummy = pd.get_dummies(all_teams)
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

    print("\n Features used for prediction:")
    print(f"  {radiant_team} winrate (last 30): {radiant_wr:.2f}")
    print(f"  {dire_team} winrate (last 30): {dire_wr:.2f}")
    print(f"  {radiant_team} form (last 5): {radiant_form:.2f}")
    print(f"  {dire_team} form (last 5): {dire_form:.2f}")
    print(f"  Head-to-head WR: {h2h_wr:.2f}\n")

    model_path = f"models/{model_name}_model.pkl"
    model = joblib.load(model_path)
    prob = model.predict_proba(X)[0]

    radiant_pct = float(prob[1]) * 100
    dire_pct = float(prob[0]) * 100

    return radiant_pct, dire_pct


def get_confidence_level(team1, team2):
    count1 = len(matches[(matches["radiant_name"] == team1) | (matches["dire_name"] == team1)])
    count2 = len(matches[(matches["radiant_name"] == team2) | (matches["dire_name"] == team2)])

    min_matches = min(count1, count2)

    if min_matches >= 20:
        return "HIGH"
    elif min_matches >= 10:
        return "MEDIUM"
    else:
        return "LOW"

def adjust_prediction(pred, confidence):
    if confidence == "LOW":
        return 0.5 + (pred - 0.5) * 0.5
    elif confidence == "MEDIUM":
        return 0.5 + (pred - 0.5) * 0.8
    return pred

if __name__ == "__main__":
    radiant_team = "Team Spirit"
    dire_team = "PSG.LGD"

    radiant_pct, dire_pct = predict_winner(radiant_team, dire_team, model_name="xgb")

    confidence = get_confidence_level(radiant_team, dire_team)

    radiant_pct = adjust_prediction(radiant_pct / 100, confidence) * 100
    dire_pct = 100 - radiant_pct

    print(f"\n  Prediction for {radiant_team} vs {dire_team}:")
    print(f"   {radiant_team} win chance: {radiant_pct:.2f}%")
    print(f"   {dire_team} win chance: {dire_pct:.2f}%")
    print(f"Confidence level: {confidence}")
