import pandas as pd

def compute_winrate(games, wins):
    return (wins + 5) / (games + 10) 

def prepare_features(df):
    df = df[["match_id", "radiant_name", "dire_name", "radiant_win"]]
    df = df.dropna()
    df["target"] = df["radiant_win"].astype(int)

    df = df.sort_values("match_id", ascending=False)

    winrate = {}
    for team in pd.concat([df["radiant_name"], df["dire_name"]]).unique():
        team_matches = df[(df["radiant_name"] == team) | (df["dire_name"] == team)].head(30)
        wins = ((team_matches["radiant_name"] == team) & (team_matches["radiant_win"] == True)).sum()
        wr = compute_winrate(len(team_matches), wins)
        winrate[team] = wr

    df["radiant_winrate"] = df["radiant_name"].map(winrate)
    df["dire_winrate"] = df["dire_name"].map(winrate)

    form_wr = {}
    for team in winrate.keys():
        team_matches = df[(df["radiant_name"] == team) | (df["dire_name"] == team)].head(5)
        wins = ((team_matches["radiant_name"] == team) & (team_matches["radiant_win"] == True)).sum()
        form_wr[team] = compute_winrate(len(team_matches), wins)

    df["radiant_form"] = df["radiant_name"].map(form_wr)
    df["dire_form"] = df["dire_name"].map(form_wr)

    h2h = {}
    for _, row in df.iterrows():
        pair = tuple(sorted([row["radiant_name"], row["dire_name"]]))
        if pair not in h2h:
            h2h[pair] = {"games": 0, "wins": 0}
        h2h[pair]["games"] += 1
        if row["radiant_win"]:
            h2h[pair]["wins"] += 1

    def get_h2h(r, d):
        pair = tuple(sorted([r, d]))
        if pair in h2h:
            return compute_winrate(h2h[pair]["games"], h2h[pair]["wins"])
        return 0.5

    df["h2h_wr"] = df.apply(lambda x: get_h2h(x["radiant_name"], x["dire_name"]), axis=1)

    X_teams = pd.get_dummies(df[["radiant_name", "dire_name"]])
    X = pd.concat([X_teams, df[["radiant_winrate", "dire_winrate", "radiant_form", "dire_form", "h2h_wr"]]], axis=1)

    y = df["target"]
    return X, y
