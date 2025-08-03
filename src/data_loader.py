import requests
import pandas as pd

OPENDOTA_API = "https://api.opendota.com/api/proMatches"

def fetch_matches(num_matches=100):
    response = requests.get(OPENDOTA_API)
    matches = response.json()
    df = pd.DataFrame(matches)
    return df.head(num_matches)

if __name__ == "__main__":
    df = fetch_matches(50)
    df.to_csv("data/matches.csv", index=False)
    print("Downloaded 50 matches")
