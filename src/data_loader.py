import requests
import pandas as pd
import time

OPENDOTA_API = "https://api.opendota.com/api/proMatches"

def fetch_matches(pages=10, per_page=100):
    all_matches = []
    last_match_id = None

    for i in range(pages):
        url = OPENDOTA_API
        if last_match_id:
            url += f"?less_than_match_id={last_match_id}"
        print(f"Fetching page {i+1}...")
        r = requests.get(url)
        matches = r.json()
        if not matches:
            break
        all_matches.extend(matches)
        last_match_id = matches[-1]["match_id"]
        time.sleep(1) 

    df = pd.DataFrame(all_matches)
    return df

if __name__ == "__main__":
    df = fetch_matches(pages=30, per_page=100) 
    df.to_csv("data/matches.csv", index=False)
    print(f"Downloaded {len(df)} matches")
