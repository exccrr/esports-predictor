import requests
import pandas as pd
import time, os

OPENDOTA_API = "https://api.opendota.com/api/proMatches"

def fetch_matches(pages=30):
    all_matches = []
    last_match_id = None

    if not os.path.exists("data"):
        os.makedirs("data")

    for i in range(pages):
        url = OPENDOTA_API
        if last_match_id:
            url += f"?less_than_match_id={last_match_id}"
        print(f" Fetching page {i+1}...")

        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            matches = r.json()
        except Exception as e:
            print(f" Error fetching page {i+1}: {e}")
            break

        if not matches:
            print(" No more matches returned, stopping.")
            break

        all_matches.extend(matches)
        last_match_id = matches[-1]["match_id"]

        pd.DataFrame(all_matches).to_csv("data/matches_partial.csv", index=False)
        print(f" Saved {len(all_matches)} matches so far...")

        time.sleep(2)

    df = pd.DataFrame(all_matches)
    return df

if __name__ == "__main__":
    df = fetch_matches(pages=40)
    df.to_csv("data/matches.csv", index=False)
    print(f"Downloaded {len(df)} matches")
