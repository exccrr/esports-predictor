from flask import Flask, render_template, request
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import predict_winner, get_confidence_level


app = Flask(__name__, template_folder="templates")

import pandas as pd
matches = pd.read_csv("data/matches.csv")
teams = sorted(set(matches["radiant_name"].dropna().unique()) | set(matches["dire_name"].dropna().unique()))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        radiant = request.form.get("radiant")
        dire = request.form.get("dire")
        model = request.form.get("model")

        if radiant and dire and radiant != dire:
            rad_pct, dire_pct = predict_winner(radiant, dire, model_name=model)
            confidence = get_confidence_level(radiant, dire)
            prediction = {
                "radiant": radiant,
                "dire": dire,
                "radiant_pct": f"{rad_pct:.2f}",
                "dire_pct": f"{dire_pct:.2f}",
                "confidence": confidence
            }

    return render_template("index.html", teams=teams, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
