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
    error = None

    if request.method == "POST":
        radiant = request.form.get("radiant")
        dire = request.form.get("dire")
        model = request.form.get("model")

        if not radiant or not dire:
            error = "Please select both teams."
        elif radiant == dire:
            error = "Teams must be different."
        else:
            try:
                rad_pct, dire_pct = predict_winner(radiant, dire, model_name=model)
                confidence = get_confidence_level(radiant, dire)
                prediction = {
                    "radiant": radiant,
                    "dire": dire,
                    "radiant_pct": round(rad_pct * 100, 2),
                    "dire_pct": round(dire_pct * 100, 2),
                    "confidence": confidence
                }
            except Exception as e:
                error = f"Prediction error: {str(e)}"

    return render_template("index.html", teams=teams, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
