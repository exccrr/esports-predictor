from flask import Flask, render_template, request
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import predict_all_models, get_confidence_level
import pandas as pd

app = Flask(__name__, template_folder="templates", static_folder="static")

matches = pd.read_csv("data/matches.csv")
teams = sorted(set(matches["radiant_name"].dropna().unique()) | set(matches["dire_name"].dropna().unique()))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    chart_url = None 

    if request.method == "POST":
        radiant = request.form.get("radiant")
        dire = request.form.get("dire")

        if radiant and dire and radiant != dire:
            try:
                results = predict_all_models(radiant, dire)
                confidence = get_confidence_level(radiant, dire)

                prediction = {
                    "radiant": radiant,
                    "dire": dire,
                    "confidence": confidence,
                    "models": {
                        model: {
                            "radiant_pct": f"{rad * 100:.2f}",
                            "dire_pct": f"{dire_pct * 100:.2f}"
                        }
                        for model, (rad, dire_pct) in results.items()
                    }
                }

                # Визуализация
                labels = ["XGBoost", "CatBoost", "Logistic"]
                radiant_vals = [results[m][0] for m in results]
                dire_vals = [results[m][1] for m in results]

                x = range(len(labels))
                plt.figure(figsize=(8, 4))
                plt.barh(x, radiant_vals, color="green", label=radiant)
                plt.barh(x, dire_vals, left=radiant_vals, color="red", label=dire)
                plt.yticks(x, labels)
                plt.xlabel("Win Probability (%)")
                plt.title(f"{radiant} vs {dire}")
                plt.legend()
                plt.tight_layout()

                chart_file = "pred_chart.png"
                chart_path = os.path.join(app.static_folder, chart_file)
                chart_url = f"/static/{chart_file}"

                plt.savefig(chart_path)
                plt.close()

            except Exception as e:
                error = f"Prediction failed: {e}"
        else:
            error = "Please select two different teams."

    return render_template("index.html", teams=teams, prediction=prediction, chart=chart_url, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5050)
