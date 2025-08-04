# Esports Win Predictor

A web application to predict the outcome of Dota 2 matches using machine learning models like XGBoost, CatBoost, and Logistic Regression.

## Features

- Predict match outcomes between any two teams.
- Supports multiple models and compares predictions.
- Displays confidence level based on historical data availability.
- Shows win probability bar chart using Matplotlib.
- Clean web interface powered by Flask.

## Models

We support the following models:
- XGBoost
- CatBoost
- Logistic Regression

## Requirements

- Python 3.9+
- Flask
- Pandas
- XGBoost
- CatBoost
- scikit-learn
- Matplotlib
- joblib

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare your dataset
Place your `matches.csv` file in the `data/` folder.

### 2. Train the models
Run the training script to train and save models:
```bash
python train_models.py
```

### 3. Run the web app
```bash
python web/app.py
```

Open your browser and go to [http://127.0.0.1:5050](http://127.0.0.1:5050)

## Example Output

- Match: `Team A` vs `Team B`
- Confidence: `HIGH`
- Win Prediction (XGBoost): 61.23% vs 38.77%
- Bar chart saved in `static/pred_chart.png`

---
