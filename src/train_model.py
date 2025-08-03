import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from feature_engineering import prepare_features

df = pd.read_csv("data/matches.csv")
X, y = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))
joblib.dump(model, "models/model.pkl")
print("Model trained and saved")
