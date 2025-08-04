import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from feature_engineering import prepare_features

df = pd.read_csv("data/matches.csv")
X, y = prepare_features(df)

pd.concat([X, y.rename("radiant_win")], axis=1).to_csv("data/matches_encoded.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "xgb": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    ),
    "catboost": CatBoostClassifier(verbose=0),
    "logistic": LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name.upper()} Accuracy: {acc:.3f}")
    joblib.dump(model, f"models/{name}_model.pkl")

plt.figure(figsize=(10, 6))
plot_importance(models["xgb"], max_num_features=10)
plt.tight_layout()
plt.savefig("models/feature_importance.png")
print("Feature importance plot saved to models/feature_importance.png")
