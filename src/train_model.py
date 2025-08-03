import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
import joblib
from feature_engineering import prepare_features
import matplotlib.pyplot as plt

df = pd.read_csv("data/matches.csv")
X, y = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"XGBoost model trained. Accuracy: {acc:.3f}")

joblib.dump(model, "models/model.pkl")

importances = model.feature_importances_
feat_df = pd.DataFrame({"feature": X.columns, "importance": importances})
feat_df = feat_df.sort_values(by="importance", ascending=False)
print("\nTop 10 important features:")
print(feat_df.head(10))

plt.figure(figsize=(10,6))
plot_importance(model, max_num_features=10)
plt.tight_layout()
plt.savefig("models/feature_importance.png")
print("Feature importance plot saved to models/feature_importance.png")
