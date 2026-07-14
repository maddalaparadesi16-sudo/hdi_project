import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

print("=================================================================")
print("     HUMAN DEVELOPMENT INDEX (HDI) PREDICTOR MODEL BUILDING")
print("=================================================================\n")

dataset_path = "data/hdi_data.csv"
if not os.path.exists(dataset_path):
    from train import generate_hdi_dataset
    df = generate_hdi_dataset()
else:
    df = pd.read_csv(dataset_path)

if df.isnull().any().any():
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

X = df[["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI per Capita"]]
y = df["HDI Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"  - R-Squared (R2) Score:  {r2:.5f}")

os.makedirs("models", exist_ok=True)
model_file = os.path.join("models", "hdi_model.pkl")

model_data = {
    "model": model,
    "features": list(X.columns),
    "metrics": { "r2": r2, "mse": mse, "mae": mae }
}

with open(model_file, "wb") as f:
    pickle.dump(model_data, f)
print(f"SUCCESS: Trained model serialized and saved to '{model_file}'.")