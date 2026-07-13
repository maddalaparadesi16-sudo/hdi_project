"""
Human Development Index (HDI) Predictor - Model Building Pipeline
This script runs the complete model training, preprocessing, evaluation, and serialization steps.
It outputs detailed status messages for each phase of the machine learning lifecycle.
"""

print("=================================================================")
print("     HUMAN DEVELOPMENT INDEX (HDI) PREDICTOR MODEL BUILDING")
print("=================================================================\n")

# Step 1: Importing Required Libraries
print("Step 1: Importing Required Libraries...")
try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import pickle
    print("SUCCESS: All required machine learning libraries imported.\n")
except ImportError as e:
    print(f"ERROR: Missing libraries: {e}\n")
    exit(1)

# Step 2: Reading the Dataset
print("Step 2: Reading the Dataset...")
import os
dataset_path = "data/hdi_data.csv"
if not os.path.exists(dataset_path):
    print("ERROR: Dataset file not found. Running dataset generation first...")
    from train import generate_hdi_dataset
    df = generate_hdi_dataset()
else:
    df = pd.read_csv(dataset_path)
    print(f"SUCCESS: Loaded dataset from '{dataset_path}' successfully.\n")

# Step 3: Understanding the Dataset
print("Step 3: Understanding the Dataset...")
print(f"  - Number of records (countries): {df.shape[0]}")
print(f"  - Number of features: {df.shape[1]}")
print("\nFirst 5 records of the dataset:")
print(df.head())
print("\nDataset Summary Statistics:")
print(df.describe())
print()

# Step 4: Data Preprocessing (Checking & Handling Null Values)
print("Step 4: Data Preprocessing...")
print("  - Checking for null/missing values:")
null_summary = df.isnull().sum()
for col, count in null_summary.items():
    print(f"    * {col}: {count} missing values")

# Handling missing values (using mean method if any exist)
has_null = df.isnull().any().any()
if has_null:
    print("  - Handling missing values via mean imputation...")
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"    * Imputed {col} with mean value: {mean_val:.2f}")
else:
    print("  - No missing values detected. Dataset is clean.")
print()

# Step 5: Selecting Dependent and Independent Variables
print("Step 5: Selecting Dependent and Independent Variables...")
# Independent features (X)
X = df[["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI per Capita"]]
# Dependent target (y)
y = df["HDI Score"]
print("  - Independent features (X):", list(X.columns))
print("  - Dependent target variable (y):", y.name)
print()

# Step 6: Splitting Dataset into Train and Test Sets
print("Step 6: Splitting Dataset into Train and Test Sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  - Training Set Size: {X_train.shape[0]} samples")
print(f"  - Testing Set Size:  {X_test.shape[0]} samples")
print()

# Step 7: Fitting the Linear Regression Model
print("Step 7: Fitting the Linear Regression Model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("  - Model training complete.")
print(f"  - Model Intercept: {model.intercept_:.6f}")
print("  - Feature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"    * {feature}: {coef:.6f}")
print()

# Step 8: Predicting the Results and Accuracy Evaluation
print("Step 8: Predicting the Results & Evaluating Performance...")
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"  - R-Squared (R2) Score:  {r2:.5f} (explains {r2*100:.2f}% of target variance)")
print(f"  - Mean Squared Error:    {mse:.6f}")
print(f"  - Mean Absolute Error:   {mae:.6f}")
print()

# Step 9: Saving the Model (Pickle Serialization)
print("Step 9: Saving the Model...")
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_file = os.path.join(model_dir, "hdi_model.pkl")

model_data = {
    "model": model,
    "features": list(X.columns),
    "metrics": {
        "r2": r2,
        "mse": mse,
        "mae": mae
    }
}

with open(model_file, "wb") as f:
    pickle.dump(model_data, f)
print(f"SUCCESS: Trained model serialized and saved to '{model_file}'.")
print("\n=================================================================")
print("                  MODEL BUILDING SUCCESSFUL")
print("=================================================================")
