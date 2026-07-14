import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Prevents UI crashing on servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#121214", "axes.facecolor": "#1a1a1e",
    "text.color": "#e1e1e6", "axes.labelcolor": "#a8a8b2",
    "xtick.color": "#a8a8b2", "ytick.color": "#a8a8b2",
    "grid.color": "#29292e"
})

def generate_hdi_dataset():
    print("Generating HDI dataset...")
    seed_profiles = [
        ("Switzerland", 84.0, 16.5, 13.9, 66933),
        ("Norway", 83.2, 18.2, 13.0, 64660),
        ("Iceland", 82.7, 19.2, 13.8, 55782),
        ("Hong Kong", 85.5, 17.3, 12.2, 62607),
        ("Australia", 84.5, 21.0, 12.7, 49238),
        ("United States", 77.2, 16.3, 13.7, 64765),
        ("China", 78.2, 14.2, 8.1, 17504),
        ("India", 67.2, 11.9, 6.7, 6590),
        ("Nigeria", 52.7, 9.8, 5.2, 4790),
        ("Mali", 58.9, 7.4, 2.3, 2084),
        # Abridged here for space - keep your original massive seed_profiles list exactly as it was!
    ]
    
    data = []
    for country, le, eys, mys, gni in seed_profiles:
        lei = (le - 20) / (85 - 20)
        ei = ((eys / 18) + (mys / 15)) / 2
        gni_capped = min(gni, 75000)
        ii = (np.log(gni_capped) - np.log(100)) / (np.log(75000) - np.log(100))
        hdi = round(min(max((lei * ei * ii) ** (1/3), 0.0), 1.0), 3)
        data.append([country, le, eys, mys, gni, hdi])
    
    np.random.seed(42)
    additional_countries = ["Andorra", "Bahrain", "Barbados", "Brunei", "Estonia"] # Kept short for clarity, keep yours
    
    for virtual_country in additional_countries:
        base = seed_profiles[np.random.choice(len(seed_profiles))]
        _, le, eys, mys, gni = base
        le_new = round(float(np.clip(le + np.random.normal(0, 3.0), 45.0, 85.0)), 1)
        eys_new = round(float(np.clip(eys + np.random.normal(0, 1.5), 4.0, 21.0)), 1)
        mys_new = round(float(np.clip(mys + np.random.normal(0, 1.2), 1.5, 14.5)), 1)
        gni_new = int(np.clip(gni * np.exp(np.random.normal(0, 0.25)), 500, 110000))
        
        lei = (le_new - 20) / (85 - 20)
        ei = ((eys_new / 18) + (mys_new / 15)) / 2
        gni_capped = min(gni_new, 75000)
        ii = (np.log(gni_capped) - np.log(100)) / (np.log(75000) - np.log(100))
        hdi = round(min(max((lei * ei * ii) ** (1/3), 0.0), 1.0), 3)
        data.append([virtual_country, le_new, eys_new, mys_new, gni_new, hdi])
        
    df = pd.DataFrame(data, columns=["Country", "Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI per Capita", "HDI Score"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/hdi_data.csv", index=False)
    print(f"Dataset saved with {len(df)} rows.")
    return df

def train_and_visualize():
    if not os.path.exists("data/hdi_data.csv"): df = generate_hdi_dataset()
    else: df = pd.read_csv("data/hdi_data.csv")
        
    for col in ["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI per Capita", "HDI Score"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
            
    def get_hdi_tier(score):
        if score >= 0.800: return "Very High"
        elif score >= 0.700: return "High"
        elif score >= 0.550: return "Medium"
        else: return "Low"
            
    df["HDI Tier"] = df["HDI Score"].apply(get_hdi_tier)
    os.makedirs("static/plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.stripplot(x="HDI Tier", y="HDI Score", data=df, order=["Low", "Medium", "High", "Very High"],
                  palette={"Very High": "#10b981", "High": "#06b6d4", "Medium": "#f59e0b", "Low": "#f43f5e"},
                  hue="HDI Tier", size=7, jitter=0.25, alpha=0.85)
    plt.savefig("static/plots/strip_plot.png", dpi=200, facecolor="#121214")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df["HDI Score"], kde=True, color="#06b6d4", bins=20, alpha=0.7)
    plt.savefig("static/plots/dist_plot.png", dpi=200, facecolor="#121214")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    features_corr = df[["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI per Capita", "HDI Score"]]
    sns.heatmap(features_corr.corr(), annot=True, cmap="coolwarm", fmt=".3f")
    plt.savefig("static/plots/heatmap.png", dpi=200, facecolor="#121214")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.scatterplot(ax=axes[0, 0], x="Life Expectancy", y="HDI Score", data=df)
    sns.scatterplot(ax=axes[0, 1], x="Expected Years of Schooling", y="HDI Score", data=df)
    sns.scatterplot(ax=axes[1, 0], x="Mean Years of Schooling", y="HDI Score", data=df)
    sns.scatterplot(ax=axes[1, 1], x="GNI per Capita", y="HDI Score", data=df)
    axes[1, 1].set_xscale('log')
    plt.savefig("static/plots/scatter_plots.png", dpi=200, facecolor="#121214")
    plt.close()

    X = df[["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI per Capita"]]
    y = df["HDI Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    os.makedirs("models", exist_ok=True)
    model_data = {
        "model": model, "features": list(X.columns),
        "metrics": { "r2": r2, "mse": mse, "mae": mae }
    }
    with open("models/hdi_model.pkl", "wb") as f: pickle.dump(model_data, f)
    print("\nTrained model and metadata serialized and saved to models/hdi_model.pkl")

if __name__ == "__main__":
    train_and_visualize()