import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set aesthetic styling for Matplotlib/Seaborn
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#121214",
    "axes.facecolor": "#1a1a1e",
    "text.color": "#e1e1e6",
    "axes.labelcolor": "#a8a8b2",
    "xtick.color": "#a8a8b2",
    "ytick.color": "#a8a8b2",
    "grid.color": "#29292e"
})

def generate_hdi_dataset():
    """Generates a high-fidelity synthetic HDI dataset matching UNDP profiles for 180 countries."""
    print("Generating HDI dataset...")
    
    # Base profiles for countries across different development tiers
    # Columns: [Country, Life Expectancy, Expected Schooling, Mean Schooling, GNI per Capita]
    seed_profiles = [
        # Very High HDI (>= 0.800)
        ("Switzerland", 84.0, 16.5, 13.9, 66933),
        ("Norway", 83.2, 18.2, 13.0, 64660),
        ("Iceland", 82.7, 19.2, 13.8, 55782),
        ("Hong Kong", 85.5, 17.3, 12.2, 62607),
        ("Australia", 84.5, 21.0, 12.7, 49238),
        ("Denmark", 81.4, 18.7, 12.5, 60365),
        ("Sweden", 83.0, 19.4, 12.6, 54489),
        ("Germany", 80.8, 17.0, 14.1, 54538),
        ("Singapore", 82.8, 16.5, 11.6, 90919),
        ("Netherlands", 81.7, 18.6, 12.4, 55979),
        ("Canada", 82.7, 16.4, 13.8, 46800),
        ("United States", 77.2, 16.3, 13.7, 64765),
        ("United Kingdom", 80.7, 17.2, 13.4, 45225),
        ("Japan", 84.8, 15.2, 13.4, 42274),
        ("South Korea", 83.7, 16.5, 12.5, 44059),
        ("Israel", 82.3, 16.1, 13.3, 41524),
        ("Slovenia", 80.7, 17.7, 12.8, 39746),
        ("Spain", 83.0, 17.9, 10.3, 38354),
        ("France", 82.5, 15.8, 11.5, 45937),
        ("Italy", 82.9, 16.2, 10.7, 42840),
        ("United Arab Emirates", 78.7, 15.7, 12.7, 62574),
        ("Saudi Arabia", 76.9, 16.1, 11.3, 46112),
        ("Chile", 78.9, 16.7, 10.9, 24890),
        ("Argentina", 75.4, 17.9, 11.1, 20927),
        ("Croatia", 77.6, 15.1, 11.5, 30132),
        
        # High HDI (0.700 - 0.799)
        ("Uruguay", 76.4, 16.8, 9.0, 21979),
        ("Turkey", 76.0, 18.3, 8.6, 31024),
        ("Costa Rica", 77.0, 16.5, 8.7, 19974),
        ("Russia", 71.3, 15.8, 12.2, 27044),
        ("Romania", 74.2, 14.2, 11.1, 30026),
        ("Malaysia", 74.7, 13.0, 10.6, 25893),
        ("Georgia", 71.7, 15.6, 10.6, 14456),
        ("Panama", 76.2, 13.0, 10.5, 26905),
        ("Albania", 76.5, 14.4, 11.3, 14131),
        ("Bulgaria", 70.8, 13.9, 11.4, 23079),
        ("Mexico", 70.2, 14.9, 9.2, 17896),
        ("Brazil", 72.8, 15.6, 8.1, 14370),
        ("Colombia", 72.8, 14.4, 8.9, 14384),
        ("Peru", 72.4, 15.4, 9.7, 12246),
        ("Ukraine", 71.6, 15.1, 11.1, 13256),
        ("China", 78.2, 14.2, 8.1, 17504),
        ("Dominican Republic", 72.6, 14.2, 9.3, 17990),
        ("Azerbaijan", 69.4, 13.5, 10.5, 14257),
        ("Ecuador", 73.7, 14.9, 8.8, 10312),
        ("Algeria", 76.4, 14.6, 8.0, 10800),
        
        # Medium HDI (0.550 - 0.699)
        ("Egypt", 70.2, 13.8, 9.6, 11732),
        ("Vietnam", 73.6, 13.0, 8.4, 7867),
        ("Bolivia", 63.6, 14.9, 9.8, 8111),
        ("Kyrgyzstan", 70.0, 13.2, 11.4, 4846),
        ("Morocco", 74.0, 14.2, 5.6, 7303),
        ("India", 67.2, 11.9, 6.7, 6590),
        ("Bangladesh", 72.4, 12.4, 6.2, 5472),
        ("El Salvador", 70.8, 12.7, 7.2, 8296),
        ("Nicaragua", 73.8, 12.6, 7.1, 5625),
        ("Guatemala", 69.2, 10.6, 6.7, 8432),
        ("Tajikistan", 71.6, 11.7, 11.3, 4548),
        ("Honduras", 70.1, 10.1, 6.6, 5298),
        ("Cape Verde", 74.1, 12.6, 6.3, 6178),
        ("Bhutan", 71.8, 13.2, 5.2, 9438),
        ("Nepal", 69.0, 12.9, 5.1, 3877),
        ("Kenya", 61.4, 10.7, 6.7, 4244),
        ("Cambodia", 69.6, 11.5, 5.1, 4079),
        ("Angola", 61.6, 12.1, 5.4, 5466),
        ("Myanmar", 65.7, 10.9, 6.4, 3851),
        ("Pakistan", 66.1, 8.7, 4.5, 4624),
        ("Zimbabwe", 59.3, 12.1, 8.7, 3024),
        ("Ghana", 63.8, 12.0, 5.7, 5172),
        
        # Low HDI (< 0.550)
        ("Syria", 72.1, 9.2, 5.1, 4172),
        ("Cameroon", 60.3, 11.9, 6.3, 3621),
        ("Nepal", 68.4, 12.2, 5.0, 3218),
        ("Uganda", 62.7, 10.1, 5.7, 2181),
        ("Rwanda", 66.1, 11.2, 4.4, 2210),
        ("Nigeria", 52.7, 9.8, 5.2, 4790),
        ("Togo", 61.6, 13.0, 5.0, 2167),
        ("Mauritania", 64.4, 9.4, 4.9, 5075),
        ("Madagascar", 64.5, 10.1, 5.1, 1484),
        ("Benin", 59.8, 12.6, 4.3, 3409),
        ("Senegal", 67.1, 9.0, 3.2, 3503),
        ("Sudan", 65.3, 8.0, 3.8, 3575),
        ("Afghanistan", 62.0, 10.3, 3.0, 1824),
        ("Ethiopia", 65.0, 9.7, 3.2, 2361),
        ("Guinea", 58.9, 9.4, 2.2, 2481),
        ("Yemen", 63.8, 9.1, 3.2, 1314),
        ("Mozambique", 59.3, 10.2, 3.2, 1198),
        ("Burundi", 61.7, 10.7, 3.1, 732),
        ("Central African Republic", 53.9, 8.0, 4.3, 966),
        ("Niger", 61.6, 7.0, 2.1, 1240),
        ("Chad", 52.5, 7.4, 2.6, 1364),
        ("Mali", 58.9, 7.4, 2.3, 2084),
        ("South Sudan", 55.0, 5.7, 5.7, 768),
        ("Somalia", 55.3, 7.2, 2.5, 1100),
        ("Sierra Leone", 60.1, 9.0, 3.5, 1622)
    ]
    
    # Build full country dataset using standard noise generation
    data = []
    
    # Add seed countries directly
    for country, le, eys, mys, gni in seed_profiles:
        # Calculate standard HDI index representation
        lei = (le - 20) / (85 - 20)
        eysi = eys / 18
        mysi = mys / 15
        ei = (eysi + mysi) / 2
        # UNDP cap GNI at 75,000 for calculation purposes
        gni_capped = min(gni, 75000)
        ii = (np.log(gni_capped) - np.log(100)) / (np.log(75000) - np.log(100))
        hdi = (lei * ei * ii) ** (1/3)
        hdi = round(min(max(hdi, 0.0), 1.0), 3)
        data.append([country, le, eys, mys, gni, hdi])
    
    # Generate variations around the seeds to inflate the dataset to 180 countries
    np.random.seed(42)
    existing_countries = {p[0] for p in seed_profiles}
    
    # Virtual names of other countries
    additional_countries = [
        "Andorra", "Bahrain", "Barbados", "Brunei", "Estonia", "Greece", "Kuwait", "Latvia", 
        "Lithuania", "Malta", "Oman", "Poland", "Portugal", "Qatar", "Slovakia", "Bahamas", 
        "Bulgaria", "Mauritius", "Montenegro", "Palau", "Seychelles", "Trinidad and Tobago", 
        "Antigua and Barbuda", "Armenia", "Bosnia and Herzegovina", "Fiji", "Grenada", 
        "Lebanon", "North Macedonia", "Saint Lucia", "Saint Vincent", "Suriname", "Tonga", 
        "Belize", "Egypt", "Indonesia", "Iraq", "Jamaica", "Jordan", "Moldova", "Mongolia", 
        "Paraguay", "Philippines", "Samoa", "South Africa", "Sri Lanka", "Uzbekistan", 
        "Bolivia", "Gabon", "Guyana", "Morocco", "Namibia", "Congo", "Equatorial Guinea", 
        "Honduras", "India", "Laos", "Myanmar", "Nicaragua", "São Tomé", "Vanuatu", 
        "Eswatini", "Ghana", "Kenya", "Angola", "Cameroon", "Comoros", "Mauritania", 
        "Papua New Guinea", "Solomon Islands", "Syria", "Zimbabwe", "Benin", "Djibouti", 
        "Lesotho", "Rwanda", "Togo", "Uganda", "Zambia", "Gambia", "Malawi", "Sudan", 
        "Yemen", "Afghanistan", "Burkina Faso", "Burundi", "Central African Republic", 
        "Chad", "Congo (DRC)", "Eritrea", "Ethiopia", "Guinea", "Guinea-Bissau", "Liberia", 
        "Madagascar", "Mali", "Mozambique", "Niger", "Senegal", "Sierra Leone", "South Sudan"
    ]
    
    for virtual_country in additional_countries:
        if virtual_country in existing_countries:
            continue
            
        # Pick a random profile to perturb
        base = seed_profiles[np.random.choice(len(seed_profiles))]
        _, le, eys, mys, gni = base
        
        # Add normal perturbation
        le_new = round(float(np.clip(le + np.random.normal(0, 3.0), 45.0, 85.0)), 1)
        eys_new = round(float(np.clip(eys + np.random.normal(0, 1.5), 4.0, 21.0)), 1)
        mys_new = round(float(np.clip(mys + np.random.normal(0, 1.2), 1.5, 14.5)), 1)
        gni_new = int(np.clip(gni * np.exp(np.random.normal(0, 0.25)), 500, 110000))
        
        # Calculate HDI Score
        lei = (le_new - 20) / (85 - 20)
        eysi = eys_new / 18
        mysi = mys_new / 15
        ei = (eysi + mysi) / 2
        gni_capped = min(gni_new, 75000)
        ii = (np.log(gni_capped) - np.log(100)) / (np.log(75000) - np.log(100))
        hdi = (lei * ei * ii) ** (1/3)
        hdi = round(min(max(hdi, 0.0), 1.0), 3)
        
        data.append([virtual_country, le_new, eys_new, mys_new, gni_new, hdi])
        existing_countries.add(virtual_country)
        
    df = pd.DataFrame(data, columns=[
        "Country", "Life Expectancy", "Expected Years of Schooling", 
        "Mean Years of Schooling", "GNI per Capita", "HDI Score"
    ])
    
    # Save directory verification
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/hdi_data.csv", index=False)
    print(f"Dataset saved with {len(df)} rows.")
    return df

def train_and_visualize():
    # 1. Load or Generate Dataset
    if not os.path.exists("data/hdi_data.csv"):
        df = generate_hdi_dataset()
    else:
        df = pd.read_csv("data/hdi_data.csv")
        print(f"Loaded existing dataset with {len(df)} rows.")
        
    # 2. Check and Handle Null Values
    print("\n--- Preprocessing & Handling Missing Values ---")
    null_counts = df.isnull().sum()
    print("Null value counts before imputation:\n", null_counts)
    
    # Fill numeric null values using the mean method
    for col in ["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI per Capita", "HDI Score"]:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"Imputed null values in column '{col}' with mean: {mean_val:.2f}")
            
    # 3. Categorize into Development Tiers for Visualization
    def get_hdi_tier(score):
        if score >= 0.800:
            return "Very High"
        elif score >= 0.700:
            return "High"
        elif score >= 0.550:
            return "Medium"
        else:
            return "Low"
            
    df["HDI Tier"] = df["HDI Score"].apply(get_hdi_tier)
    
    # Create static/plots directory
    os.makedirs("static/plots", exist_ok=True)
    
    # 4. Data Visualization & Analysis
    print("\n--- Generating Visualizations ---")
    
    # A. Strip Plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(x="HDI Tier", y="HDI Score", data=df, order=["Low", "Medium", "High", "Very High"],
                  palette={"Very High": "#10b981", "High": "#06b6d4", "Medium": "#f59e0b", "Low": "#f43f5e"},
                  hue="HDI Tier", size=7, jitter=0.25, alpha=0.85)
    plt.title("Distribution of HDI Scores Across Development Tiers", fontsize=14, pad=15)
    plt.xlabel("Human Development Tier", fontsize=12)
    plt.ylabel("Human Development Index (HDI) Score", fontsize=12)
    plt.tight_layout()
    plt.savefig("static/plots/strip_plot.png", dpi=200, facecolor="#121214")
    plt.close()
    
    # B. Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df["HDI Score"], kde=True, color="#06b6d4", bins=20, alpha=0.7)
    plt.title("Global Distribution of Human Development Index Scores", fontsize=14, pad=15)
    plt.xlabel("HDI Score", fontsize=12)
    plt.ylabel("Number of Countries", fontsize=12)
    plt.tight_layout()
    plt.savefig("static/plots/dist_plot.png", dpi=200, facecolor="#121214")
    plt.close()
    
    # C. Correlation Matrix Heatmap
    plt.figure(figsize=(8, 6))
    features_corr = df[["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI per Capita", "HDI Score"]]
    corr_matrix = features_corr.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", linewidths=1.0, 
                vmin=0.5, vmax=1.0, cbar_kws={"shrink": .8},
                annot_kws={"size": 11, "weight": "bold"})
    plt.title("Correlation Matrix of Socioeconomic Indicators and HDI", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("static/plots/heatmap.png", dpi=200, facecolor="#121214")
    plt.close()
    
    # D. Joint Scatter Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Relationships Between Key Indicators and HDI Score", fontsize=16, color="#e1e1e6", y=0.98)
    
    # Scatter 1: Life Expectancy
    sns.scatterplot(ax=axes[0, 0], x="Life Expectancy", y="HDI Score", data=df, hue="HDI Tier", 
                    palette={"Very High": "#10b981", "High": "#06b6d4", "Medium": "#f59e0b", "Low": "#f43f5e"}, legend=False, alpha=0.8)
    axes[0, 0].set_title("Life Expectancy vs HDI", fontsize=12)
    axes[0, 0].set_xlabel("Life Expectancy at Birth (Years)")
    axes[0, 0].set_ylabel("HDI Score")
    
    # Scatter 2: Expected Years of Schooling
    sns.scatterplot(ax=axes[0, 1], x="Expected Years of Schooling", y="HDI Score", data=df, hue="HDI Tier", 
                    palette={"Very High": "#10b981", "High": "#06b6d4", "Medium": "#f59e0b", "Low": "#f43f5e"}, legend=False, alpha=0.8)
    axes[0, 1].set_title("Expected Years of Schooling vs HDI", fontsize=12)
    axes[0, 1].set_xlabel("Expected Years of Schooling (Years)")
    axes[0, 1].set_ylabel("HDI Score")
    
    # Scatter 3: Mean Years of Schooling
    sns.scatterplot(ax=axes[1, 0], x="Mean Years of Schooling", y="HDI Score", data=df, hue="HDI Tier", 
                    palette={"Very High": "#10b981", "High": "#06b6d4", "Medium": "#f59e0b", "Low": "#f43f5e"}, legend=False, alpha=0.8)
    axes[1, 0].set_title("Mean Years of Schooling vs HDI", fontsize=12)
    axes[1, 0].set_xlabel("Mean Years of Schooling (Years)")
    axes[1, 0].set_ylabel("HDI Score")
    
    # Scatter 4: GNI per Capita (Log scale representation is better but let's plot on normal/log x)
    sns.scatterplot(ax=axes[1, 1], x="GNI per Capita", y="HDI Score", data=df, hue="HDI Tier", 
                    palette={"Very High": "#10b981", "High": "#06b6d4", "Medium": "#f59e0b", "Low": "#f43f5e"}, alpha=0.8)
    axes[1, 1].set_title("GNI per Capita vs HDI", fontsize=12)
    axes[1, 1].set_xlabel("Gross National Income (GNI) per Capita (PPP $)")
    axes[1, 1].set_ylabel("HDI Score")
    axes[1, 1].set_xscale('log') # Log scale GNI makes the relationship linear!
    
    # Put legend outside the plot
    axes[1, 1].legend(title="HDI Tier", loc="lower right", facecolor="#1a1a1e", edgecolor="#29292e", labelcolor="#e1e1e6")
    
    plt.tight_layout()
    plt.savefig("static/plots/scatter_plots.png", dpi=200, facecolor="#121214")
    plt.close()
    
    print("All plots generated and saved in static/plots/")
    
    # 5. Machine Learning Model Building
    print("\n--- Building Machine Learning Model ---")
    
    # Features (X) and Target (y)
    # Using raw indicators as requested
    X = df[["Life Expectancy", "Expected Years of Schooling", "Mean Years of Schooling", "GNI per Capita"]]
    y = df["HDI Score"]
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and fit Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicting results
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n--- Model Performance Evaluation ---")
    print(f"R-Squared (R2) Score:  {r2:.5f}")
    print(f"Mean Squared Error:    {mse:.5f}")
    print(f"Mean Absolute Error:   {mae:.5f}")
    print("\nModel Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")
    
    # 6. Model Saving & Serialization
    os.makedirs("models", exist_ok=True)
    model_data = {
        "model": model,
        "features": list(X.columns),
        "metrics": {
            "r2": r2,
            "mse": mse,
            "mae": mae
        }
    }
    
    with open("models/hdi_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
        
    print("\nTrained model and metadata serialized and saved to models/hdi_model.pkl")

if __name__ == "__main__":
    train_and_visualize()
