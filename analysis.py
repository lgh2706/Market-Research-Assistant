import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Disable GPU (Not needed anymore since we removed Neural Network)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")
if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

def train_predictive_model(primary_csv, related_csv, model_type="linear_regression"):
    try:
        print("📥 Loading datasets...")
        primary_df = pd.read_csv(primary_csv)
        related_df = pd.read_csv(related_csv)
    except Exception as e:
        print(f"❌ Error reading CSV files: {e}")
        return None, None, f"Error reading CSV files: {e}"

    if primary_df.shape[1] < 2 or related_df.shape[1] < 2:
        print("❌ One of the datasets does not contain enough columns for training.")
        return None, None, "One of the datasets does not contain enough columns for training."

    print("🔄 Merging datasets...")
    merged_df = pd.merge(primary_df, related_df, on='date', how='outer').fillna(0)

    if merged_df.empty:
        print("❌ Merged dataset is empty.")
        return None, None, "Merged dataset is empty."

    print("🚀 Preparing training data...")
    target = primary_df.columns[1]  # First feature column as target variable (y)
    features = [col for col in related_df.columns if col != "date"]  # Exclude "date" column from predictors

    if len(features) == 0:
        print("❌ Not enough predictor variables.")
        return None, None, "Not enough predictor variables."

    X = merged_df[features]
    y = merged_df[target]

    # ✅ Remove columns with all NaN values
    X = X.dropna(axis=1, how='all')

    # ✅ Feature Selection: Drop weak features (< 0.2 correlation)
    correlation_matrix = merged_df.corr(numeric_only=True)
    strong_features = correlation_matrix[target].abs().sort_values(ascending=False)
    strong_features = strong_features[strong_features > 0.2].index.tolist()

    if target in strong_features:
        strong_features.remove(target)  # Remove target from predictor list

    if len(strong_features) == 0:
        print("⚠️ No strongly correlated features found! Using all features.")
        strong_features = features  # Use all features if no strong ones exist

    X = merged_df[strong_features]
    print(f"📊 Selected features: {strong_features}")

    # ✅ Standardize features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split (Now 75%-25% instead of 80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    print(f"🏋️ Training model: {model_type}...")

    # ✅ Model Selection with Optimized Parameters
    if model_type == "linear_regression":
        model = LinearRegression()
    
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=2, random_state=42)
    
    else:
        print("❌ Invalid model type selected.")
        return None, None, "Invalid model type selected."

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Model trained successfully. MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

    model_filename = os.path.join(GENERATED_DIR, "predictive_model.pkl")
    joblib.dump(model, model_filename)
    print(f"💾 Model saved to: {model_filename}")

    return model_filename, None, f"Model trained successfully. MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}"
