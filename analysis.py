import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Disable GPU (Not needed since we removed Neural Network)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")
if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

def add_time_lags(df, target_col, lags=7):
    """Creates lagged features for time-series forecasting."""
    for lag in range(1, lags + 1):
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    df.dropna(inplace=True)  # Remove rows with NaN values
    return df

def train_predictive_model(primary_csv, related_csv, model_type="linear_regression"):
    """Trains a predictive model using supervised learning with time-lagged features and cross-validation."""
    try:
        print("📥 Loading datasets...")
        primary_df = pd.read_csv(primary_csv)
        related_df = pd.read_csv(related_csv)
    except Exception as e:
        return None, None, f"❌ Error reading CSV files: {e}"

    print("🔄 Merging datasets...")
    merged_df = pd.merge(primary_df, related_df, on='date', how='outer').fillna(0)

    if merged_df.empty:
        return None, None, "❌ Merged dataset is empty."

    print("🚀 Preparing training data...")
    target = primary_df.columns[1]  # First feature column as target variable
    features = [col for col in related_df.columns if col != "date"]

    # ✅ Apply time-lagging
    merged_df = add_time_lags(merged_df, target, lags=7)

    # ✅ Ensure `date` column is formatted correctly for time-series models
    if 'date' in merged_df.columns:
        merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
        merged_df.set_index('date', inplace=True)  # Set as index

    X = merged_df[features]
    y = merged_df[target]

    # ✅ Ensure `y` is a 1D Series for ARIMA
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]  

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

    # ✅ K-Fold Cross-Validation (5-fold)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"🏋️ Training model: {model_type}...")

    # ✅ Model Selection with Optimized Parameters
    if model_type == "linear_regression":
        model = LinearRegression()
    
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=2, random_state=42)

    elif model_type == "arima":
        from statsmodels.tsa.arima.model import ARIMA

        # ✅ Ensure `y` has a proper datetime index
        if not isinstance(y.index, pd.DatetimeIndex):
            print("⚠️ Warning: ARIMA requires a datetime index. Converting...")
            y.index = pd.date_range(start="2020-01-01", periods=len(y), freq="D")  # Fake index if missing

        model = ARIMA(y, order=(5,1,0))
        model = model.fit()

    else:
        return None, None, "❌ Invalid model type selected."

    if model_type in ["linear_regression", "random_forest"]:
        scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
        print(f"📊 Cross-validation R² Scores: {scores}")
        print(f"📊 Mean R² Score: {scores.mean():.4f}")
    
    if model_type in ["linear_regression", "random_forest"]:
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
    else:  # For ARIMA
        y_pred = model.predict(start=len(y), end=len(y)+5)

    mse = mean_squared_error(y[-len(y_pred):], y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y[-len(y_pred):], y_pred)

    print(f"✅ Model trained successfully. MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

    # ✅ Save Model
    model_filename = os.path.join(GENERATED_DIR, "predictive_model.pkl")
    joblib.dump(model, model_filename)

    # ✅ Generate `run_analysis.py` script
    script_filename = os.path.join(GENERATED_DIR, "run_analysis.py")
    try:
        with open(script_filename, "w") as f:
            f.write(f"""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load trained model
model = joblib.load("predictive_model.pkl")

# Load dataset
df = pd.read_csv("{primary_csv}")
target_col = df.columns[1]

# Apply feature selection
selected_features = {strong_features}
X = df[selected_features]
y = df[target_col]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Make predictions
y_pred = model.predict(X_scaled)

# Evaluate performance
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("🔎 Model Performance:")
print(f"Mean Squared Error: {{mse:.4f}}")
print(f"R² Score: {{r2:.4f}}")
""")
        print(f"💾 Script saved to: {script_filename}")
    except Exception as e:
        print(f"❌ Error saving script: {e}")
        script_filename = None  # Prevents NoneType error

    return model_filename, script_filename, f"Model trained successfully. MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}"
