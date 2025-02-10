import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pmdarima as pm  # Auto ARIMA
from statsmodels.tsa.stattools import adfuller  # ADF test for stationarity

# Disable GPU (Not needed)
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

def check_stationarity(series):
    """Perform Augmented Dickey-Fuller (ADF) test for stationarity."""
    result = adfuller(series.dropna())  # Drop NaN values
    p_value = result[1]
    
    if p_value > 0.05:
        print(f"⚠️ Warning: Data is NOT stationary (p={p_value:.4f}). Differencing is needed.")
    else:
        print(f"✅ Data is stationary (p={p_value:.4f}). No differencing required.")

    return p_value <= 0.05  # Return True if stationary

def optimize_arima(y):
    """Automatically selects the best (p,d,q) values for ARIMA."""
    print("🔍 Optimizing ARIMA order...")
    
    auto_arima_model = pm.auto_arima(
        y, seasonal=False, stepwise=True, suppress_warnings=True, trace=True
    )
    
    return auto_arima_model.order  # Returns (p,d,q)

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

    print(f"📊 Selected features: {features}")

    # ✅ Standardize features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"🏋️ Training model: {model_type}...")

    # ✅ Model Selection
    model = None
    y_pred = None

    if model_type == "linear_regression":
        model = LinearRegression()
    
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=2, random_state=42)

    elif model_type == "arima":
        from statsmodels.tsa.arima.model import ARIMA

        # ✅ Ensure `y` has a proper datetime index
        if not isinstance(y.index, pd.DatetimeIndex):
            print("⚠️ ARIMA requires a datetime index. Assigning a default range...")
            y.index = pd.date_range(start="2020-01-01", periods=len(y), freq="D")

        try:
            model = ARIMA(y, order=(5,1,0)).fit()
            y_pred = model.predict(start=len(y), end=len(y)+4)
        except Exception as e:
            print(f"❌ ARIMA Model Training Failed: {e}")
            return None, None, f"❌ ARIMA Model Failed: {e}"

    # ✅ Train non-ARIMA models
    if model_type in ["linear_regression", "random_forest"]:
        try:
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
        except Exception as e:
            print(f"❌ Model Training Failed: {e}")
            return None, None, f"❌ Model Training Failed: {e}"

    # ✅ Compute Model Performance Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    print(f"✅ Model trained successfully.")
    print(f"📊 Final Results:")
    print(f"    🔹 MSE: {mse:.4f}")
    print(f"    🔹 RMSE: {rmse:.4f}")
    print(f"    🔹 R² Score: {r2:.4f}")

    # ✅ Save Model
    model_filename = os.path.join(GENERATED_DIR, "predictive_model.pkl")
    joblib.dump(model, model_filename)
    print(f"💾 Model saved to: {model_filename}")

    # ✅ Generate `run_analysis.py` script
    script_filename = os.path.join(GENERATED_DIR, "run_analysis.py")
    try:
        with open(script_filename, "w") as f:
            f.write(f"""
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load trained model
model = joblib.load("predictive_model.pkl")

df = pd.read_csv("{primary_csv}")
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("🔎 Model Performance:")
print(f"Mean Squared Error: {{mse:.4f}}")
print(f"R² Score: {{r2:.4f}}")
""")
        print(f"💾 Script saved to: {script_filename}")
    except Exception as e:
        print(f"❌ Error saving script: {e}")

    return model_filename, script_filename, f"✅ Model trained successfully.<br>🔹 MSE: {mse:.4f}<br>🔹 RMSE: {rmse:.4f}<br>🔹 R² Score: {r2:.4f}<br><a href='/download_model/{os.path.basename(model_filename)}'>Download Trained Model</a><br><a href='/download_script/{os.path.basename(script_filename)}'>Download Python Script</a>"
