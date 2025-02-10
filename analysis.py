import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score

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

def select_best_features(X, y):
    """Uses Recursive Feature Elimination (RFE) to find the most important features."""
    model = Ridge(alpha=1.0)
    selector = RFE(model, n_features_to_select=5)  # ✅ Keep the 5 most important features
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    print(f"📊 Best features selected: {selected_features.tolist()}")
    return selected_features

def train_predictive_model(primary_csv, related_csv, model_type="linear_regression"):
    """Trains a predictive model using optimized Ridge Regression and Random Forest."""
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

    # ✅ Ensure `date` column is formatted correctly
    if 'date' in merged_df.columns:
        merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce', utc=True)
        merged_df.set_index('date', inplace=True)  # Set as index

    X = merged_df[features]
    y = merged_df[target]

    # ✅ Ensure `y` is a 1D Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]  

    print(f"📊 Initial Features: {features}")

    # ✅ Feature Selection Using Recursive Feature Elimination (RFE)
    best_features = select_best_features(X, y)
    X = X[best_features]  # ✅ Keep only best features

    # ✅ Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"🏋️ Training model: {model_type}...")

    # ✅ Model Selection with Polynomial Features & Ridge Regression Optimization
    model = None
    if model_type == "linear_regression":
        poly = PolynomialFeatures(degree=2, include_bias=False)  # ✅ Add polynomial features
        X_poly = poly.fit_transform(X_scaled)

        # ✅ Hyperparameter tuning for Ridge Regression
        param_grid = {'alpha': [0.1, 1, 10, 100]}  # ✅ Testing different regularization strengths
        grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
        grid_search.fit(X_poly, y)
        best_alpha = grid_search.best_params_['alpha']
        print(f"✅ Best Ridge alpha: {best_alpha}")
        model = Ridge(alpha=best_alpha)  # ✅ Use the best alpha
    
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=2, random_state=42)
        X_poly = X_scaled  # ✅ No need for polynomial features in Random Forest

    # ✅ Train Model
    try:
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
    except Exception as e:
        print(f"❌ Model Training Failed: {e}")
        return None, None, f"❌ Model Training Failed: {e}"

    # ✅ Compute Model Performance Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    print(f"✅ Model trained successfully. 🔹 MSE: {mse:.4f} 🔹 RMSE: {rmse:.4f} 🔹 R² Score: {r2:.4f}")

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

    return model_filename, script_filename, f"✅ Model trained successfully. 🔹 MSE: {mse:.4f} 🔹 RMSE: {rmse:.4f} 🔹 R² Score: {r2:.4f}"
