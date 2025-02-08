import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Disable GPU to prevent TensorFlow errors on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")
if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

def train_predictive_model(primary_csv, related_csv, model_type="linear_regression"):
    try:
        # Load datasets
        primary_df = pd.read_csv(primary_csv)
        related_df = pd.read_csv(related_csv)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return None, None, f"Error reading CSV files: {e}"

    if primary_df.shape[1] < 2 or related_df.shape[1] < 2:
        print("One of the datasets does not contain enough columns for training.")
        return None, None, "One of the datasets does not contain enough columns for training."

    # Merge datasets based on date
    merged_df = pd.merge(primary_df, related_df, on='date', how='outer').fillna(0)
    if merged_df.empty:
        print("Merged dataset is empty.")
        return None, None, "Merged dataset is empty."

    # Define target and predictor variables
    target = primary_df.columns[1]  # First keyword from primary industry (Y)
    features = related_df.columns[1:]  # All related industry keywords (X)

    if len(features) == 0:
        print("Not enough predictor variables.")
        return None, None, "Not enough predictor variables."

    X = merged_df[features]
    y = merged_df[target]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "neural_network":
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, verbose=0)
    else:
        print("Invalid model type selected.")
        return None, None, "Invalid model type selected."

    print("Training model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Model trained successfully. MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

    # Save trained model
    model_filename = os.path.join(GENERATED_DIR, "predictive_model.pkl")
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

    # Generate Python script
    script_filename = os.path.join(GENERATED_DIR, "run_analysis.py")
    with open(script_filename, "w") as f:
        f.write(f"""
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load trained model
model = joblib.load("predictive_model.pkl")

df = pd.read_csv("{primary_csv}")
X = df.iloc[:, 1:]
y = df.iloc[:, 0]  # First column is the target variable

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Model Performance:")
print(f"Mean Squared Error: {{mse:.4f}}")
print(f"R² Score: {{r2:.4f}}")
""")
    print(f"Script saved to {script_filename}")

    return model_filename, script_filename, f"Model trained successfully. MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}"
