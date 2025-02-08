import os
import pandas as pd
import joblib
from flask import request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Ensure writable directory exists for storing generated files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")
if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)


def train_predictive_model(primary_csv, related_csv):
    """Train a predictive model using Google Trends data from both industries."""
    try:
        primary_df = pd.read_csv(primary_csv)
        related_df = pd.read_csv(related_csv)
    except Exception as e:
        return None, f"Error reading CSV files: {e}"
    
    if primary_df.shape[1] < 2 or related_df.shape[1] < 2:
        return None, "One of the datasets does not contain enough columns for training."
    
    merged_df = pd.merge(primary_df, related_df, on='date', how='outer')
    target = primary_df.columns[1]  # Use the second column from primary as the target variable
    features = merged_df.columns[2:]  # Use remaining columns as features
    
    if len(features) == 0:
        return None, "Not enough predictor variables."
    
    X = merged_df[features].fillna(0)  # Fill missing values if any
    y = merged_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    if r2 < 0.5:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
    model_filename = os.path.join(GENERATED_DIR, "predictive_model.pkl")
    joblib.dump(model, model_filename)
    
    return model_filename, f"Model trained successfully. MSE: {mse:.4f}, R² Score: {r2:.4f}"
