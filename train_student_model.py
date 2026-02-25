
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH_ORIGINAL = 'data/ev_battery_data_with_km.csv'  # Ground truth for Stage 1
DATA_PATH_STUDENT = 'data/student_data.csv'            # Dataset for Stage 2
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """Load both datasets."""
    print("Loading datasets...")
    df_orig = pd.read_csv(DATA_PATH_ORIGINAL)
    df_student = pd.read_csv(DATA_PATH_STUDENT)
    
    # Rename columns for consistency if needed
    # df_orig columns: ['Battery Type', 'Total KM Traveled (km)', 'Charging Duration (min)', 'Charging Cycles', 'Efficiency (%)', 'Battery Temp (°C)']
    # df_student columns: ['battery_type', 'total_dist_km', 'charging_time_min', 'SOH_teacher']
    
    # Standardize column names for processing
    df_orig.rename(columns={
        'Battery Type': 'battery_type',
        'Total KM Traveled (km)': 'total_dist_km',
        'Charging Duration (min)': 'charging_time_min',
        'Charging Cycles': 'charging_cycles',
        'Efficiency (%)': 'efficiency',
        'Battery Temp (°C)': 'battery_temp'
    }, inplace=True)
    
    return df_orig, df_student

def train_stage_1(df_orig):
    """
    Train Stage 1 models on the ORIGINAL dataset (which has the sensors).
    Inputs: battery_type, total_dist_km, charging_time_min
    Outputs: charging_cycles, efficiency, battery_temp (Latent Features)
    """
    print("\n--- Phase 2, Step 2: Training Stage 1 (Latent Feature Estimation) ---")
    
    # Inputs for Stage 1
    X = df_orig[['battery_type', 'total_dist_km', 'charging_time_min']]
    
    # Targets for Stage 1 (Virtual Sensors)
    targets = ['charging_cycles', 'efficiency', 'battery_temp']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['total_dist_km', 'charging_time_min']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['battery_type'])
        ])
    
    stage1_models = {}
    
    for target in targets:
        y = df_orig[target]
        print(f"Training Stage 1 model for: {target}")
        
        # Using Random Forest as per plan
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        model.fit(X, y)
        stage1_models[target] = model
        
        # Simple evaluation on training set (since this is just for creating features)
        y_pred = model.predict(X)
        print(f"  {target} R2: {r2_score(y, y_pred):.4f}")
        
        # Save model
        joblib.dump(model, os.path.join(MODELS_DIR, f'stage1_{target}.pkl'))
        
    return stage1_models

def train_stage_2(df_student, stage1_models):
    """
    Train Stage 2 model on the STUDENT dataset.
    1. Use Stage 1 models to predict latent features for summary dataset.
    2. Train final SOH Estimator.
    """
    print("\n--- Phase 2, Step 2: Training Stage 2 (Final SOH Estimation) ---")
    
    # 1. Generate Latent Features
    print("Generating latent features for student dataset...")
    X_student_base = df_student[['battery_type', 'total_dist_km', 'charging_time_min']]
    
    # Create a copy to avoid SettingWithCopy warnings
    df_student_augmented = df_student.copy()
    
    for target, model in stage1_models.items():
        df_student_augmented[f'pred_{target}'] = model.predict(X_student_base)
    
    # 2. Train Stage 2 Model
    # Inputs: Original Inputs + Latent Features
    # Target: SOH_teacher
    
    feature_cols = ['battery_type', 'total_dist_km', 'charging_time_min', 
                    'pred_charging_cycles', 'pred_efficiency', 'pred_battery_temp']
    target_col = 'SOH_teacher'
    
    # Filter 0 values in target if necessary (assuming 0 is missing/error based on analysis)
    # Keeping them for now unless they skew results massively, but usually 0 degradation is suspicious if other cols are non-zero.
    # User constraint: "Use ONLY provided datasets".
    # Inspecting: the mean is 21. If 0 is valid "no degradation", we keep it. 
    # Let's clean extremely low values if they look like errors, but for now we follow "Do not use synthetic data" strictly.
    # However, let's remove rows where SOH_teacher is EXACTLY 0.0 if it makes sense.
    # Checking the data snippet: lines like "0,46,NCM_Type1,93.28... , 0.0" -> SOH=0. 
    # Wait, SOH usually means State of Health (100% is new). 
    # But values are 0-42. This looks like "Capacity Fade %" or "Degradation %". 
    # If it is Degradation %, then 0 is possible (brand new).
    # But checking line 2: 91km driven, SOH=1.07. Line 3: 93km, SOH=0.0. A bit inconsistent.
    # We will compute metrics on ALL data.
    
    X = df_student_augmented[feature_cols]
    y = df_student_augmented[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing
    # Note: 'battery_type' is the only categorical. 
    # Latent features are numerical.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['total_dist_km', 'charging_time_min', 'pred_charging_cycles', 'pred_efficiency', 'pred_battery_temp']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['battery_type'])
        ])
    
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = -np.inf
    best_name = ""
    
    results = []
    
    for name, regressor in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Model: {name} | R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        results.append({'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae})
        
        if r2 > best_score:
            best_score = r2
            best_model = pipeline
            best_name = name
            
    print(f"\nBest Stage 2 Model: {best_name} (R2={best_score:.4f})")
    
    # Save best model
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'stage2_soh_model.pkl'))
    
    # Save scaler/preprocessor separately if needed, but Pipeline handles it.
    
    return best_model, results, df_student_augmented

if __name__ == "__main__":
    # 1. Load Data
    df_orig, df_student = load_data()
    
    # 2. Stage 1: Latent Feature Estimation
    stage1_models = train_stage_1(df_orig)
    
    # 3. Stage 2: Final Health Estimation
    best_student_model, evaluation_results, df_augmented = train_stage_2(df_student, stage1_models)
    
    # 4. Save analysis results
    pd.DataFrame(evaluation_results).to_csv("features_evaluation.csv", index=False)
    print("\nTraining Pipeline Completed. Models saved.")
