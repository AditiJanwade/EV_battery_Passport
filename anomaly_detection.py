
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# Configuration
DATA_PATH_STUDENT = 'data/student_data.csv'
MODEL_PATH = 'models/stage2_soh_model.pkl'
STAGE1_MODELS_DIR = 'models'
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_models():
    print("Loading models...")
    stage2_model = joblib.load(MODEL_PATH)
    stage1_models = {
        'charging_cycles': joblib.load(os.path.join(STAGE1_MODELS_DIR, 'stage1_charging_cycles.pkl')),
        'efficiency': joblib.load(os.path.join(STAGE1_MODELS_DIR, 'stage1_efficiency.pkl')),
        'battery_temp': joblib.load(os.path.join(STAGE1_MODELS_DIR, 'stage1_battery_temp.pkl'))
    }
    return stage1_models, stage2_model

def evaluate_anomalies():
    print("\n--- Phase 3: Anomaly Detection ---")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH_STUDENT)
    
    # 2. Load Models
    stage1_models, stage2_model = load_models()
    
    # 3. Generate Latent Features (Stage 1 prediction)
    X_input = df[['battery_type', 'total_dist_km', 'charging_time_min']]
    
    df_aug = df.copy()
    for target, model in stage1_models.items():
        df_aug[f'pred_{target}'] = model.predict(X_input)
        
    # 4. Predict SOH (Stage 2)
    # The pipeline handles preprocessing
    X_stage2 = df_aug[['battery_type', 'total_dist_km', 'charging_time_min', 
                       'pred_charging_cycles', 'pred_efficiency', 'pred_battery_temp']]
    
    df_aug['SOH_student'] = stage2_model.predict(X_stage2)
    
    # 5. Compute Residuals
    # Residual = |Teacher - Student|
    df_aug['residual'] = np.abs(df_aug['SOH_teacher'] - df_aug['SOH_student'])
    
    residual_mean = df_aug['residual'].mean()
    residual_std = df_aug['residual'].std()
    
    print(f"Residual Mean: {residual_mean:.4f}")
    print(f"Residual Std:  {residual_std:.4f}")
    
    # 6. Thresholding Logic (Mean + k * STD)
    k = 3
    threshold = residual_mean + (k * residual_std)
    print(f"Anomaly Threshold (Mean + {k}*STD): {threshold:.4f}")
    
    df_aug['is_anomaly_detected'] = df_aug['residual'] > threshold
    num_anomalies = df_aug['is_anomaly_detected'].sum()
    print(f"Detected Anomalies: {num_anomalies} / {len(df_aug)} ({num_anomalies/len(df_aug)*100:.2f}%)")
    
    # 7. ROC Curve Generation
    # PROBLEM: We don't have "Ground Truth" anomaly labels in the dataset.
    # The user says "Use teacher SOH as reference ground truth."
    # Interpretation: We typically validate Anomaly Detection against labeled anomalies.
    # Since we lack them, for the purpose of the requirement/paper, we will define 
    # "True Anomalies" based on a statistical heuristic of the TEACHER's behavior 
    # (e.g., if Teacher output deviates significantly from the population mean, or if residuals are high).
    # OR, we treat the 'threshold' classification as the system output and compare against... what?
    # Let's assume a synthetic ground truth for demonstration: 
    # "True" anomalies are points where the Teacher value is > 95th percentile (extreme degradation).
    
    # Defining Synthetic Ground Truth for ROC evaluation (Teacher-based)
    # Let's say "Real Anomalies" are cases with Extreme Degradation (SOH_teacher > 35, approx top 5-10%)
    # This tests: "Can the Student's Residual predict Extreme Degradation cases?"
    # (High residual -> simpler model fails to capture complex extreme behavior -> anomaly)
    
    # Check distribution of SOH_teacher
    teacher_95 = df_aug['SOH_teacher'].quantile(0.95)
    df_aug['true_anomaly_label'] = (df_aug['SOH_teacher'] > teacher_95).astype(int)
    print(f"Synthetic Ground Truth (Top 5% Degradation > {teacher_95:.2f}): {df_aug['true_anomaly_label'].sum()} instances")
    
    # Compute ROC based on Residuals vs Synthetic Label
    fpr, tpr, thresholds = roc_curve(df_aug['true_anomaly_label'], df_aug['residual'])
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Save ROC Plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Anomaly Detection ROC (Residual vs Extreme Degradation)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))
    plt.close()
    
    # Save results
    df_aug.to_csv(os.path.join(OUTPUT_DIR, 'anomaly_results.csv'), index=False)
    
    # Create comparison table
    comparison = {
        'Metric': ['Residual Mean', 'Residual Std', 'Threshold (3SD)', 'Detected Anomalies', 'ROC AUC'],
        'Value': [residual_mean, residual_std, threshold, num_anomalies, roc_auc]
    }
    pd.DataFrame(comparison).to_csv(os.path.join(OUTPUT_DIR, 'anomaly_metrics.csv'), index=False)
    print("Anomaly Detection Completed. Results saved.")

if __name__ == "__main__":
    evaluate_anomalies()
