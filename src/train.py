import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score
from data_pipeline import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL, create_preprocessing_pipeline

def train_and_serialize():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    X_train = train_df.drop(columns=[TARGET_COL, 'customer_id'])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL, 'customer_id'])
    y_test = test_df[TARGET_COL]
    
    preprocessor = create_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(preprocessor, "artifacts/preprocessing_pipeline.joblib")
    
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train_proc, y_train)
    base_preds = baseline.predict(X_test_proc)
    print("--- BASELINE MODEL PERFORMANCE ---")
    print(classification_report(y_test, base_preds, zero_division=0))
    
    # Gradient Boosting Classifier with cost-weighted loss
    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_proc, y_train)
    
    y_probs = model.predict_proba(X_test_proc)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recalls, precisions)
    
    best_thresh = 0.5
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_test, (y_probs >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print("--- PRODUCTION MODEL PERFORMANCE (HistGradientBoosting) ---")
    print(f"PR-AUC Score: {pr_auc:.4f}")
    print(f"Optimal Decision Threshold (Macro F1): {best_thresh:.3f} | Best Macro F1: {best_f1:.4f}")
    
    final_preds = (y_probs >= best_thresh).astype(int)
    print(classification_report(y_test, final_preds))
    
    artifact_payload = {
        "model": model,
        "threshold": best_thresh,
        "pr_auc": pr_auc,
        "features": NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    }
    joblib.dump(artifact_payload, "artifacts/churn_model.joblib")
    print("[Train] Artifacts successfully serialized to artifacts/")

if __name__ == "__main__":
    train_and_serialize()
