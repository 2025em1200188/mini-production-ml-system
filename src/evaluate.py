import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
from data_pipeline import TARGET_COL

def evaluate_test_set():
    preprocessor = joblib.load("artifacts/preprocessing_pipeline.joblib")
    artifact = joblib.load("artifacts/churn_model.joblib")
    model = artifact["model"]
    threshold = artifact["threshold"]
    
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop(columns=[TARGET_COL, 'customer_id'])
    y_test = test_df[TARGET_COL]
    
    X_test_proc = preprocessor.transform(X_test)
    probs = model.predict_proba(X_test_proc)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    precisions, recalls, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recalls, precisions)
    roc_auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print("================ MODEL EVALUATION SUMMARY ================")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"Decision Threshold: {threshold:.3f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, preds, digits=4))

if __name__ == "__main__":
    evaluate_test_set()
