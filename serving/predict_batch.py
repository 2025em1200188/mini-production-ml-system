import argparse
import time
import os
import joblib
import pandas as pd

def run_batch_inference(input_path: str, output_path: str):
    start_time = time.time()
    print(f"[Batch CLI] Ingesting batch file: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    preprocessor = joblib.load("artifacts/preprocessing_pipeline.joblib")
    artifact = joblib.load("artifacts/churn_model.joblib")
    model = artifact["model"]
    threshold = artifact["threshold"]

    df = pd.read_csv(input_path)
    record_count = len(df)
    
    record_ids = df['customer_id'] if 'customer_id' in df.columns else df.index
    df_features = df.drop(columns=['customer_id', 'churn'], errors='ignore')

    processed_features = preprocessor.transform(df_features)
    probabilities = model.predict_proba(processed_features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    output_df = pd.DataFrame({
        "record_id": record_ids,
        "churn_probability": probabilities.round(4),
        "churn_prediction": predictions
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    elapsed_time = time.time() - start_time
    throughput = record_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"[Batch CLI] Successfully processed {record_count} records in {elapsed_time:.3f}s ({throughput:.2f} records/sec).")
    print(f"[Batch CLI] Predictions exported to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Scoring CLI for Churn System")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="data/batch_predictions.csv", help="Path to output CSV file")
    args = parser.parse_args()
    
    run_batch_inference(args.input, args.output)
