import time
import numpy as np
import requests
import pandas as pd

def benchmark_online_api(endpoint_url: str = "http://127.0.0.1:8000/predict", n_requests: int = 200):
    payload = {
        "tenure": 24.0,
        "monthly_charges": 65.0,
        "total_charges": 1560.0,
        "support_tickets_30d": 1,
        "contract_type": "One-Year",
        "payment_method": "Bank Transfer",
        "paperless_billing": "No"
    }
    
    latencies = []
    print(f"[Profiling] Benchmarking REST API over {n_requests} sequential requests...")
    
    for _ in range(n_requests):
        t0 = time.perf_counter()
        try:
            res = requests.post(endpoint_url, json=payload)
            t1 = time.perf_counter()
            if res.status_code == 200:
                latencies.append((t1 - t0) * 1000.0)
        except Exception as e:
            print(f"API connection error: {e}")
            break
            
    if latencies:
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print("\n--- ONLINE API LATENCY BENCHMARK (ms) ---")
        print(f"p50 Latency: {p50:.2f} ms")
        print(f"p95 Latency: {p95:.2f} ms")
        print(f"p99 Latency: {p99:.2f} ms")

def calculate_psi(reference: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    buckets = np.linspace(0, 100, num_buckets + 1)
    quantiles = np.percentile(reference, buckets)
    quantiles[0] -= 1e-5
    quantiles[-1] += 1e-5
    
    ref_counts, _ = np.histogram(reference, bins=quantiles)
    curr_counts, _ = np.histogram(current, bins=quantiles)
    
    ref_pct = ref_counts / len(reference)
    curr_pct = curr_counts / len(current)
    
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    curr_pct = np.where(curr_pct == 0, 1e-4, curr_pct)
    
    psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi_value)

def run_drift_analysis():
    print("\n--- OPERATIONAL DRIFT MONITORING BLUEPRINT ---")
    try:
        ref_data = pd.read_csv("data/train.csv")['monthly_charges'].dropna().values
        curr_data = ref_data * 1.25 + np.random.normal(5, 2, size=len(ref_data))
        
        psi = calculate_psi(ref_data, curr_data)
        print("Feature Monitored: 'monthly_charges'")
        print(f"Calculated PSI: {psi:.4f}")
        if psi < 0.1:
            print("Drift Alert Level: LOW (No action required)")
        elif 0.1 <= psi < 0.2:
            print("Drift Alert Level: MODERATE (Queue model for retraining evaluation)")
        else:
            print("Drift Alert Level: CRITICAL (Trigger automated model retraining pipeline)")
    except Exception as e:
        print(f"Error in drift analysis: {e}")

if __name__ == "__main__":
    run_drift_analysis()
