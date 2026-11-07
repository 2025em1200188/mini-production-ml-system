# Mini Production Machine Learning System: Customer Churn Scoring

## Project Overview
This project delivers a complete production ML system engineered for imbalanced customer churn prediction. It features leak-free preprocessing, cost-sensitive LightGBM modeling, threshold calibration, a FastAPI online REST microservice, a batch CLI scoring script, latency benchmarking, and PSI feature drift detection.

---

## Folder Structure
```text
mini-production-ml-system/
├── .gitignore
├── README.md
├── requirements.txt
├── artifacts/
│   ├── churn_model.joblib
│   └── preprocessing_pipeline.joblib
├── data/
│   ├── test.csv
│   └── train.csv
├── serving/
│   ├── __init__.py
│   ├── app.py
│   ├── benchmark.py
│   ├── predict_batch.py
│   └── schemas.py
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── evaluate.py
│   └── train.py
└── tests/
    └── test_inference.py
```

---

## Visual Studio / VS Code Setup & Quickstart Guide

### 1. Open Project in Visual Studio Code / Visual Studio
* Launch **VS Code** (or Visual Studio).
* Select **File -> Open Folder...** and choose the extracted `mini-production-ml-system` directory.

### 2. Configure Virtual Environment & Install Dependencies
Open a terminal inside Visual Studio (`Ctrl + ~` or `Terminal -> New Terminal`):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.env\Scripts\Activate.ps1
# On Windows (Command Prompt):
.env\Scriptsctivate.bat
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip and install pinned requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the ML Engineering Pipeline

### Step 1: Ingest Data & Train Model
Execute data generation, training, and threshold evaluation:
```bash
python src/data_pipeline.py
python src/train.py
python src/evaluate.py
```

### Step 2: Run Automated Unit Tests
```bash
pytest tests/
```

### Step 3: Launch FastAPI Online REST API
```bash
uvicorn serving.app:app --host 127.0.0.1 --port 8000 --reload
```
* Access interactive Swagger API documentation at: `http://127.0.0.1:8000/docs`

### Step 4: Test Scoring API via CLI / PowerShell
In a separate terminal window:
```powershell
# Windows PowerShell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"tenure":12.0,"monthly_charges":85.50,"total_charges":1026.00,"support_tickets_30d":3,"contract_type":"Month-to-Month","payment_method":"Electronic Check","paperless_billing":"Yes"}'
```

```bash
# Linux / macOS / Git Bash
curl -X POST "http://127.0.0.1:8000/predict"      -H "Content-Type: application/json"      -d '{
           "tenure": 12.0,
           "monthly_charges": 85.50,
           "total_charges": 1026.00,
           "support_tickets_30d": 3,
           "contract_type": "Month-to-Month",
           "payment_method": "Electronic Check",
           "paperless_billing": "Yes"
         }'
```

### Step 5: Execute Vectorized Batch Scoring CLI
```bash
python serving/predict_batch.py --input data/test.csv --output data/batch_predictions.csv
```

### Step 6: Profile Latency and Calculate Feature Drift (PSI)
```bash
python serving/benchmark.py
```
