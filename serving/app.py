import os
import joblib
import pandas as pd
from fastapi import FastAPI, status
from serving.schemas import ChurnPredictionInput, ChurnPredictionOutput

app = FastAPI(
    title="Customer Churn Prediction Microservice",
    description="Production ML API for real-time customer churn scoring.",
    version="1.0.0"
)

PREPROCESSOR = None
MODEL = None
THRESHOLD = 0.5

@app.on_event("startup")
def load_artifacts():
    global PREPROCESSOR, MODEL, THRESHOLD
    try:
        PREPROCESSOR = joblib.load("artifacts/preprocessing_pipeline.joblib")
        payload = joblib.load("artifacts/churn_model.joblib")
        MODEL = payload["model"]
        THRESHOLD = payload["threshold"]
        print("[Service Startup] ML Artifacts successfully loaded.")
    except Exception as e:
        print(f"[Service Startup Error] Failed to load artifacts: {str(e)}")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    if PREPROCESSOR is None or MODEL is None:
        return {"status": "degraded", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=ChurnPredictionOutput)
def predict_churn(input_data: ChurnPredictionInput):
    fallback_applied = False
    try:
        if PREPROCESSOR is None or MODEL is None:
            raise RuntimeError("Model artifacts uninitialized.")
            
        raw_dict = input_data.model_dump()
        df_input = pd.DataFrame([raw_dict])
        
        if pd.isna(df_input.loc[0, 'total_charges']):
            df_input.loc[0, 'total_charges'] = df_input.loc[0, 'tenure'] * df_input.loc[0, 'monthly_charges']

        processed_input = PREPROCESSOR.transform(df_input)
        prob = float(MODEL.predict_proba(processed_input)[0, 1])
        pred = int(prob >= THRESHOLD)
        
    except Exception as err:
        fallback_applied = True
        prob = 0.50
        pred = 1 if input_data.support_tickets_30d >= 3 and input_data.contract_type == "Month-to-Month" else 0
        
    return ChurnPredictionOutput(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        decision_threshold=round(THRESHOLD, 4),
        status="success",
        fallback_applied=fallback_applied
    )
