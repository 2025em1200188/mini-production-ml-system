import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERICAL_FEATURES = ['tenure', 'monthly_charges', 'total_charges', 'support_tickets_30d']
CATEGORICAL_FEATURES = ['contract_type', 'payment_method', 'paperless_billing']
TARGET_COL = 'churn'

def generate_synthetic_churn_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.round(np.random.uniform(18.0, 120.0, size=n_samples), 2)
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 50, size=n_samples), 2)
    total_charges = np.maximum(total_charges, 0.0)
    support_tickets = np.random.poisson(lam=1.2, size=n_samples)
    
    contract_type = np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(['Electronic Check', 'Credit Card', 'Bank Transfer'], size=n_samples, p=[0.4, 0.3, 0.3])
    paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.7, 0.3])
    
    logit = (-1.5 
             + 0.03 * (monthly_charges - 60) 
             - 0.04 * tenure 
             + 0.45 * support_tickets 
             + 1.2 * (contract_type == 'Month-to-Month'))
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'customer_id': [f"CUST_{i+10000}" for i in range(n_samples)],
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'support_tickets_30d': support_tickets,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'paperless_billing': paperless_billing,
        'churn': churn
    })
    
    mask = np.random.rand(*df[NUMERICAL_FEATURES].shape) < 0.02
    for idx, col in enumerate(NUMERICAL_FEATURES):
        df.loc[mask[:, idx], col] = np.nan
        
    return df

def create_preprocessing_pipeline() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, NUMERICAL_FEATURES),
        ('cat', cat_pipeline, CATEGORICAL_FEATURES)
    ])
    return preprocessor

def run_ingestion_and_split(data_dir: str = "data", test_size: float = 0.2, random_state: int = 42):
    os.makedirs(data_dir, exist_ok=True)
    df = generate_synthetic_churn_data(n_samples=10000, random_state=random_state)
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(df, df[TARGET_COL]):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    print(f"[Data Pipeline] Saved train ({len(train_df)}) and test ({len(test_df)}) partitions.")

if __name__ == "__main__":
    run_ingestion_and_split()
