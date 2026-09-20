import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering step for customer churn prediction:
    - MonthlyChargesRatio: compares current monthly bill to average past bills
    - TotalServices: count of active add-on services
    - TenureCohort: groups tenure into 4 simple buckets (0-12m, 12-24m, etc.)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Drop customerID if it exists
        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'])

        # Fix blank spaces in TotalCharges
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0.0)

        # 1. Ratio of current monthly bill vs historical average
        avg_monthly = X['TotalCharges'] / (X['tenure'] + 1)
        X['MonthlyChargesRatio'] = X['MonthlyCharges'] / (avg_monthly + 0.001)

        # 2. Count of active services
        service_cols = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        active_services = [c for c in service_cols if c in X.columns]
        X['TotalServices'] = (X[active_services] == 'Yes').sum(axis=1)

        # 3. Tenure cohorts
        X['TenureCohort'] = pd.cut(
            X['tenure'],
            bins=[-1, 12, 24, 48, 100],
            labels=['0-12m', '12-24m', '24-48m', '48m+']
        ).astype(str)

        return X
