import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERICAL_FEATURES = [
    'departure_hour', 'day_of_week', 'month', 'distance_miles',
    'temperature_c', 'wind_speed_kmh', 'visibility_km', 'precipitation_mm',
    'pressure_hpa', 'humidity_pct', 'congestion_index', 'airline_delay_rate'
]

CATEGORICAL_FEATURES = ['airline', 'origin', 'destination']

class FlightDataPreprocessor:
    def __init__(self):
        self.transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERICAL_FEATURES),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURES)
            ]
        )
        self.is_fitted = False
        self.feature_names = []

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        X_trans = self.transformer.fit_transform(X)
        self.is_fitted = True

        # Extract transformed feature names
        num_names = NUMERICAL_FEATURES
        cat_encoder = self.transformer.named_transformers_['cat']
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names = num_names + cat_names

        return X_trans

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform.")
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        return self.transformer.transform(X)

    def save(self, filepath: str):
        joblib.dump({
            'transformer': self.transformer,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'FlightDataPreprocessor':
        data = joblib.load(filepath)
        instance = cls()
        instance.transformer = data['transformer']
        instance.is_fitted = data['is_fitted']
        instance.feature_names = data['feature_names']
        return instance
