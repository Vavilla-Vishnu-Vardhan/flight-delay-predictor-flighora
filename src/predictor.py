import os
import json
import torch
import pandas as pd
import numpy as np
from src.preprocessor import FlightDataPreprocessor
from src.model import FlightDelayMultiTaskNN

class FlightDelayPredictor:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'flight_delay_nn.pt')
        self.preprocessor_path = os.path.join(model_dir, 'preprocessors.joblib')
        self.metrics_path = os.path.join(model_dir, 'metrics.json')

        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError("Model artifacts missing. Train the model first using train.py.")

        self.preprocessor = FlightDataPreprocessor.load(self.preprocessor_path)
        input_dim = len(self.preprocessor.feature_names)

        self.model = FlightDelayMultiTaskNN(input_dim=input_dim)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.metrics = {}
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                self.metrics = json.load(f)

    def predict(self, flight_data: dict) -> dict:
        """
        Takes raw flight dictionary and returns delay probability, predicted duration, risk assessment & contributing factors.
        """
        df = pd.DataFrame([flight_data])
        X = self.preprocessor.transform(df)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            prob, mins = self.model(X_tensor)

        delay_probability = float(prob.item())
        delay_minutes = max(0.0, float(mins.item()))

        # Determine risk level
        if delay_probability < 0.25:
            risk_level = "LOW"
            status = "ON-TIME"
        elif delay_probability < 0.50:
            risk_level = "MODERATE"
            status = "SLIGHT DELAY RISK"
        elif delay_probability < 0.75:
            risk_level = "HIGH"
            status = "PROBABLE DELAY"
        else:
            risk_level = "CRITICAL"
            status = "HIGH DELAY RISK"

        # Key risk factors analysis
        contributing_factors = []
        if flight_data.get('wind_speed_kmh', 0) > 30:
            contributing_factors.append(f"High wind speeds ({flight_data['wind_speed_kmh']} km/h)")
        if flight_data.get('visibility_km', 10) < 4.0:
            contributing_factors.append(f"Low visibility ({flight_data['visibility_km']} km)")
        if flight_data.get('precipitation_mm', 0) > 4.0:
            contributing_factors.append(f"Heavy precipitation ({flight_data['precipitation_mm']} mm)")
        if flight_data.get('congestion_index', 1.0) > 3.0:
            contributing_factors.append(f"Airport airspace congestion index ({flight_data['congestion_index']}/5.0)")
        if flight_data.get('departure_hour', 12) in [17, 18, 19, 20]:
            contributing_factors.append(f"Evening rush hour departure time ({flight_data['departure_hour']}:00)")

        if not contributing_factors:
            contributing_factors.append("Standard flight operating conditions")

        return {
            "flight_id": flight_data.get("flight_id", "FL-CUSTOM"),
            "airline": flight_data.get("airline"),
            "origin": flight_data.get("origin"),
            "destination": flight_data.get("destination"),
            "delay_probability": round(delay_probability, 4),
            "predicted_delay_minutes": round(delay_minutes, 1),
            "risk_level": risk_level,
            "status": status,
            "contributing_factors": contributing_factors
        }
