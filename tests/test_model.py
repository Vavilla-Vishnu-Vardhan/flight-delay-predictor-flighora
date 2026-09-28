import os
import sys
import pytest
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset_generator import generate_flight_weather_dataset
from src.preprocessor import FlightDataPreprocessor
from src.model import FlightDelayMultiTaskNN
from src.predictor import FlightDelayPredictor

def test_dataset_generation():
    df = generate_flight_weather_dataset(100)
    assert len(df) == 100
    assert 'is_delayed' in df.columns
    assert 'delay_minutes' in df.columns

def test_preprocessor_and_model_forward():
    df = generate_flight_weather_dataset(50)
    preprocessor = FlightDataPreprocessor()
    X = preprocessor.fit_transform(df)

    assert X.shape[0] == 50
    assert X.shape[1] > 10

    model = FlightDelayMultiTaskNN(input_dim=X.shape[1])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    probs, mins = model(X_tensor)

    assert probs.shape == (50, 1)
    assert mins.shape == (50, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert (mins >= 0.0).all()
