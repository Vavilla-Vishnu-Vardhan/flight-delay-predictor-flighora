from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FlightPredictionRequest(BaseModel):
    flight_id: Optional[str] = Field(default="FL-1001", description="Unique flight identifier")
    airline: str = Field(default="Delta", description="Operating airline carrier")
    origin: str = Field(default="JFK", description="Origin airport code")
    destination: str = Field(default="LAX", description="Destination airport code")
    departure_hour: int = Field(default=14, ge=0, le=23, description="Scheduled departure hour (0-23)")
    day_of_week: int = Field(default=2, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    month: int = Field(default=7, ge=1, le=12, description="Month of departure (1-12)")
    distance_miles: float = Field(default=2475.0, ge=50, le=10000, description="Flight distance in miles")
    temperature_c: float = Field(default=22.5, description="Origin temperature in Celsius")
    wind_speed_kmh: float = Field(default=28.0, ge=0, description="Wind speed in km/h")
    visibility_km: float = Field(default=8.5, ge=0, le=20, description="Visibility in km")
    precipitation_mm: float = Field(default=1.2, ge=0, description="Precipitation rate in mm")
    pressure_hpa: float = Field(default=1012.5, description="Atmospheric pressure in hPa")
    humidity_pct: float = Field(default=65.0, ge=0, le=100, description="Humidity percentage")
    congestion_index: float = Field(default=3.1, ge=1.0, le=5.0, description="Airport airspace congestion index")
    airline_delay_rate: float = Field(default=0.15, ge=0.0, le=1.0, description="Historical airline delay rate")

class FlightPredictionResponse(BaseModel):
    flight_id: str
    airline: str
    origin: str
    destination: str
    delay_probability: float
    predicted_delay_minutes: float
    risk_level: str
    status: str
    contributing_factors: List[str]

class AIAnalysisRequest(BaseModel):
    prediction_result: Dict[str, Any]
    weather_data: Dict[str, Any]

class AlternativeFlightsRequest(BaseModel):
    origin: str
    destination: str
    departure_hour: int
    current_prediction: Dict[str, Any]

class ModelMetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    mae_minutes: float
    rmse_minutes: float
    dataset_size: int
    feature_importance: Dict[str, float]
