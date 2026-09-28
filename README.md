# ✈️ Flighora - Predictive Flight Delay Intelligence System

An AI-powered prediction and decision support platform for flight delay forecasting, 1-second real-time METAR weather radar streaming, statutory passenger refund calculation, and smart alternative flight recommendations. Built using **PyTorch**, **FastAPI**, **Leaflet OpenStreetMap**, and **PyTorch Deep Neural Networks**.

---

## 🌟 Key Features

- **Multi-Task PyTorch Deep Neural Network**:
  1. **Binary Delay Risk Probability** (Classification)
  2. **Estimated Delay Duration in Minutes** (Regression)
- **1-Second Real-Time METAR Weather Radar**: Live 1-second streaming telemetry updates for temperature, wind speed, visibility, and airspace congestion index.
- **FlightAware-Inspired Web Interface**: Sleek dark navy interface featuring transparent vector branding, Flight Operational Parameters control, and interactive Leaflet OpenStreetMap airspace radar.
- **Multi-Region Statutory Refund Calculator**: Instant calculation of airline passenger refund rights under **DGCA CAR Section 3 (India 🇮🇳)**, **EU261 (Europe 🇪🇺)**, **US DOT (USA 🇺🇸)**, and **UK261 (UK 🇬🇧)**.
- **AI Flight Assistant Chatbot**: Integrated intelligent conversational assistant answering flight delay risk, baggage rules, seat pitch, and statutory refund queries.
- **Smart Alternative Flight Rerouting**: Analyzes alternative flight corridors to recommend non-delayed options with high risk reduction.

---

## 🏗️ System Architecture

```
flight-delay-analysis-system/
├── data/                       # Synthetic flight and weather dataset
├── models/                     # PyTorch model checkpoints, joblib scalers, metrics
├── src/
│   ├── dataset_generator.py    # Synthetic flight & weather data generator
│   ├── preprocessor.py         # Sklearn OneHotEncoder & StandardScaler transformer
│   ├── model.py                # PyTorch Multi-Task Deep Neural Network
│   ├── train.py                # Training script with metrics & feature importance
│   ├── predictor.py            # High-level inference engine
│   ├── ai_workflow.py          # AI operational analysis & rerouting engine
│   ├── weather_service.py      # Real-time METAR weather service with 1s streaming
│   └── api/
│       ├── main.py             # FastAPI REST service & static web routes
│       └── schemas.py          # Pydantic data validation models
├── static/                     # Web UI static assets & index.html application
├── tests/                      # Automated unit and API integration tests
│   ├── test_model.py
│   └── test_api.py
├── run_system.py               # Unified system orchestrator
└── requirements.txt            # Project dependencies
```

---

## 🚀 Quick Start & Usage

### 1. Launch the System
Launch the unified Flighora server:
```bash
python run_system.py
```

### 2. Access Web Interface
- ✈️ **Flighora Primary Web Application**: [http://127.0.0.1:8001/](http://127.0.0.1:8001/)
- 📄 **FastAPI Documentation (Swagger)**: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

### 3. Run Automated Tests
```bash
pytest tests/
```

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `POST /api/v1/predict` | `POST` | Serves PyTorch model delay probability & duration forecast |
| `GET /api/v1/weather/live` | `GET` | Serves 1-second real-time METAR airport weather radar |
| `POST /api/v1/compensation/calculate` | `POST` | Calculates statutory passenger refund amount (DGCA/EU261/USDOT) |
| `POST /api/v1/ai/chat` | `POST` | AI Flight Assistant conversational endpoint |
| `GET /api/v1/misery-map` | `GET` | Serves airport disruption index telemetry for Leaflet radar map |

---

## 📊 Deep Learning Model Performance

- **Classification Accuracy**: ~88% - 92%
- **ROC-AUC Score**: ~0.90+
- **Delay Duration MAE**: ~12 - 16 Minutes
- **Key Risk Features**: Wind Speed, Visibility, Congestion Index, Departure Hour, Historical Airline Delay Rate.
