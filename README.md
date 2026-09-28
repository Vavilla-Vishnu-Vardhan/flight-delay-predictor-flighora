# ✈️ Predictive Flight Delay Analysis System

An AI-powered prediction and decision support system for flight-delay forecasting, operational risk analysis, and smart alternative flight recommendations. Built using **PyTorch**, **FastAPI**, **Streamlit**, and **LangChain**.

---

## 🌟 Key Features

- **Multi-Task Deep Neural Network**: PyTorch architecture trained on synthetic flight telemetry and weather parameters to simultaneously predict:
  1. **Binary Delay Risk Probability** (Classification)
  2. **Estimated Delay Duration in Minutes** (Regression)
- **LangChain AI Workflows**: Automated operational diagnostics, delay root cause analysis, passenger guidance, and custom scenario querying.
- **Smart Alternative Flight Rerouting**: Analyzes alternative flight corridors and departure windows to recommend non-delayed options with high risk reduction.
- **FastAPI REST Service**: Production-ready API endpoints exposing predictions, model metrics, feature importances, and AI services.
- **Interactive Streamlit Dashboard**: Rich visual interface featuring real-time risk gauges, loss curves, feature importance charts, and historical data exploration.

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
│   ├── ai_workflow.py          # LangChain operational analysis & rerouting engine
│   └── api/
│       ├── main.py             # FastAPI REST service & API routes
│       └── schemas.py          # Pydantic data validation models
├── tests/                      # Automated unit and API integration tests
│   ├── test_model.py
│   └── test_api.py
├── app.py                      # Interactive Streamlit Web UI
├── run_system.py               # Unified system orchestrator
└── requirements.txt            # Project dependencies
```

---

## 🚀 Quick Start & Usage

### 1. Run Automated Training & Tests
Train the PyTorch neural network and execute unit tests:
```bash
python src/train.py
pytest tests/
```

### 2. Launch the System (FastAPI + Streamlit)
Launch the unified system orchestrator:
```bash
python run_system.py
```
- **Streamlit Web UI**: `http://localhost:8501`
- **FastAPI Documentation**: `http://localhost:8001/docs`

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `POST /api/v1/predict` | `POST` | Serves PyTorch model delay probability & duration forecast |
| `GET /api/v1/metrics` | `GET` | Returns Accuracy, F1, ROC-AUC, MAE & feature importances |
| `POST /api/v1/ai/analyze` | `POST` | Triggers LangChain AI operational risk breakdown |
| `POST /api/v1/ai/alternative-flights` | `POST` | Generates lower-risk alternative flight recommendations |
| `GET /api/v1/historical/data` | `GET` | Fetches historical flight dataset samples |

---

## 📊 Deep Learning Model Performance

- **Classification Accuracy**: ~88% - 92%
- **ROC-AUC Score**: ~0.90+
- **Delay Duration MAE**: ~12 - 16 Minutes
- **Key Risk Features**: Wind Speed, Visibility, Congestion Index, Departure Hour, Historical Airline Delay Rate.
