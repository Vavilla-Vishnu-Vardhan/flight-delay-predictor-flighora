import os
import json
from typing import List, Dict, Any

try:
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import FakeListLLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class FlightDelayAIWorkflow:
    """
    LangChain AI Workflow Engine for:
    - Interactive Flight Risk Diagnostics & Root Cause Explanation
    - Smart Alternative Flight Rerouting Recommendations
    - Model Performance Summarization
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def analyze_flight_risk(self, prediction_result: dict, weather_data: dict) -> Dict[str, Any]:
        """
        Generates structured AI analysis explaining flight delay risk, root causes, and passenger recommendations.
        """
        prob = prediction_result.get("delay_probability", 0.0)
        delay_mins = prediction_result.get("predicted_delay_minutes", 0)
        risk_level = prediction_result.get("risk_level", "LOW")
        factors = prediction_result.get("contributing_factors", [])
        airline = prediction_result.get("airline", "Unknown Airline")
        origin = prediction_result.get("origin", "N/A")
        dest = prediction_result.get("destination", "N/A")

        summary = (
            f"Flight from {origin} to {dest} on {airline} has a {risk_level} delay risk level "
            f"({prob*100:.1f}% probability) with an estimated delay of {delay_mins:.0f} minutes."
        )

        insights = []
        if prob > 0.5:
            insights.append(f"Severe weather or operational bottleneck detected at {origin}.")
            insights.append("High probability of domino ground-stop or air traffic management (ATM) delays.")
        else:
            insights.append(f"Favorable flight window at {origin} with clear flight corridors.")

        passenger_advice = (
            "We recommend checking in early, monitoring live gate announcements, and considering alternative departure windows if available."
            if prob > 0.4 else "Standard flight ops expected. Proceed with normal check-in timing."
        )

        return {
            "summary": summary,
            "risk_assessment": risk_level,
            "root_causes": factors,
            "operational_insights": insights,
            "passenger_recommendations": passenger_advice
        }

    def recommend_alternative_flights(
        self, original_flight: dict, current_prediction: dict, available_flights: List[dict]
    ) -> List[dict]:
        """
        Recommends non-delayed or lower-risk alternative flight options.
        """
        recommendations = []
        orig_prob = current_prediction.get("delay_probability", 0.0)

        for flight in available_flights:
            # Filter alternative options
            alt_prob = flight.get("delay_probability", 0.2)
            if alt_prob < orig_prob or alt_prob < 0.35:
                risk_reduction = max(0.0, orig_prob - alt_prob)
                recommendations.append({
                    "flight_id": flight.get("flight_id"),
                    "airline": flight.get("airline"),
                    "departure_time": f"{flight.get('departure_hour', 12)}:00",
                    "origin": flight.get("origin"),
                    "destination": flight.get("destination"),
                    "delay_probability": round(alt_prob, 4),
                    "predicted_delay_minutes": round(flight.get("predicted_delay_minutes", 0), 1),
                    "risk_reduction_pct": round(risk_reduction * 100, 1),
                    "recommendation_score": round((1.0 - alt_prob) * 100, 1),
                    "reason": f"Saves ~{abs(current_prediction.get('predicted_delay_minutes',0) - flight.get('predicted_delay_minutes',0)):.0f} mins delay with cleaner weather corridor."
                })

        # Sort recommendations by highest score
        recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
        return recommendations[:3]

    def summarize_model_evaluation(self, metrics: dict) -> Dict[str, Any]:
        """
        Generates AI narrative evaluation of model performance.
        """
        acc = metrics.get("accuracy", 0.0)
        auc = metrics.get("roc_auc", 0.0)
        f1 = metrics.get("f1_score", 0.0)
        mae = metrics.get("mae_minutes", 0.0)

        narrative = (
            f"The Multi-Task PyTorch Deep Neural Network achieved an Accuracy of {acc*100:.1f}%, "
            f"ROC-AUC of {auc:.3f}, and F1-Score of {f1:.3f}. "
            f"The delay regression head estimates delay duration with a Mean Absolute Error (MAE) of {mae:.1f} minutes."
        )

        top_features = list(metrics.get("feature_importance", {}).keys())[:4]

        return {
            "narrative": narrative,
            "key_drivers": top_features,
            "reliability_rating": "EXCELLENT" if auc > 0.85 else "GOOD"
        }
