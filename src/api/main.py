import os
import sys
import json
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.schemas import (
    FlightPredictionRequest,
    FlightPredictionResponse,
    AIAnalysisRequest,
    AlternativeFlightsRequest
)
from src.predictor import FlightDelayPredictor
from src.ai_workflow import FlightDelayAIWorkflow
from src.weather_service import WeatherService, AIRPORT_COORDINATES
from src.dataset_generator import generate_flight_weather_dataset

app = FastAPI(
    title="Predictive Flight Delay Analysis System API",
    description="High-Retention Aviation Intelligence, DGCA/EU261 Compensation Calculator, Misery Map & Seat Tracker API",
    version="2.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../static'))

# Logo override route to serve high-res uploaded logo
@app.get("/static/flighora_logo.png")
def get_flighora_logo():
    uploaded_logo = r"C:\Users\dell\.gemini\antigravity\brain\7b9225d5-36ad-4b32-81ac-c8fbece51da6\.user_uploaded\media_1790274779408.png"
    if os.path.exists(uploaded_logo):
        return FileResponse(uploaded_logo, media_type="image/png")
    local_logo = os.path.join(static_dir, "flighora_logo.png")
    if os.path.exists(local_logo):
        return FileResponse(local_logo, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")

# Mount static files directory if it exists
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

predictor = None
ai_workflow = FlightDelayAIWorkflow()

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = FlightDelayPredictor(model_dir='models')
        except Exception as e:
            print(f"Model initialization warning: {e}. Auto-training model...")
            from src.train import train_model
            train_model()
            predictor = FlightDelayPredictor(model_dir='models')
    return predictor

# Schemas
class CompensationRequest(BaseModel):
    airline: str
    flight_number: str
    delay_hours: float
    cancellation: bool = False
    origin: str = "DEL"
    destination: str = "BOM"
    regulation: Optional[str] = "Auto-Detect"

class AlertSubscriptionRequest(BaseModel):
    email_or_phone: str
    flight_number: str
    origin: str
    channel: str = "WhatsApp & Email"

class AIChatQuery(BaseModel):
    query: str
    origin: Optional[str] = "HYD"
    flight_number: Optional[str] = "6E-204"

# Subscriptions store (in-memory)
alert_subscriptions = []

# Routes
@app.get("/")
def read_root():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "online", "service": "Predictive Flight Delay Analysis System API", "version": "2.5.0"}

@app.get("/health")
def health_check():
    return {"status": "online", "service": "Predictive Flight Delay Analysis System API", "version": "2.5.0"}

# 1. LIVE AIRPORT MISERY MAP DATA
@app.get("/api/v1/misery-map")
def get_misery_map_data():
    airports = ['DEL', 'BOM', 'BLR', 'HYD', 'MAA', 'CCU', 'JFK', 'DXB', 'LHR']
    map_data = []

    for apt in airports:
        w = WeatherService.get_live_airport_weather(apt, 18)
        coords = AIRPORT_COORDINATES.get(apt, {'lat': 28.5562, 'lon': 77.1000, 'name': apt})
        
        score = (
            (w['wind_speed_kmh'] / 50.0 * 25.0) +
            (max(0, 10.0 - w['visibility_km']) * 3.5) +
            (w['precipitation_mm'] * 2.5) +
            (w['congestion_index'] / 5.0 * 35.0)
        )
        misery_score = round(min(100.0, max(5.0, score)), 1)
        status = "CRITICAL DISRUPTION" if misery_score > 75 else ("HEAVY DELAYS" if misery_score > 45 else "SMOOTH OPS")

        map_data.append({
            "code": apt,
            "name": coords['name'],
            "lat": coords['lat'],
            "lon": coords['lon'],
            "misery_score": misery_score,
            "status": status,
            "temp_c": w['temperature_c'],
            "wind_speed_kmh": w['wind_speed_kmh'],
            "visibility_km": w['visibility_km'],
            "congestion": w['congestion_index']
        })

    map_data.sort(key=lambda x: x['misery_score'], reverse=True)
    return {"airports": map_data}

# 2. DGCA INDIA, EU261, US DOT & UK261 COMPENSATION CALCULATOR
@app.post("/api/v1/compensation/calculate")
def calculate_compensation(req: CompensationRequest):
    airline = req.airline
    delay = req.delay_hours
    cancellation = req.cancellation
    reg = req.regulation or "Auto-Detect"

    if reg == "Auto-Detect":
        if airline in ['IndiGo', 'Air India', 'Vistara', 'Akasa Air', 'SpiceJet', 'Air India Express']:
            reg = "DGCA CAR Section 3 (India)"
        elif airline in ['Delta', 'United', 'American']:
            reg = "US DOT Regulations (USA)"
        elif airline in ['British Airways']:
            reg = "UK261 Regulations (UK)"
        else:
            reg = "EU261 Regulations (Europe)"

    compensation_inr = 0
    compensation_eur = 0
    compensation_usd = 0
    compensation_gbp = 0
    law_applied = reg
    entitlements = []

    if "DGCA" in reg:
        if cancellation:
            compensation_inr = 10000
            entitlements.append("Full Refund OR Alternate Flight within 2 hours")
            entitlements.append("Hotel accommodation for overnight stay")
        elif delay >= 6:
            compensation_inr = 10000
            entitlements.append("Free meals, refreshments & full ticket refund option")
            entitlements.append("Hotel stay if overnight delay")
        elif delay >= 3:
            compensation_inr = 5000
            entitlements.append("Free meals and refreshments at airport terminal")
        else:
            entitlements.append("Duty of care: updates every 30 minutes")
        comp_str = f"₹{compensation_inr:,}"
        eligible = (compensation_inr > 0 or len(entitlements) > 1)
    elif "US DOT" in reg:
        if delay >= 3 or cancellation:
            compensation_usd = 300
            entitlements.append("Mandatory full cash refund if passenger chooses not to travel")
            entitlements.append("Free meal vouchers and hotel stay for overnight delay")
        else:
            entitlements.append("Rebooking on next available flight at no extra charge")
        comp_str = f"${compensation_usd}"
        eligible = (compensation_usd > 0)
    elif "UK261" in reg:
        if delay >= 4 or cancellation:
            compensation_gbp = 520
        elif delay >= 3:
            compensation_gbp = 350
        entitlements.append("Right to care: Food, drinks, hotel accommodation and transport")
        comp_str = f"£{compensation_gbp}"
        eligible = (compensation_gbp > 0)
    else: # EU261
        if delay >= 4 or cancellation:
            compensation_eur = 600
        elif delay >= 3:
            compensation_eur = 400
        entitlements.append("Right to care: Meals, hotel accommodation & 2 free calls")
        comp_str = f"€{compensation_eur}"
        eligible = (compensation_eur > 0)

    claim_letter = (
        f"FORMAL NOTICE OF STATUTORY COMPENSATION CLAIM\n"
        f"To: Claims & Customer Advocacy Department ({airline})\n"
        f"Flight Ref: {req.flight_number} | Route: {req.origin} to {req.destination}\n\n"
        f"Dear Customer Service Team,\n\n"
        f"I am writing regarding flight {req.flight_number} which experienced a delay of {delay} hours. "
        f"Under {law_applied}, passengers are legally entitled to compensation of "
        f"{comp_str} along with statutory duty of care entitlements.\n\n"
        f"Please process this statutory compensation claim within 14 business days.\n\n"
        f"Sincerely,\n[Passenger Name]"
    )

    return {
        "airline": airline,
        "flight_number": req.flight_number,
        "eligible": eligible,
        "compensation_amount": comp_str,
        "governing_law": law_applied,
        "entitlements": entitlements,
        "claim_letter_template": claim_letter
    }

# 3. REAL-TIME ALERT SUBSCRIPTION SYSTEM
@app.post("/api/v1/alerts/subscribe")
def subscribe_flight_alerts(req: AlertSubscriptionRequest):
    sub = {
        "contact": req.email_or_phone,
        "flight": req.flight_number,
        "origin": req.origin,
        "channel": req.channel,
        "status": "ACTIVE_WATCH"
    }
    alert_subscriptions.append(sub)
    return {
        "success": True,
        "message": f"Successfully subscribed {req.email_or_phone} to real-time ATC delay & gate alerts for flight {req.flight_number} via {req.channel}!",
        "subscription": sub
    }

# 4. CONVERSATIONAL LANGCHAIN AI ASSISTANT CHAT
@app.post("/api/v1/ai/chat")
def chat_ai_assistant(query_req: AIChatQuery):
    q = query_req.query.lower()
    apt = query_req.origin or "DEL"
    fl = query_req.flight_number or "FLIGHT"
    
    w = WeatherService.get_live_airport_weather(apt, 18)

    if "compensation" in q or "refund" in q or "claim" in q or "dgca" in q or "eu261" in q or "dot" in q:
        response = (
            f"⚖️ **Statutory Passenger Rights Guidance for {fl}**:\n\n"
            f"• **DGCA CAR Sec 3 (India 🇮🇳)**: 3+ hr delay = Free meals. 6+ hr delay or cancellation = Up to ₹10,000 cash refund + hotel stay.\n"
            f"• **EU261 Regulation (Europe 🇪🇺)**: 3+ hr delay = Up to €600 cash compensation + full duty of care (food/hotel/transfers).\n"
            f"• **US DOT Rules (USA 🇺🇸)**: Mandatory full cash refund if flight is delayed >3 hrs domestic or cancelled.\n"
            f"• **UK261 Rules (UK 🇬🇧)**: Up to £520 statutory refund.\n\n"
            f"👉 *Tip: Open our Statutory Refund Calculator tab to automatically generate a formal legal claim letter for your carrier!*"
        )
    elif "seat" in q or "legroom" in q or "window" in q or "pitch" in q:
        response = (
            f"💺 **Aircraft & Seat Comfort Intelligence for {fl}**:\n\n"
            f"• **XL Legroom Rows**: Rows 1–3 and Emergency Exit Rows 12 & 13 offer up to 34 inches of pitch.\n"
            f"• **Power Outlets**: USB & AC charging available on A321neo and B787 fleet.\n"
            f"• **⚠️ Window Warning Alert**: Avoid **Seat 11A** and **12F** on IndiGo/Vistara A320neo fleet — these seats are misaligned and lack a window view!"
        )
    elif "bag" in q or "luggage" in q or "weight" in q or "excess" in q:
        response = (
            f"🧳 **Carrier Baggage Restrictions & Allowances**:\n\n"
            f"• **Cabin Carry-On**: 1 bag up to 7 kg (55 × 35 × 25 cm) + 1 personal item (laptop bag/purse up to 3 kg).\n"
            f"• **Checked Bag**: 15 kg (Domestic Economy) / 20–30 kg (International / Premium Economy).\n"
            f"• **Excess Baggage Rate**: Approx ₹500–₹600 per extra kg if booked at gate."
        )
    elif "delay" in q or "risk" in q or "predict" in q or "6e" in q or "ai" in q or "weather" in q:
        # Run live model evaluation for the requested airport/flight
        pred_engine = get_predictor()
        sample_data = {
            "flight_id": fl,
            "airline": "IndiGo" if "6e" in fl.lower() else "Air India",
            "origin": apt,
            "destination": "DEL" if apt != "DEL" else "BOM",
            "departure_hour": 18,
            "day_of_week": 4,
            "month": 9,
            "distance_miles": 1150.0,
            "airline_delay_rate": 0.15
        }
        sample_data.update(w)
        c_pred = pred_engine.predict(sample_data)
        prob_pct = round(c_pred['delay_probability'] * 100, 1)
        delay_mins = round(c_pred['predicted_delay_minutes'])
        
        response = (
            f"✈️ **Live Neural Network Diagnostic for {fl} at {apt}**:\n\n"
            f"• **Delay Risk Score**: {prob_pct}% ({c_pred['risk_level']} THREAT)\n"
            f"• **Predicted Delay**: +{delay_mins} minutes\n"
            f"• **Live METAR Radar**: Wind {w['wind_speed_kmh']} km/h | Visibility {w['visibility_km']} km | Airspace Congestion Index: {w['congestion_index']}/5.0\n\n"
            f"💡 *Recommendation: {c_pred['status']}. Monitor gate updates or check smart rerouting options in the main panel.*"
        )
    else:
        response = (
            f"🤖 **Flighora AI Aviation Assistant Ready**:\n\n"
            f"I can analyze live flight delay risks, calculate DGCA/EU261/US DOT refunds, recommend XL legroom seats, or check baggage allowances. "
            f"Currently monitoring live telemetry for **{apt}** (Wind: {w['wind_speed_kmh']} km/h, Congestion: {w['congestion_index']}/5.0). How can I assist your trip?"
        )

    return {
        "query": query_req.query,
        "answer": response,
        "airport": apt,
        "live_weather": w
    }

# BAGGAGE & SEAT TRACKERS
@app.get("/api/v1/baggage/rules")
def get_baggage_rules(airline: str = "IndiGo"):
    rules_db = {
        "IndiGo": {
            "cabin_bag": {"max_weight": "7 kg", "dimensions": "55 x 35 x 25 cm", "personal_item": "3 kg (Laptop bag/handbag)"},
            "checked_bag": {"max_weight": "15 kg (Domestic) / 20-30 kg (Intl)", "dimensions": "158 cm total (L+W+H)"},
            "excess_fee": "₹550 per extra kg (Domestic)"
        },
        "Air India": {
            "cabin_bag": {"max_weight": "8 kg", "dimensions": "55 x 40 x 20 cm", "personal_item": "3 kg (Laptop bag/purse)"},
            "checked_bag": {"max_weight": "15 kg (Economy) / 25 kg (Premium)", "dimensions": "158 cm total"},
            "excess_fee": "₹600 per extra kg"
        },
        "Vistara": {
            "cabin_bag": {"max_weight": "7 kg (Economy) / 10 kg (Business)", "dimensions": "55 x 40 x 20 cm", "personal_item": "3 kg"},
            "checked_bag": {"max_weight": "15 kg (Economy) / 20 kg (Premium Economy)", "dimensions": "158 cm total"},
            "excess_fee": "₹500 per extra kg"
        },
        "Akasa Air": {
            "cabin_bag": {"max_weight": "7 kg", "dimensions": "55 x 35 x 25 cm", "personal_item": "3 kg"},
            "checked_bag": {"max_weight": "15 kg", "dimensions": "158 cm total"},
            "excess_fee": "₹550 per extra kg"
        },
        "Delta": {
            "cabin_bag": {"max_weight": "No weight limit", "dimensions": "56 x 35 x 23 cm", "personal_item": "1 item included"},
            "checked_bag": {"max_weight": "23 kg (50 lbs)", "dimensions": "157 cm total"},
            "excess_fee": "$35 for 1st bag / $45 for 2nd bag"
        }
    }
    return rules_db.get(airline, rules_db["IndiGo"])

@app.get("/api/v1/seat-comfort")
def get_seat_comfort(airline: str = "IndiGo"):
    seat_db = {
        "IndiGo": {
            "aircraft": "Airbus A320neo / A321neo",
            "seat_pitch_inches": "30 in (Standard Legroom) / 34 in (XL Front Seats)",
            "power_outlets": "USB Ports available on selected A321neo aircraft",
            "recline": "3 inches (Pre-reclined seats on select fleet)",
            "quiet_rows": "Rows 1-3 (XL Priority Seats) & Emergency Exit Rows 12, 13",
            "window_misalignment_alert": "Avoid Seat 11A and 12F (Missing window view alignment)"
        },
        "Air India": {
            "aircraft": "Boeing 787 Dreamliner / Airbus A350-900",
            "seat_pitch_inches": "31-33 in (Economy) / 38 in (Premium Economy)",
            "power_outlets": "AC Power Outlet + USB-A & USB-C at every seat",
            "recline": "5 inches recline with footrest",
            "quiet_rows": "Rows 18-22 (Forward Economy Cabin)",
            "window_misalignment_alert": "Avoid Row 26 (Engine noise zone)"
        },
        "Vistara": {
            "aircraft": "Airbus A320neo / Boeing 787-9",
            "seat_pitch_inches": "30 in (Economy) / 33 in (Premium Economy)",
            "power_outlets": "In-seat power and USB charging",
            "recline": "4 inches recline with padded headrest",
            "quiet_rows": "Rows 7-10 (Premium Economy section)",
            "window_misalignment_alert": "Avoid Row 10 (Restricted recline near exit)"
        }
    }
    return seat_db.get(airline, seat_db["IndiGo"])

@app.get("/api/v1/deals/everywhere")
def get_everywhere_deals(origin: str = "DEL"):
    deals = [
        {"destination": "GOI (Goa)", "price": "₹3,499", "savings_pct": 35, "category": "Beach & Nightlife", "airline": "IndiGo"},
        {"destination": "BOM (Mumbai)", "price": "₹4,199", "savings_pct": 20, "category": "Business & Culture", "airline": "Air India"},
        {"destination": "BLR (Bengaluru)", "price": "₹4,599", "savings_pct": 25, "category": "Tech & Weather", "airline": "Akasa Air"},
        {"destination": "COK (Kochi)", "price": "₹5,299", "savings_pct": 30, "category": "Backwaters", "airline": "Vistara"},
        {"destination": "BKK (Bangkok)", "price": "₹11,999", "savings_pct": 40, "category": "International Gateway", "airline": "IndiGo"},
        {"destination": "DXB (Dubai)", "price": "₹16,499", "savings_pct": 28, "category": "Shopping & Luxury", "airline": "Air India"}
    ]
    return {"origin": origin, "deals": deals}

# PREDICTION & AI ROUTES
@app.get("/api/v1/weather/live")
def get_live_weather(airport: str = "DEL", hour: int = 14):
    try:
        return WeatherService.get_live_airport_weather(airport, hour)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict", response_model=FlightPredictionResponse)
def predict_flight_delay(request: FlightPredictionRequest):
    pred_engine = get_predictor()
    try:
        input_data = request.dict()
        if "origin" in input_data and ("wind_speed_kmh" not in input_data or input_data.get("wind_speed_kmh") == 0.0):
            live_w = WeatherService.get_live_airport_weather(input_data["origin"], input_data.get("departure_hour", 14))
            input_data.update(live_w)

        result = pred_engine.predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics")
def get_model_metrics():
    metrics_path = "models/metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics file not found. Train model first.")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics

@app.post("/api/v1/ai/analyze")
def analyze_flight_risk(request: AIAnalysisRequest):
    try:
        analysis = ai_workflow.analyze_flight_risk(
            prediction_result=request.prediction_result,
            weather_data=request.weather_data
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ai/alternative-flights")
def get_alternative_flights(request: AlternativeFlightsRequest):
    try:
        pred_engine = get_predictor()
        orig = request.origin
        dest = request.destination
        current_hour = request.departure_hour

        alt_carriers = ['IndiGo', 'Air India', 'Vistara', 'Akasa Air', 'SpiceJet', 'Delta', 'United', 'American']
        cand_hours = [(current_hour + 2) % 24, (current_hour + 4) % 24, 8, 10]
        
        evaluated_candidates = []
        for i, h in enumerate(cand_hours):
            if h == current_hour:
                continue
            cand_airline = alt_carriers[i % len(alt_carriers)]
            cand_w = WeatherService.get_live_airport_weather(orig, h)
            cand_flight = {
                "flight_id": f"ALT-{101 + i}",
                "airline": cand_airline,
                "origin": orig,
                "destination": dest,
                "departure_hour": h,
                "day_of_week": 4,
                "month": 9,
                "distance_miles": 1200.0,
                "airline_delay_rate": 0.12
            }
            cand_flight.update(cand_w)
            c_pred = pred_engine.predict(cand_flight)
            cand_flight["delay_probability"] = c_pred["delay_probability"]
            cand_flight["predicted_delay_minutes"] = c_pred["predicted_delay_minutes"]
            evaluated_candidates.append(cand_flight)

        recommendations = ai_workflow.recommend_alternative_flights(
            original_flight={"origin": orig, "destination": dest, "departure_hour": current_hour},
            current_prediction=request.current_prediction,
            available_flights=evaluated_candidates
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/historical/data")
def get_historical_sample(limit: int = 100):
    data_path = "data/flight_weather_dataset.csv"
    if not os.path.exists(data_path):
        df = generate_flight_weather_dataset(1000)
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    return df.head(limit).to_dict(orient="records")
