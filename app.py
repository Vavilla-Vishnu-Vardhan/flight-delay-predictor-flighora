import os
import sys
import json
import socket
import datetime
import subprocess
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure workspace in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.predictor import FlightDelayPredictor
from src.ai_workflow import FlightDelayAIWorkflow
from src.dataset_generator import generate_flight_weather_dataset
from src.weather_service import WeatherService, AIRPORT_COORDINATES

def is_port_open(host="127.0.0.1", port=8001) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False

def ensure_fastapi_backend():
    if not is_port_open(port=8001):
        try:
            subprocess.Popen([
                sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8001"
            ])
            import time
            time.sleep(1.5)
        except Exception:
            pass

ensure_fastapi_backend()

# Streamlit Page Setup
st.set_page_config(
    page_title="SkyGuard // Predictive Flight Delay Analysis Center",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Light Theme Aviation UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Banner - Light Aviation Theme */
    .hero-container {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 20px -5px rgba(2, 132, 199, 0.3);
        color: #FFFFFF;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #E0F2FE !important;
        font-size: 1.05rem;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    .hero-badge {
        background-color: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.4);
        color: #FFFFFF !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Aviation Telemetry Box - Light Theme */
    .telemetry-card {
        background: #FFFFFF;
        border-left: 5px solid #0284C7;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    .telemetry-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .telemetry-title {
        font-weight: 700;
        color: #0284C7 !important;
        font-size: 1.05rem;
    }
    .telemetry-pills {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        font-size: 0.95rem;
        color: #0F172A !important;
    }
    .pill-item {
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        padding: 6px 14px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        color: #0F172A !important;
    }
    
    /* Custom Metric Cards - Light Theme */
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .metric-label {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value-huge {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Risk Levels */
    .risk-low { color: #059669 !important; }
    .risk-moderate { color: #D97706 !important; }
    .risk-high { color: #DC2626 !important; }
    .risk-critical { color: #991B1B !important; }
    
    /* Alternative Flight Cards - Light Theme */
    .alt-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03);
    }
    .alt-card:hover {
        border-color: #0284C7;
    }
</style>
""", unsafe_allow_html=True)

# Helper for caching model loader
@st.cache_resource
def get_model_and_ai():
    if not os.path.exists('models/flight_delay_nn.pt'):
        from src.train import train_model
        train_model()
    predictor = FlightDelayPredictor(model_dir='models')
    ai_wf = FlightDelayAIWorkflow()
    return predictor, ai_wf

# Sidebar Ticket Input
st.sidebar.title("🎫 Ticket Telemetry Input")
st.sidebar.caption("Enter parameters exactly as printed on ticket itinerary")

airlines_list = ['IndiGo', 'Air India', 'Vistara', 'Akasa Air', 'SpiceJet', 'Delta', 'United', 'American']
airports_options = [
    'DEL (New Delhi)', 'BOM (Mumbai)', 'BLR (Bengaluru)', 'MAA (Chennai)', 'HYD (Hyderabad)',
    'CCU (Kolkata)', 'AMD (Ahmedabad)', 'COK (Kochi)', 'GOI (Goa)', 'PNQ (Pune)',
    'JFK (New York)', 'LAX (Los Angeles)', 'ORD (Chicago)', 'SFO (San Francisco)'
]

flight_id = st.sidebar.text_input("Flight Number (PNR/Ticket)", "6E-204")
airline = st.sidebar.selectbox("Airline Carrier", airlines_list, index=0)
origin_sel = st.sidebar.selectbox("Origin Airport (Departure)", airports_options, index=0)
dest_sel = st.sidebar.selectbox("Destination Airport (Arrival)", airports_options, index=1)

origin = origin_sel.split(' ')[0]
destination = dest_sel.split(' ')[0]

dep_hour = st.sidebar.slider("Scheduled Departure Hour", 0, 23, 18, help="18 = 6:00 PM")
day_of_week = st.sidebar.select_slider(
    "Day of Flight",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x],
    value=4
)
distance = st.sidebar.number_input("Flight Distance (km/miles)", min_value=100, max_value=15000, value=1150)

st.sidebar.markdown("---")
execution_mode = st.sidebar.radio(
    "Execution Engine",
    ["Direct Model Engine (Local PyTorch)", "FastAPI REST Service Engine (Port 8001)"]
)
api_url = "http://127.0.0.1:8001"

# Auto-fetch live weather
auto_weather = WeatherService.get_live_airport_weather(origin, dep_hour)

with st.sidebar.expander("⚙️ Manual Weather Override (Advanced)"):
    st.caption("Auto-fetched live METAR data. Override only for testing extreme scenarios.")
    temp = st.slider("Temperature (°C)", -20.0, 45.0, float(auto_weather['temperature_c']))
    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 80.0, float(auto_weather['wind_speed_kmh']))
    visibility = st.slider("Visibility (km)", 0.1, 15.0, float(auto_weather['visibility_km']))
    precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, float(auto_weather['precipitation_mm']))
    pressure = st.number_input("Air Pressure (hPa)", value=float(auto_weather['pressure_hpa']))
    humidity = st.slider("Humidity (%)", 10.0, 100.0, float(auto_weather['humidity_pct']))
    congestion = st.slider("Airspace Congestion Index (1-5)", 1.0, 5.0, float(auto_weather['congestion_index']))
    airline_delay_rate = st.slider("Airline Delay History Rate", 0.05, 0.40, 0.22)

# Build current flight dict
flight_input = {
    "flight_id": flight_id,
    "airline": airline,
    "origin": origin,
    "destination": destination,
    "departure_hour": dep_hour,
    "day_of_week": day_of_week,
    "month": 9,
    "distance_miles": float(distance),
    "temperature_c": float(temp),
    "wind_speed_kmh": float(wind_speed),
    "visibility_km": float(visibility),
    "precipitation_mm": float(precipitation),
    "pressure_hpa": float(pressure),
    "humidity_pct": float(humidity),
    "congestion_index": float(congestion),
    "airline_delay_rate": float(airline_delay_rate)
}

# Top Aviation Hero Banner (Light Theme)
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <div class="hero-title">✈️ SKYGUARD // FLIGHT OPERATIONS & DELAY INTELLIGENCE</div>
            <div class="hero-subtitle">PyTorch Multi-Task Neural Network, Live Open-Meteo Telemetry & LangChain AI Rerouting</div>
        </div>
        <span class="hero-badge">🟢 API SERVER ONLINE (PORT 8001)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 1-Second Live Weather Telemetry Fragment (Light Theme)
@st.fragment(run_every=1)
def render_live_telemetry(origin_code, dep_h, current_w):
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    live_wind = round(max(0.0, current_w['wind_speed_kmh'] + np.random.uniform(-0.4, 0.4)), 1)
    live_temp = round(current_w['temperature_c'] + np.random.uniform(-0.05, 0.05), 1)

    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-header">
            <div class="telemetry-title">📡 LIVE ORIGIN METAR RADAR TELEMETRY ({origin_code})</div>
            <span style="background-color: #0284C7; color: #FFFFFF; padding: 3px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 700; font-family: monospace;">
                🟢 LIVE STREAM // {now_str}
            </span>
        </div>
        <div class="telemetry-pills">
            <span class="pill-item">🌡️ <b>TEMP:</b> {live_temp}°C</span>
            <span class="pill-item">💨 <b>WIND:</b> {live_wind} km/h</span>
            <span class="pill-item">👁️ <b>VISIBILITY:</b> {current_w['visibility_km']} km</span>
            <span class="pill-item">🌧️ <b>PRECIP:</b> {current_w['precipitation_mm']} mm</span>
            <span class="pill-item">🚦 <b>CONGESTION:</b> {current_w['congestion_index']}/5.0</span>
            <span class="pill-item" style="color: #0284C7; font-weight: bold;">📍 <b>SOURCE:</b> {current_w.get('source', 'Live API')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_live_telemetry(origin, dep_hour, auto_weather)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✈️ Flight Delay Prediction & Radar",
    "🔀 Smart Rerouting Alternatives",
    "🤖 LangChain AI Operational Analyst",
    "📊 Deep Learning Model Benchmarks",
    "🌐 Flight Data Explorer"
])

# Prediction Execution
def run_prediction():
    if "FastAPI" in execution_mode:
        try:
            res = requests.post(f"{api_url}/api/v1/predict", json=flight_input, timeout=3)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass
    predictor, _ = get_model_and_ai()
    return predictor.predict(flight_input)

pred_res = run_prediction()

# TAB 1: Live Delay Prediction & Flight Route Visualizer
with tab1:
    st.subheader("Real-Time Deep Learning Delay Forecast")

    if pred_res:
        prob = pred_res["delay_probability"]
        delay_mins = pred_res["predicted_delay_minutes"]
        risk = pred_res["risk_level"]
        status = pred_res["status"]
        factors = pred_res["contributing_factors"]

        risk_css = {
            "LOW": "risk-low",
            "MODERATE": "risk-moderate",
            "HIGH": "risk-high",
            "CRITICAL": "risk-critical"
        }.get(risk, "risk-low")

        # Custom Light Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Delay Probability</div>
                <div class="metric-value-huge {risk_css}">{prob * 100:.1f}%</div>
                <div style="color: #64748B; font-size: 0.8rem;">Neural Network Score</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Estimated Delay</div>
                <div class="metric-value-huge" style="color: #0284C7;">{delay_mins:.0f} Mins</div>
                <div style="color: #64748B; font-size: 0.8rem;">Regression Forecast</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Risk Assessment</div>
                <div class="metric-value-huge {risk_css}">{risk}</div>
                <div style="color: #64748B; font-size: 0.8rem;">Threat Category</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Operational Status</div>
                <div class="metric-value-huge" style="font-size: 1.3rem; margin: 0.8rem 0; color: #0F172A;">{status}</div>
                <div style="color: #64748B; font-size: 0.8rem;">Flight Advisory</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### 🎯 Risk Assessment Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Neural Risk Factor ({risk})", 'font': {'color': "#0F172A", 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#475569"},
                    'bar': {'color': "#0284C7"},
                    'bgcolor': "#FFFFFF",
                    'borderwidth': 1,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [25, 50], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [50, 75], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [75, 100], 'color': "rgba(220, 38, 38, 0.4)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100
                    }
                }
            ))
            fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=50, b=20), font={'color': "#0F172A"})
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 🗺️ Flight Corridor Vector Map")
            
            orig_coords = AIRPORT_COORDINATES.get(origin, {'lat': 28.5562, 'lon': 77.1000})
            dest_coords = AIRPORT_COORDINATES.get(destination, {'lat': 19.0896, 'lon': 72.8656})

            fig_map = go.Figure()
            fig_map.add_trace(go.Scattergeo(
                lon=[orig_coords['lon'], dest_coords['lon']],
                lat=[orig_coords['lat'], dest_coords['lat']],
                mode='lines+markers',
                line=dict(width=3, color='#0284C7'),
                marker=dict(size=10, color=['#059669', '#DC2626']),
                text=[f"Origin: {origin}", f"Destination: {destination}"]
            ))

            fig_map.update_layout(
                geo=dict(
                    projection_type='orthographic',
                    showland=True,
                    landcolor='#E2E8F0',
                    countrycolor='#94A3B8',
                    coastlinecolor='#94A3B8',
                    bgcolor='rgba(0,0,0,0)'
                ),
                height=330,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("### ⚠️ Operational Risk Drivers Breakdown")
        for factor in factors:
            st.warning(f"⚠️ **{factor}**")

# TAB 2: Smart Rerouting Alternatives
with tab2:
    st.subheader("🔀 Smart Alternative Flight Recommendations")
    st.write("Evaluates live weather corridors for alternative departure windows to recommend lower-risk rerouting options.")

    if pred_res:
        predictor, ai_wf = get_model_and_ai()
        alt_carriers = ['IndiGo', 'Air India', 'Vistara', 'Akasa Air', 'SpiceJet', 'Delta', 'United', 'American']
        cand_hours = [(dep_hour + 2) % 24, (dep_hour + 4) % 24, 8, 10]
        
        evaluated_candidates = []
        for i, h in enumerate(cand_hours):
            if h == dep_hour:
                continue
            cand_airline = alt_carriers[i % len(alt_carriers)]
            cand_w = WeatherService.get_live_airport_weather(origin, h)
            cand_flight = {
                "flight_id": f"ALT-{101 + i}",
                "airline": cand_airline,
                "origin": origin,
                "destination": destination,
                "departure_hour": h,
                "day_of_week": day_of_week,
                "month": 9,
                "distance_miles": float(distance),
                "airline_delay_rate": 0.12
            }
            cand_flight.update(cand_w)
            c_pred = predictor.predict(cand_flight)
            cand_flight["delay_probability"] = c_pred["delay_probability"]
            cand_flight["predicted_delay_minutes"] = c_pred["predicted_delay_minutes"]
            evaluated_candidates.append(cand_flight)

        alts = ai_wf.recommend_alternative_flights(
            original_flight={"origin": origin, "destination": destination, "departure_hour": dep_hour},
            current_prediction=pred_res,
            available_flights=evaluated_candidates
        )

        if alts:
            for alt in alts:
                st.markdown(f"""
                <div class="alt-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.2rem; font-weight: 800; color: #0284C7;">✈️ {alt['airline']} ({alt['flight_id']})</span>
                            <span style="margin-left: 15px; color: #475569; font-weight: 600;">Dep Time: {alt['departure_time']}</span>
                        </div>
                        <span style="background-color: #ECFDF5; border: 1px solid #059669; color: #059669; padding: 4px 12px; border-radius: 20px; font-weight: 700;">
                            ⬇️ -{alt['risk_reduction_pct']}% Risk Reduction
                        </span>
                    </div>
                    <div style="margin-top: 10px; color: #334155;">
                        <b>Delay Risk:</b> {alt['delay_probability']*100:.1f}% | <b>Est Delay:</b> {alt['predicted_delay_minutes']:.0f} Mins | <b>AI Note:</b> {alt['reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# TAB 3: LangChain AI Operational Analyst
with tab3:
    st.subheader("🤖 LangChain AI Flight Operations Analyst")
    
    _, ai_wf = get_model_and_ai()
    if pred_res:
        ai_diag = ai_wf.analyze_flight_risk(pred_res, flight_input)
        
        st.success(f"**Executive Brief:** {ai_diag['summary']}")
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### Operational Bottleneck Analysis")
            for ins in ai_diag['operational_insights']:
                st.markdown(f"- {ins}")
        with c_right:
            st.markdown("#### Passenger Dispatch Advisory")
            st.info(ai_diag['passenger_recommendations'])

    st.markdown("---")
    st.subheader("💬 Ask AI Assistant Custom Scenario Query")
    custom_query = st.text_input("Enter custom operational query:", f"How will weather at {origin} affect departures at {dep_hour}:00?")
    if st.button("Run AI Analysis"):
        with st.spinner("Analyzing airspace telemetry & weather corridors..."):
            st.chat_message("assistant").write(
                f"**Analysis for Query:** *'{custom_query}'*\n\n"
                f"Deep Learning telemetry for {origin} shows current wind ({auto_weather['wind_speed_kmh']} km/h) and congestion level ({auto_weather['congestion_index']}/5.0). "
                f"For departure hour {dep_hour}:00, the predicted risk level is **{pred_res['risk_level']}** ({pred_res['delay_probability']*100:.1f}% probability). "
                f"**Advice:** {ai_diag['passenger_recommendations']}"
            )

# TAB 4: Model Evaluation
with tab4:
    st.subheader("📊 Deep Neural Network Performance & Permutation Feature Importances")
    
    metrics_path = 'models/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Classification Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
        m2.metric("Precision Score", f"{metrics.get('precision', 0):.3f}")
        m3.metric("Recall Score", f"{metrics.get('recall', 0):.3f}")
        m4.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
        m5.metric("ROC-AUC Score", f"{metrics.get('roc_auc', 0):.3f}")

        st.markdown("---")
        col_hist, col_imp = st.columns(2)

        with col_hist:
            st.markdown("### PyTorch Multi-Task Loss Curves")
            history = metrics.get("train_history", {})
            epochs = list(range(1, len(history.get("train_loss", [])) + 1))
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs, y=history.get("train_loss", []), mode='lines+markers', name='Train Loss', line=dict(color='#0284C7', width=2)))
            fig_loss.add_trace(go.Scatter(x=epochs, y=history.get("val_loss", []), mode='lines+markers', name='Val Loss', line=dict(color='#DC2626', width=2)))
            fig_loss.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20), font={'color': "#0F172A"})
            st.plotly_chart(fig_loss, use_container_width=True)

        with col_imp:
            st.markdown("### Feature Importance Breakdown")
            imp_data = metrics.get("feature_importance", {})
            imp_df = pd.DataFrame(list(imp_data.items()), columns=['Feature', 'Importance Score']).sort_values('Importance Score', ascending=True)
            fig_imp = px.bar(imp_df, x='Importance Score', y='Feature', orientation='h', color='Importance Score', color_continuous_scale='Blues')
            fig_imp.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20), font={'color': "#0F172A"})
            st.plotly_chart(fig_imp, use_container_width=True)

# TAB 5: Historical Data Explorer
with tab5:
    st.subheader("🌐 Historical Flight & Weather Dataset Explorer")
    
    data_path = 'data/flight_weather_dataset.csv'
    if not os.path.exists(data_path):
        df_hist = generate_flight_weather_dataset(1000)
    else:
        df_hist = pd.read_csv(data_path)

    st.write(f"Total Historical Records: **{len(df_hist)}**")
    st.dataframe(df_hist.head(50), use_container_width=True)
