import os
import numpy as np
import pandas as pd

def generate_flight_weather_dataset(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset combining flight schedules, weather conditions,
    and historical metrics to train a Deep Learning flight delay prediction model.
    """
    np.random.seed(random_seed)

    airlines = ['Delta', 'United', 'American', 'Southwest', 'JetBlue', 'Alaska']
    airports = ['JFK', 'LAX', 'ORD', 'ATL', 'DFW', 'SFO', 'DEN', 'MIA', 'SEA', 'BOS']

    airline_base_delay = {'Delta': 0.12, 'United': 0.18, 'American': 0.20, 'Southwest': 0.22, 'JetBlue': 0.25, 'Alaska': 0.14}

    # Generate synthetic attributes
    flight_ids = [f"FL-{1000 + i}" for i in range(n_samples)]
    selected_airlines = np.random.choice(airlines, size=n_samples)
    origins = np.random.choice(airports, size=n_samples)
    destinations = [
        np.random.choice([a for a in airports if a != o]) for o in origins
    ]

    dep_hours = np.random.randint(0, 24, size=n_samples)
    day_of_week = np.random.randint(0, 7, size=n_samples)
    months = np.random.randint(1, 13, size=n_samples)
    distances = np.random.randint(200, 3000, size=n_samples)

    # Weather attributes
    temperatures = np.random.normal(loc=15, scale=12, size=n_samples).round(1)
    wind_speeds = np.clip(np.random.exponential(scale=15, size=n_samples), 0, 70).round(1)
    visibilities = np.clip(np.random.normal(loc=10, scale=3, size=n_samples), 0.5, 15.0).round(1)
    precipitations = np.clip(np.random.exponential(scale=2.0, size=n_samples) * (np.random.rand(n_samples) > 0.6), 0, 45).round(1)
    pressures = np.random.normal(loc=1013, scale=8, size=n_samples).round(1)
    humidities = np.clip(np.random.normal(loc=65, scale=18, size=n_samples), 20, 100).round(1)

    congestion_index = np.clip(1.0 + (dep_hours >= 7) * (dep_hours <= 10) * 1.5 + (dep_hours >= 16) * (dep_hours <= 20) * 2.0 + np.random.normal(0, 0.5, n_samples), 1.0, 5.0).round(2)
    historical_airline_delay_rate = np.array([airline_base_delay[a] for a in selected_airlines])

    # Compute realistic latent delay score
    delay_score = (
        (wind_speeds > 35) * 1.8 +
        (visibilities < 3.0) * 2.2 +
        (precipitations > 5.0) * 1.6 +
        (congestion_index > 3.2) * 1.4 +
        (dep_hours >= 17) * (dep_hours <= 21) * 0.9 +
        (historical_airline_delay_rate * 3.5) +
        np.random.normal(0, 0.8, n_samples)
    )

    is_delayed = (delay_score > 2.2).astype(int)
    
    # Delay minutes calculation
    delay_minutes = np.where(
        is_delayed == 1,
        np.clip((delay_score * 18 + np.random.exponential(scale=20, size=n_samples)), 15, 240).round(0),
        0
    )

    df = pd.DataFrame({
        'flight_id': flight_ids,
        'airline': selected_airlines,
        'origin': origins,
        'destination': destinations,
        'departure_hour': dep_hours,
        'day_of_week': day_of_week,
        'month': months,
        'distance_miles': distances,
        'temperature_c': temperatures,
        'wind_speed_kmh': wind_speeds,
        'visibility_km': visibilities,
        'precipitation_mm': precipitations,
        'pressure_hpa': pressures,
        'humidity_pct': humidities,
        'congestion_index': congestion_index,
        'airline_delay_rate': historical_airline_delay_rate,
        'is_delayed': is_delayed,
        'delay_minutes': delay_minutes
    })

    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_flight_weather_dataset(5000)
    df.to_csv('data/flight_weather_dataset.csv', index=False)
    print(f"Generated {len(df)} records saved to data/flight_weather_dataset.csv")
