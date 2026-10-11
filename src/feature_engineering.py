import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def define_congestion_binary(volume):
    """
    Standard highway capacity threshold:
    - 0: Free Flow / Uncongested (< 3500 vph)
    - 1: Congested (>= 3500 vph)
    """
    return int(volume >= 3500)

def define_congestion_tier(volume):
    """
    3-Tier Level-of-Service congestion hierarchy:
    - 0: Free Flow (Low)       [< 2200 vph]
    - 1: Moderate Transit      [2200 - 4800 vph]
    - 2: Severe Congestion     [>= 4800 vph]
    """
    if volume < 2200:
        return 0
    elif volume < 4800:
        return 1
    else:
        return 2

CONGESTION_BINARY_LABELS = {
    0: "Free-Flow / Uncongested",
    1: "Congested / High Flow"
}

CONGESTION_TIER_LABELS = {
    0: "Low Flow (<2,200 vph)",
    1: "Moderate Transit (2,200-4,800 vph)",
    2: "Severe Congestion (>4,800 vph)"
}

def compute_diurnal_lookup(df):
    """
    Computes empirical average traffic volume by (hour, is_weekend) for fallback lag imputation.
    """
    lookup = df.groupby(['hour', 'is_weekend'])['traffic_volume'].mean().to_dict()
    return lookup

def engineer_features(df_raw, is_training=True, encoders=None):
    """
    Generates rich temporal, meteorological, cyclical, and autoregressive lag features.
    """
    df = df_raw.copy()

    # Datetime features
    if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['date_time'] = pd.to_datetime(df['date_time'])

    df = df.sort_values('date_time').reset_index(drop=True)

    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # Commuter and off-peak flags
    morning_rush = ((df['hour'] >= 6) & (df['hour'] <= 9) & (df['is_weekend'] == 0)).astype(int)
    evening_rush = ((df['hour'] >= 15) & (df['hour'] <= 18) & (df['is_weekend'] == 0)).astype(int)
    df['is_rush_hour'] = morning_rush | evening_rush
    df['is_offpeak'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    # Temperature in Celsius
    df['temp_c'] = df['temp'] - 273.15

    # Weather indicators
    df['is_raining'] = (df['rain_1h'] > 0.05).astype(int)
    df['is_snowing'] = (df['snow_1h'] > 0.05).astype(int)
    df['is_holiday'] = (df['holiday'] != 'None').astype(int)

    # Weather severity factor
    severe_weather = ['Thunderstorm', 'Squall', 'Snow']
    mild_weather = ['Rain', 'Drizzle', 'Mist', 'Fog', 'Haze']
    df['weather_severity'] = df['weather_main'].apply(
        lambda w: 2 if w in severe_weather else (1 if w in mild_weather else 0)
    )

    # Autoregressive lag and rolling features
    if 'traffic_volume' in df.columns:
        df['lag_1'] = df['traffic_volume'].shift(1).bfill()
        df['lag_2'] = df['traffic_volume'].shift(2).bfill()
        df['lag_24'] = df['traffic_volume'].shift(24).bfill()
        df['rolling_3'] = df['traffic_volume'].shift(1).rolling(3).mean().bfill()
        
        # Targets
        df['congestion_binary'] = df['traffic_volume'].apply(define_congestion_binary)
        df['congestion_tier'] = df['traffic_volume'].apply(define_congestion_tier)

    # Encoders & Diurnal Lookup Handling
    if is_training:
        weather_encoder = LabelEncoder()
        df['weather_encoded'] = weather_encoder.fit_transform(df['weather_main'].astype(str))
        diurnal_lookup = compute_diurnal_lookup(df)
        encoders = {
            'weather_encoder': weather_encoder,
            'diurnal_lookup': diurnal_lookup
        }
    else:
        weather_encoder = encoders.get('weather_encoder')
        known = set(weather_encoder.classes_)
        safe_weather = df['weather_main'].astype(str).apply(lambda x: x if x in known else list(known)[0])
        df['weather_encoded'] = weather_encoder.transform(safe_weather)
        
        # If lags not present during inference, impute using diurnal baseline
        diurnal_lookup = encoders.get('diurnal_lookup', {})
        if 'lag_1' not in df.columns or df['lag_1'].isna().any():
            baseline_vol = df.apply(lambda r: diurnal_lookup.get((r['hour'], r['is_weekend']), 3200), axis=1)
            df['lag_1'] = baseline_vol
            df['lag_2'] = baseline_vol * 0.98
            df['lag_24'] = baseline_vol
            df['rolling_3'] = baseline_vol

    return df, encoders

# Base feature list
BASE_FEATURE_COLS = [
    'hour', 'dayofweek', 'month', 'is_weekend', 'is_rush_hour', 'is_offpeak',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'temp_c', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_encoded',
    'is_raining', 'is_snowing', 'is_holiday', 'weather_severity',
    'lag_1', 'lag_2', 'lag_24', 'rolling_3'
]

# Hybrid feature inputs combining base features + unsupervised clustering outputs
HYBRID_FEATURE_COLS = BASE_FEATURE_COLS + ['cluster_regime', 'is_density_anomaly']
