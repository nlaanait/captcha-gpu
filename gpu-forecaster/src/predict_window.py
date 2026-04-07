import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import argparse
import sys
import logging

logging.getLogger("NP").setLevel(logging.ERROR)

def predict_best_timeslots(gpu_type, start_time=None, top_n=5):
    # Load the data
    df = pd.read_csv('../lambda-cloud-api/availability_stats.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    gpu_data = df[df['type'] == gpu_type].copy()
    if gpu_data.empty:
        print(f"No data found for GPU type: {gpu_type}")
        sys.exit(1)
        
    gpu_data = gpu_data[['timestamp', 'available']]
    gpu_data.columns = ['ds', 'y']
    gpu_data['y'] = gpu_data['y'].astype(float)
    
    # Resample to 1H blocks
    gpu_data.set_index('ds', inplace=True)
    gpu_data = gpu_data.resample('1H').max().reset_index()
    gpu_data['y'] = gpu_data['y'].fillna(0)
    
    gpu_data = gpu_data.sort_values('ds').reset_index(drop=True)
    
    # Train model
    print(f"Training model for {gpu_type} on {len(gpu_data)} hourly records...")
    m = NeuralProphet(epochs=50)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(gpu_data, freq='1H')
    
    # Determine the 24-hour window
    last_ds = gpu_data['ds'].max()
    
    if start_time is None:
        window_start = last_ds
    else:
        window_start = pd.to_datetime(start_time).floor('H')
        
    # We need to forecast enough periods to cover window_start + 24 hours
    # from the last known data point.
    hours_diff = int((window_start + pd.Timedelta(hours=24) - last_ds).total_seconds() / 3600)
    
    if hours_diff <= 0:
        print("The requested window is in the past compared to the dataset. Please provide a future date.")
        sys.exit(1)
        
    future = m.make_future_dataframe(gpu_data, periods=hours_diff)
    forecast = m.predict(future)
    
    # Filter the forecast to just the 24-hour window requested
    window_end = window_start + pd.Timedelta(hours=24)
    window_forecast = forecast[(forecast['ds'] >= window_start) & (forecast['ds'] < window_end)].copy()
    
    if window_forecast.empty:
        print("Could not generate forecast for the specified window.")
        sys.exit(1)
        
    # Sort by highest probability (yhat1)
    top_slots = window_forecast.sort_values('yhat1', ascending=False).head(top_n)
    
    print("\n" + "="*60)
    print(f"TOP {top_n} TIMESLOTS FOR {gpu_type.upper()}")
    print(f"Window: {window_start} to {window_end}")
    print("="*60)
    
    for i, (_, row) in enumerate(top_slots.iterrows(), 1):
        prob = row['yhat1']
        # Cap probability at 1.0 for display purposes and floor at 0
        prob_display = max(0.0, min(prob, 1.0)) * 100
        print(f"{i}. {row['ds']}  --  Availability Score: {prob_display:.2f}")
        
    return top_slots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict the best timeslots for GPU availability in a 24h window.')
    parser.add_argument('--gpu', type=str, required=True, help='GPU instance type')
    parser.add_argument('--start', type=str, default=None, help='Start time of the 24h window (YYYY-MM-DD HH:MM:SS). Defaults to end of dataset.')
    parser.add_argument('--top', type=int, default=5, help='Number of top timeslots to return.')
    
    args = parser.parse_args()
    predict_best_timeslots(args.gpu, args.start, args.top)
