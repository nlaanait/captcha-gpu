import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, load
import argparse
import sys
import os
import logging
import warnings

logging.getLogger("NP").setLevel(logging.ERROR)

# Based on our evaluation, these GPUs benefit from weighted intra-hour polling.
# Others should use a uniform polling strategy to hedge against unpredictable noise.
WEIGHTED_GPUS = ['gpu_1x_a10', 'gpu_1x_gh200']

import json

def generate_api_strategy(gpu_type, start_time=None, top_n=5, as_json=False):
    import sys
    import os
    original_stdout = sys.stdout
    if as_json:
        sys.stdout = open(os.devnull, 'w')
        
    import warnings
    import logging
    import pytorch_lightning as pl
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    
    # Load 1H model
    model_path = f"models/{gpu_type}.np"
    if not os.path.exists(model_path):
        if not as_json: print(f"Pre-trained 1H model not found at {model_path}. Please run train_models.py first.")
        else: 
            sys.stdout.close()
            sys.stdout = original_stdout
        sys.exit(1)
        
    m_1h = load(model_path)

    
    # Load data context
    df = pd.read_csv('../lambda-cloud-api/availability_stats.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    gpu_data = df[df['type'] == gpu_type].copy()
    if gpu_data.empty:
        if not as_json: print(f"No data found for GPU type: {gpu_type}")
        sys.exit(1)
        
    gpu_data = gpu_data[['timestamp', 'available']]
    gpu_data.columns = ['ds', 'y']
    gpu_data['y'] = gpu_data['y'].astype(float)
    gpu_data.set_index('ds', inplace=True)
    
    df_1h = gpu_data.resample('1H').max().reset_index()
    df_1h['y'] = df_1h['y'].fillna(0)
    df_1h = df_1h.sort_values('ds').reset_index(drop=True)
    
    last_ds = df_1h['ds'].max()
    if start_time is None:
        window_start = last_ds
    else:
        window_start = pd.to_datetime(start_time).floor('H')
        
    hours_diff = int((window_start + pd.Timedelta(hours=24) - last_ds).total_seconds() / 3600)
    
    if hours_diff <= 0:
        if not as_json: print("The requested window is in the past compared to the dataset. Please provide a future date.")
        sys.exit(1)
        
    # Forecast 1H to find the best timeslots
    future_1h = m_1h.make_future_dataframe(df_1h, periods=hours_diff)
    forecast_1h = m_1h.predict(future_1h)
    
    window_end = window_start + pd.Timedelta(hours=24)
    window_forecast = forecast_1h[(forecast_1h['ds'] >= window_start) & (forecast_1h['ds'] < window_end)].copy()
    
    if window_forecast.empty:
        if not as_json: print("Could not generate forecast for the specified window.")
        sys.exit(1)
        
    top_slots = window_forecast.sort_values('yhat1', ascending=False).head(top_n)
    
    use_weighted = gpu_type in WEIGHTED_GPUS
    forecast_15m = None
    
    if use_weighted:
        if not as_json: print(f"Training 15-minute model for {gpu_type} to generate intra-hour polling weights...")
        df_15m = gpu_data.resample('15min').max().reset_index()
        df_15m['y'] = df_15m['y'].fillna(0)
        df_15m = df_15m.sort_values('ds').reset_index(drop=True)
        
        m_15m = NeuralProphet(epochs=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_15m.fit(df_15m, freq='15min')
            
        # Forecast 15m blocks up to the end of the window
        last_ds_15m = df_15m['ds'].max()
        periods_15m = int((window_end - last_ds_15m).total_seconds() / 60 / 15)
        future_15m = m_15m.make_future_dataframe(df_15m, periods=periods_15m)
        forecast_15m = m_15m.predict(future_15m)
        
    if not as_json:
        print("\n" + "="*80)
        print(f"API POLLING STRATEGY FOR {gpu_type.upper()}")
        print(f"Window: {window_start} to {window_end}")
        print("="*80)
    
    json_output = []
    
    for i, (_, row) in enumerate(top_slots.iterrows(), 1):
        hour_start = row['ds']
        hour_end = hour_start + pd.Timedelta(minutes=45) # 4th block starts at +45m
        prob_1h = max(0.0, min(row['yhat1'], 1.0)) * 100
        
        slot_data = {
            "timeslot_start": hour_start.strftime('%Y-%m-%d %H:%M:%S'),
            "timeslot_end": (hour_start + pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
            "availability_score": float(prob_1h),
            "strategy": "WEIGHTED POLLING" if use_weighted else "UNIFORM POLLING",
            "blocks": []
        }
        
        if not as_json:
            print(f"\n{i}. Timeslot: {hour_start.strftime('%Y-%m-%d %H:%M')} to {(hour_start + pd.Timedelta(hours=1)).strftime('%H:%M')}")
            print(f"   Overall Availability Score: {prob_1h:.2f}%")
        
        if use_weighted:
            if not as_json: print("   Strategy: WEIGHTED POLLING (Targeted burst)")
            # Extract the 4 15-minute blocks for this hour
            blocks = forecast_15m[(forecast_15m['ds'] >= hour_start) & (forecast_15m['ds'] <= hour_end)].copy()
            if len(blocks) != 4:
                if not as_json: print("   [Warning] Could not fetch 15m blocks. Defaulting to Uniform.")
                blocks_info = [(hour_start + pd.Timedelta(minutes=m), 25.0) for m in [0, 15, 30, 45]]
                slot_data["strategy"] = "UNIFORM POLLING (Fallback)"
            else:
                # Rank blocks by their 15-minute prediction score
                blocks = blocks.sort_values('yhat1', ascending=False)
                # Assign 80% of budget to top 2 (40% each), 20% to bottom 2 (10% each)
                blocks['budget'] = [40.0, 40.0, 10.0, 10.0]
                # Re-sort chronologically for printing
                blocks = blocks.sort_values('ds')
                blocks_info = zip(blocks['ds'], blocks['budget'])
        else:
            if not as_json: print("   Strategy: UNIFORM POLLING (Evenly distributed)")
            blocks_info = [(hour_start + pd.Timedelta(minutes=m), 25.0) for m in [0, 15, 30, 45]]
            
        for block_start, budget in blocks_info:
            block_end = block_start + pd.Timedelta(minutes=15)
            slot_data["blocks"].append({
                "block_start": block_start.strftime('%Y-%m-%d %H:%M:%S'),
                "block_end": block_end.strftime('%Y-%m-%d %H:%M:%S'),
                "budget_percent": float(budget)
            })
            if not as_json:
                print(f"     -> {block_start.strftime('%H:%M')} - {block_end.strftime('%H:%M')}: Spend {budget:.0f}% of hourly API limit")
        
        json_output.append(slot_data)
        
    if as_json:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(json.dumps(json_output, indent=2))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate API polling strategy.')
    parser.add_argument('--gpu', type=str, required=True, help='GPU instance type')
    parser.add_argument('--start', type=str, default=None, help='Start time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--top', type=int, default=3, help='Top N timeslots')
    
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    generate_api_strategy(args.gpu, args.start, args.top, args.json)
