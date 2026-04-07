import pandas as pd
from neuralprophet import NeuralProphet
import argparse
import sys
import logging
from datetime import datetime
import numpy as np

# Suppress excessive logging from NeuralProphet
logging.getLogger("NP").setLevel(logging.ERROR)

def perform_backtesting(df, param_grid):
    # Sort by time
    df = df.sort_values('ds').reset_index(drop=True)
    n = len(df)
    
    # We will do a simple rolling window backtest
    splits = 3
    window_size = int(n * 0.2)
    train_size = int(n * 0.4)
    
    best_params = None
    best_error = float('inf')
    
    from itertools import product
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"Starting grid search over {len(experiments)} combinations...")
    
    for params in experiments:
        errors = []
        for i in range(splits):
            start_train = i * window_size
            end_train = train_size + i * window_size
            end_val = end_train + window_size
            
            if end_val > n:
                end_val = n
                
            train_df = df.iloc[start_train:end_train]
            val_df = df.iloc[end_train:end_val]
            
            if len(val_df) == 0:
                continue
                
            m = NeuralProphet(**params)
            
            # Fit
            m.fit(train_df, freq='min')
            
            future = m.make_future_dataframe(train_df, periods=len(val_df))
            forecast = m.predict(future)
            
            y_pred = forecast['yhat1'].values
            y_true = val_df['y'].values
            
            mse = np.mean((y_true - y_pred)**2)
            errors.append(mse)
            
        avg_error = np.mean(errors) if errors else float('inf')
        if avg_error < best_error:
            best_error = avg_error
            best_params = params
            
    return best_params, best_error

def find_next_available(df, best_params, target_time):
    # Train model on all data
    m = NeuralProphet(**best_params)
    m.fit(df, freq='min')
    
    # Forecast future
    # Predict the next 24 hours (1440 minutes) from the last data point
    future = m.make_future_dataframe(df, periods=1440) 
    forecast = m.predict(future)
    
    # Only consider predictions AFTER the target_time
    target_time = pd.to_datetime(target_time)
    future_forecast = forecast[forecast['ds'] >= target_time]
    
    if future_forecast.empty:
        # If target_time is beyond the forecast horizon, let's forecast more
        last_time = df['ds'].max()
        mins_diff = int((target_time - last_time).total_seconds() / 60)
        periods = max(1440, mins_diff + 1440)
        future = m.make_future_dataframe(df, periods=periods)
        forecast = m.predict(future)
        future_forecast = forecast[forecast['ds'] >= target_time]
        
    if future_forecast.empty:
        return None
    
    # The most likely timeslot is the one with the maximum predicted availability (yhat1)
    best_row = future_forecast.loc[future_forecast['yhat1'].idxmax()]
    
    return best_row

def main():
    parser = argparse.ArgumentParser(description='Predict GPU availability using backtesting.')
    parser.add_argument('--gpu', type=str, required=True, help='GPU instance type')
    parser.add_argument('--time', type=str, required=True, help='Target time (YYYY-MM-DD HH:MM:SS) - representing "right now"')
    
    args = parser.parse_args()
    
    # Load the data
    df = pd.read_csv('../lambda-cloud-api/availability_stats.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    gpu_data = df[df['type'] == args.gpu].copy()
    if gpu_data.empty:
        print(f"No data found for GPU type: {args.gpu}")
        sys.exit(1)
        
    gpu_data = gpu_data[['timestamp', 'available']]
    gpu_data.columns = ['ds', 'y']
    gpu_data['y'] = gpu_data['y'].astype(int)
    
    # Parameter grid for hyperparameter tuning
    param_grid = {
        'n_changepoints': [5, 10],
        'epochs': [20, 50],
        'batch_size': [32]
    }
    
    print("Performing backtesting for hyperparameter tuning...")
    best_params, best_error = perform_backtesting(gpu_data, param_grid)
    print(f"Best parameters found: {best_params} with MSE: {best_error:.4f}")
    
    print("Training final model and searching for the next most likely available timeslot...")
    best_row = find_next_available(gpu_data, best_params, args.time)
    
    if best_row is not None:
        print(f"--- RESULTS ---")
        print(f"GPU: {args.gpu}")
        print(f"Requested start time (right now): {args.time}")
        print(f"Most likely next available timeslot: {best_row['ds']}")
        print(f"Predicted availability score: {best_row['yhat1']:.4f}")
    else:
        print("Could not find a future timeslot within the forecast horizon.")

if __name__ == "__main__":
    main()
