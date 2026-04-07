import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import argparse
import sys
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

logging.getLogger("NP").setLevel(logging.ERROR)

def evaluate_24h(gpu_type):
    df = pd.read_csv('../lambda-cloud-api/availability_stats.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    gpu_data = df[df['type'] == gpu_type].copy()
    if gpu_data.empty:
        print(f"No data found for GPU type: {gpu_type}")
        sys.exit(1)
        
    gpu_data = gpu_data[['timestamp', 'available']]
    gpu_data.columns = ['ds', 'y']
    gpu_data['y'] = gpu_data['y'].astype(float)
    
    # IMPROVEMENT: Resample the minutely irregular data to an hourly grid
    # Taking the max means if the GPU was available at ANY minute in that hour, 
    # it's considered available for that hour. This massively reduces noise.
    gpu_data.set_index('ds', inplace=True)
    gpu_data = gpu_data.resample('H').max().reset_index()
    gpu_data['y'] = gpu_data['y'].fillna(0)
    
    gpu_data = gpu_data.sort_values('ds').reset_index(drop=True)
    
    max_date = gpu_data['ds'].max()
    cutoff_date = max_date - pd.Timedelta(hours=24)
    
    train_df = gpu_data[gpu_data['ds'] < cutoff_date].copy()
    test_df = gpu_data[gpu_data['ds'] >= cutoff_date].copy()
    
    # Validation set to tune the threshold (the 24 hours prior to the test set)
    val_cutoff = cutoff_date - pd.Timedelta(hours=24)
    train_only_df = train_df[train_df['ds'] < val_cutoff].copy()
    val_df = train_df[train_df['ds'] >= val_cutoff].copy()
    
    print(f"Training data from {train_df['ds'].min()} to {train_df['ds'].max()} ({len(train_df)} hourly rows)")
    print(f"Test data from {test_df['ds'].min()} to {test_df['ds'].max()} ({len(test_df)} hourly rows)")
    
    m = NeuralProphet(epochs=50)
    m.fit(train_only_df, freq='H')
    
    future_val = pd.DataFrame({'ds': val_df['ds'], 'y': None})
    forecast_val = m.predict(future_val)
    
    merged_val = pd.merge(val_df, forecast_val, on='ds', how='inner')
    y_pred = merged_val['yhat1'].values
    y_true = merged_val['y_x'].values
    
    best_f1 = -1
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred_bin = (y_pred >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Optimal Threshold on Validation Set: {best_thresh:.2f} (F1: {best_f1:.4f})")
    
    print("Training final model on full training set and testing on final 24h test set...")
    m = NeuralProphet(epochs=50)
    m.fit(train_df, freq='H')
    
    future_test = pd.DataFrame({'ds': test_df['ds'], 'y': None})
    forecast_test = m.predict(future_test)
        
    merged_test = pd.merge(test_df, forecast_test, on='ds', how='inner')
    y_pred = merged_test['yhat1'].values
    y_true = merged_test['y_x'].values
    
    y_pred_bin = (y_pred >= best_thresh).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin, zero_division=0)
    recall = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    
    print("\n--- FINAL 24-HOUR FORECAST EVALUATION RESULTS (HOURLY RESAMPLED) ---")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate 24h forecasting accuracy.')
    parser.add_argument('--gpu', type=str, required=True, help='GPU instance type')
    args = parser.parse_args()
    
    evaluate_24h(args.gpu)
