import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
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
    gpu_data['y'] = gpu_data['y'].astype(int)
    
    gpu_data = gpu_data.sort_values('ds').reset_index(drop=True)
    max_date = gpu_data['ds'].max()
    cutoff_date = max_date - pd.Timedelta(hours=24)
    
    train_df = gpu_data[gpu_data['ds'] < cutoff_date].copy()
    test_df = gpu_data[gpu_data['ds'] >= cutoff_date].copy()
    
    # Validation set to tune the threshold (the 24 hours prior to the test set)
    val_cutoff = cutoff_date - pd.Timedelta(hours=24)
    train_only_df = train_df[train_df['ds'] < val_cutoff].copy()
    val_df = train_df[train_df['ds'] >= val_cutoff].copy()
    
    configs = [
        {'name': 'Baseline (defaults)', 'params': {'epochs': 20, 'batch_size': 32, 'drop_missing': True}},
        {'name': 'High Capacity & Changepoints', 'params': {'n_changepoints': 20, 'epochs': 50, 'learning_rate': 0.01, 'batch_size': 32, 'drop_missing': True}},
        {'name': 'Explicit Seasonality', 'params': {'daily_seasonality': True, 'weekly_seasonality': True, 'epochs': 30, 'batch_size': 32, 'drop_missing': True}},
    ]
    
    best_f1_overall = -1
    best_config_overall = None
    best_thresh_overall = 0.5
    
    print("Evaluating configurations on Validation Set (tuning threshold)...")
    for config in configs:
        m = NeuralProphet(**config['params'])
        m.fit(train_only_df, freq='min')
        
        future = pd.DataFrame({'ds': val_df['ds'], 'y': None})
        forecast = m.predict(future)
        forecast_val = forecast
        
        # Ensure alignment
        merged = pd.merge(val_df, forecast_val, on='ds', how='inner')
        y_pred = merged['yhat1'].values
        y_true = merged['y_x'].values
        
        best_f1 = -1
        best_thresh = 0.5
        for thresh in np.arange(0.05, 0.95, 0.05):
            y_pred_bin = (y_pred >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        print(f"Config '{config['name']}': Best Val F1 = {best_f1:.4f} at threshold = {best_thresh:.2f}")
        if best_f1 > best_f1_overall:
            best_f1_overall = best_f1
            best_config_overall = config
            best_thresh_overall = best_thresh
            
    print(f"\nWinner: '{best_config_overall['name']}' with Threshold {best_thresh_overall:.2f} (Val F1: {best_f1_overall:.4f})")
    
    print("\nTraining winning config on full training set and testing on final 24h test set...")
    m = NeuralProphet(**best_config_overall['params'])
    m.fit(train_df, freq='min')
    
    future = pd.DataFrame({'ds': test_df['ds'], 'y': None})
    forecast = m.predict(future)
    forecast_test = forecast
        
    merged_test = pd.merge(test_df, forecast_test, on='ds', how='inner')
    y_pred = merged_test['yhat1'].values
    y_true = merged_test['y_x'].values
    
    for thresh, name in [(0.5, 'Default 0.5 Threshold'), (best_thresh_overall, f'Tuned {best_thresh_overall:.2f} Threshold')]:
        y_pred_bin = (y_pred >= thresh).astype(int)
        accuracy = accuracy_score(y_true, y_pred_bin)
        precision = precision_score(y_true, y_pred_bin, zero_division=0)
        recall = recall_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        
        print(f"\n--- TEST SET RESULTS ({name}) ---")
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1 Score:  {f1*100:.2f}%")

if __name__ == '__main__':
    evaluate_24h('gpu_1x_a10')