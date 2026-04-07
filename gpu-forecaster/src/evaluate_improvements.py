import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import sys
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch

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
    
    # Feature Engineering (Exogenous features)
    gpu_data['hour'] = gpu_data['ds'].dt.hour
    gpu_data['minute'] = gpu_data['ds'].dt.minute
    gpu_data['dayofweek'] = gpu_data['ds'].dt.dayofweek
    
    gpu_data = gpu_data.sort_values('ds').reset_index(drop=True)
    max_date = gpu_data['ds'].max()
    cutoff_date = max_date - pd.Timedelta(hours=24)
    
    train_df = gpu_data[gpu_data['ds'] < cutoff_date].copy()
    test_df = gpu_data[gpu_data['ds'] >= cutoff_date].copy()
    
    val_cutoff = cutoff_date - pd.Timedelta(hours=24)
    train_only_df = train_df[train_df['ds'] < val_cutoff].copy()
    val_df = train_df[train_df['ds'] >= val_cutoff].copy()
    
    # Define experiments based on user suggestions
    configs = [
        {
            'name': '1. Baseline Regression (Tuned Threshold)',
            'params': {'epochs': 30, 'batch_size': 32},
            'use_exogenous': False,
            'is_bce': False
        },
        {
            'name': '2. Logistic Components (BCE Loss)',
            'params': {
                'epochs': 30, 'batch_size': 32, 
                'loss_func': torch.nn.BCEWithLogitsLoss,
                'normalize': 'off'  # Essential for raw BCE with targets 0/1
            },
            'use_exogenous': False,
            'is_bce': True
        },
        {
            'name': '3. Extensive Exogenous Features',
            'params': {'epochs': 30, 'batch_size': 32},
            'use_exogenous': True,
            'is_bce': False
        }
    ]
    
    best_overall_f1 = -1
    best_overall_config = None
    best_overall_thresh = None
    
    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        
        m = NeuralProphet(**config['params'])
        if config['use_exogenous']:
            m.add_future_regressor('hour')
            m.add_future_regressor('minute')
            m.add_future_regressor('dayofweek')
            df_fit = train_only_df.copy()
        else:
            df_fit = train_only_df[['ds', 'y']].copy()
            
        # Fit on train_only to find best threshold
        m.fit(df_fit, freq='min')
        
        future_val = pd.DataFrame({'ds': val_df['ds'], 'y': None})
        if config['use_exogenous']:
            future_val['hour'] = future_val['ds'].dt.hour
            future_val['minute'] = future_val['ds'].dt.minute
            future_val['dayofweek'] = future_val['ds'].dt.dayofweek
            
        forecast_val = m.predict(future_val)
        
        merged_val = pd.merge(val_df, forecast_val, on='ds', how='inner')
        y_pred = merged_val['yhat1'].values
        y_true = merged_val['y_x'].values
        
        if config['is_bce']:
            # Logits to probabilities
            y_pred = 1 / (1 + np.exp(-y_pred))
            
        best_val_f1 = -1
        best_thresh = 0.5
        for thresh in np.arange(0.05, 0.95, 0.05):
            y_pred_bin = (y_pred >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred_bin, zero_division=0)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_thresh = thresh
                
        print(f"Optimal Threshold on Val = {best_thresh:.2f} (F1: {best_val_f1:.4f})")
        
        if best_val_f1 > best_overall_f1:
            best_overall_f1 = best_val_f1
            best_overall_config = config
            best_overall_thresh = best_thresh
            
    print(f"\n*** Winner: {best_overall_config['name']} with threshold {best_overall_thresh:.2f} ***")
    print("Training winning configuration on full training set and testing on 24h future window...")
    
    m = NeuralProphet(**best_overall_config['params'])
    if best_overall_config['use_exogenous']:
        m.add_future_regressor('hour')
        m.add_future_regressor('minute')
        m.add_future_regressor('dayofweek')
        df_fit_final = train_df.copy()
    else:
        df_fit_final = train_df[['ds', 'y']].copy()
        
    m.fit(df_fit_final, freq='min')
    
    future_test = pd.DataFrame({'ds': test_df['ds'], 'y': None})
    if best_overall_config['use_exogenous']:
        future_test['hour'] = future_test['ds'].dt.hour
        future_test['minute'] = future_test['ds'].dt.minute
        future_test['dayofweek'] = future_test['ds'].dt.dayofweek
        
    forecast_test = m.predict(future_test)
    merged_test = pd.merge(test_df, forecast_test, on='ds', how='inner')
    
    y_pred = merged_test['yhat1'].values
    y_true = merged_test['y_x'].values
    
    if best_overall_config['is_bce']:
        y_pred = 1 / (1 + np.exp(-y_pred))
        
    # Evaluate with optimal threshold
    y_pred_bin = (y_pred >= best_overall_thresh).astype(int)
    
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    
    print("\n--- FINAL 24H TEST RESULTS ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")

if __name__ == '__main__':
    evaluate_24h('gpu_1x_a10')
