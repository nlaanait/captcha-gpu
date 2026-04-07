import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import argparse
import sys
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

logging.getLogger("NP").setLevel(logging.ERROR)

def run_experiments(gpu_type):
    df = pd.read_csv('../lambda-cloud-api/availability_stats.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    gpu_data_raw = df[df['type'] == gpu_type].copy()
    if gpu_data_raw.empty:
        print(f"No data found for GPU type: {gpu_type}")
        sys.exit(1)
        
    gpu_data_raw = gpu_data_raw[['timestamp', 'available']]
    gpu_data_raw.columns = ['ds', 'y']
    gpu_data_raw['y'] = gpu_data_raw['y'].astype(float)
    gpu_data_raw.set_index('ds', inplace=True)
    
    blocks = ['15min', '30min', '45min', '1H']
    results = []

    for block in blocks:
        print(f"\n{'='*50}")
        print(f"Running experiment for block size: {block}")
        print(f"{'='*50}")
        
        gpu_data = gpu_data_raw.resample(block).max().reset_index()
        gpu_data['y'] = gpu_data['y'].fillna(0)
        gpu_data = gpu_data.sort_values('ds').reset_index(drop=True)
        
        max_date = gpu_data['ds'].max()
        cutoff_date = max_date - pd.Timedelta(hours=24)
        
        train_df = gpu_data[gpu_data['ds'] < cutoff_date].copy()
        test_df = gpu_data[gpu_data['ds'] >= cutoff_date].copy()
        
        val_cutoff = cutoff_date - pd.Timedelta(hours=24)
        train_only_df = train_df[train_df['ds'] < val_cutoff].copy()
        val_df = train_df[train_df['ds'] >= val_cutoff].copy()
        
        print(f"Training data: {len(train_only_df)} rows, Validation data: {len(val_df)} rows, Test data: {len(test_df)} rows")
        
        # 1. Train on train_only_df and tune threshold on val_df
        m_val = NeuralProphet(epochs=50)
        # Suppress pytorch lightning warnings during fit
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_val.fit(train_only_df, freq=block)
        
        future_val = pd.DataFrame({'ds': val_df['ds'], 'y': None})
        forecast_val = m_val.predict(future_val)
        
        merged_val = pd.merge(val_df, forecast_val, on='ds', how='inner')
        y_pred_val = merged_val['yhat1'].values
        y_true_val = merged_val['y_x'].values
        
        best_f1 = -1
        best_thresh = 0.5
        for thresh in np.arange(0.05, 0.95, 0.05):
            y_pred_bin = (y_pred_val >= thresh).astype(int)
            f1 = f1_score(y_true_val, y_pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        print(f"Optimal Threshold on Validation Set: {best_thresh:.2f} (F1: {best_f1:.4f})")
        
        # 2. Train final model on train_df and evaluate on test_df
        print("Training final model on full training set...")
        m_test = NeuralProphet(epochs=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_test.fit(train_df, freq=block)
            
        future_test = pd.DataFrame({'ds': test_df['ds'], 'y': None})
        forecast_test = m_test.predict(future_test)
            
        merged_test = pd.merge(test_df, forecast_test, on='ds', how='inner')
        y_pred_test = merged_test['yhat1'].values
        y_true_test = merged_test['y_x'].values
        
        y_pred_bin_test = (y_pred_test >= best_thresh).astype(int)
        
        accuracy = accuracy_score(y_true_test, y_pred_bin_test)
        precision = precision_score(y_true_test, y_pred_bin_test, zero_division=0)
        recall = recall_score(y_true_test, y_pred_bin_test, zero_division=0)
        f1 = f1_score(y_true_test, y_pred_bin_test, zero_division=0)
        
        print(f"--- TEST SET RESULTS ({block}) ---")
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1 Score:  {f1*100:.2f}%")
        
        results.append({
            'Block Size': block,
            'Threshold': best_thresh,
            'Val F1': best_f1,
            'Test Accuracy': accuracy,
            'Test Precision': precision,
            'Test Recall': recall,
            'Test F1': f1
        })

    print("\n" + "="*70)
    print("FINAL SUMMARY ACROSS ALL BLOCK SIZES")
    print("="*70)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment with different resampling block sizes.')
    parser.add_argument('--gpu', type=str, required=True, help='GPU instance type')
    args = parser.parse_args()
    
    run_experiments(args.gpu)
