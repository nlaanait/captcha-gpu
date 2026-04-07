import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings

logging.getLogger("NP").setLevel(logging.ERROR)

def evaluate_gpu(gpu_type, df_full):
    gpu_data = df_full[df_full['type'] == gpu_type].copy()
    if gpu_data.empty:
        return None
        
    gpu_data = gpu_data[['timestamp', 'available']]
    gpu_data.columns = ['ds', 'y']
    gpu_data['y'] = gpu_data['y'].astype(float)
    
    gpu_data.set_index('ds', inplace=True)
    gpu_data = gpu_data.resample('1H').max().reset_index()
    gpu_data['y'] = gpu_data['y'].fillna(0)
    
    gpu_data = gpu_data.sort_values('ds').reset_index(drop=True)
    
    max_date = gpu_data['ds'].max()
    cutoff_date = max_date - pd.Timedelta(hours=24)
    
    train_df = gpu_data[gpu_data['ds'] < cutoff_date].copy()
    test_df = gpu_data[gpu_data['ds'] >= cutoff_date].copy()
    
    if len(train_df) < 48 or len(test_df) < 10: # Minimum data check
        return {'GPU': gpu_type, 'Error': 'Not enough data'}
    
    val_cutoff = cutoff_date - pd.Timedelta(hours=24)
    train_only_df = train_df[train_df['ds'] < val_cutoff].copy()
    val_df = train_df[train_df['ds'] >= val_cutoff].copy()
    
    m_val = NeuralProphet(epochs=50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_val.fit(train_only_df, freq='1H')
    
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
            
    m_test = NeuralProphet(epochs=50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_test.fit(train_df, freq='1H')
        
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
    
    return {
        'GPU': gpu_type,
        'Threshold': best_thresh,
        'Val F1': best_f1,
        'Test Acc': accuracy,
        'Test Prec': precision,
        'Test Rec': recall,
        'Test F1': f1,
        'True Positives (Test)': np.sum((y_pred_bin_test == 1) & (y_true_test == 1)),
        'True Test Avg': np.mean(y_true_test)
    }

def main():
    df_full = pd.read_csv('../lambda-cloud-api/availability_stats.csv')
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    gpu_types = df_full['type'].unique()
    
    print(f"Found {len(gpu_types)} unique GPU types.")
    
    results = []
    for gpu in gpu_types:
        print(f"Evaluating {gpu}...")
        res = evaluate_gpu(gpu, df_full)
        if res is not None:
            results.append(res)
            
    print("\n" + "="*80)
    print("GPU EVALUATION METRICS (1H BLOCKS)")
    print("="*80)
    
    # Filter out errors
    valid_results = [r for r in results if 'Error' not in r]
    
    if not valid_results:
        print("No valid results found.")
        return
        
    summary_df = pd.DataFrame(valid_results)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    f1_scores = summary_df['Test F1']
    print("\n--- SENSITIVITY ANALYSIS ---")
    print(f"Mean F1: {f1_scores.mean():.4f}")
    print(f"Std Dev F1: {f1_scores.std():.4f}")
    print(f"Min F1: {f1_scores.min():.4f}")
    print(f"Max F1: {f1_scores.max():.4f}")

if __name__ == "__main__":
    main()
