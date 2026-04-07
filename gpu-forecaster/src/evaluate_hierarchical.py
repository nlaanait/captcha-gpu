import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import sys
import os
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings

logging.getLogger("NP").setLevel(logging.ERROR)

def evaluate_hierarchical(gpu_type, df_full):
    gpu_data = df_full[df_full['type'] == gpu_type].copy()
    if gpu_data.empty:
        return None
        
    gpu_data = gpu_data[['timestamp', 'available']]
    gpu_data.columns = ['ds', 'y']
    gpu_data['y'] = gpu_data['y'].astype(float)
    gpu_data.set_index('ds', inplace=True)
    
    # Create 1H dataset
    df_1h = gpu_data.resample('1H').max().reset_index()
    df_1h['y'] = df_1h['y'].fillna(0)
    df_1h = df_1h.sort_values('ds').reset_index(drop=True)
    
    # Create 15m dataset
    df_15m = gpu_data.resample('15min').max().reset_index()
    df_15m['y'] = df_15m['y'].fillna(0)
    df_15m = df_15m.sort_values('ds').reset_index(drop=True)
    
    max_date = df_1h['ds'].max()
    cutoff_date = max_date - pd.Timedelta(hours=24)
    
    train_1h = df_1h[df_1h['ds'] < cutoff_date].copy()
    test_1h = df_1h[df_1h['ds'] >= cutoff_date].copy()
    
    train_15m = df_15m[df_15m['ds'] < cutoff_date].copy()
    test_15m = df_15m[df_15m['ds'] >= cutoff_date].copy()
    
    if len(train_1h) < 48 or len(test_1h) < 10:
        return {'GPU': gpu_type, 'Error': 'Not enough data'}
        
    print(f"Training 1H and 15min models for {gpu_type}...")
    
    m_1h = NeuralProphet(epochs=50)
    m_15m = NeuralProphet(epochs=50)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_1h.fit(train_1h, freq='1H')
        m_15m.fit(train_15m, freq='15min')
        
    # Forecast 24h using 1H model
    future_1h = pd.DataFrame({'ds': test_1h['ds'], 'y': None})
    forecast_1h = m_1h.predict(future_1h)
    merged_1h = pd.merge(test_1h, forecast_1h, on='ds', how='inner')
    
    # We will pick the top 5 hours based on 1H yhat1 score to simulate a polling strategy
    top_hours = merged_1h.sort_values('yhat1', ascending=False).head(5)
    
    # Forecast 24h using 15m model
    future_15m = pd.DataFrame({'ds': test_15m['ds'], 'y': None})
    forecast_15m = m_15m.predict(future_15m)
    merged_15m = pd.merge(test_15m, forecast_15m, on='ds', how='inner')
    
    # Evaluate 15m ranking within the top predicted hours
    # For each selected hour, we look at the 4 15-minute blocks
    # We evaluate if the block with the highest predicted 15m score actually had availability
    
    correct_top1 = 0
    total_eval_hours = 0
    
    # Also track NDCG-like metric or Top-2 recall (did the actual availability fall in top 2 weighted blocks?)
    correct_top2 = 0
    
    for _, row in top_hours.iterrows():
        hour_start = row['ds']
        hour_end = hour_start + pd.Timedelta(minutes=45) # The 4th block starts at +45m
        
        # True availability in this hour (1H level)
        if row['y_x'] == 0:
            continue # If it wasn't actually available in this hour, we skip evaluating intra-hour ranking accuracy
            
        # Get the 4 15-minute blocks for this hour
        blocks = merged_15m[(merged_15m['ds'] >= hour_start) & (merged_15m['ds'] <= hour_end)]
        
        if len(blocks) != 4:
            continue
            
        total_eval_hours += 1
        
        # Sort blocks by predicted 15m score
        ranked_blocks = blocks.sort_values('yhat1', ascending=False)
        
        # True available blocks
        true_available = ranked_blocks[ranked_blocks['y_x'] > 0]
        
        if true_available.empty:
            continue # Should not happen if 1H was true, but just in case of mismatch
            
        # Did the #1 ranked block contain availability?
        top1_block = ranked_blocks.iloc[0]
        if top1_block['y_x'] > 0:
            correct_top1 += 1
            
        # Did the Top 2 ranked blocks contain availability?
        top2_blocks = ranked_blocks.iloc[:2]
        if top2_blocks['y_x'].sum() > 0:
            correct_top2 += 1

    top1_acc = (correct_top1 / total_eval_hours) if total_eval_hours > 0 else 0
    top2_acc = (correct_top2 / total_eval_hours) if total_eval_hours > 0 else 0

    return {
        'GPU': gpu_type,
        'Top 5 Hours Selected': len(top_hours),
        'Hours Actually Available': total_eval_hours,
        'Top1 15m Block Acc': top1_acc,
        'Top2 15m Blocks Acc': top2_acc
    }

def main():
    df_full = pd.read_csv('../lambda-cloud-api/availability_stats.csv')
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    gpu_types = df_full['type'].unique()
    
    results = []
    for gpu in gpu_types:
        res = evaluate_hierarchical(gpu, df_full)
        if res is not None and 'Error' not in res:
            results.append(res)
            
    print("\n" + "="*80)
    print("HIERARCHICAL POLLING EVALUATION (1H -> 15m weights)")
    print("="*80)
    
    if not results:
        print("No valid results found.")
        return
        
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x*100:.2f}%" if isinstance(x, float) else str(x)))
    
    print("\nInterpretation:")
    print("- 'Hours Actually Available': Out of the Top 5 hours predicted by 1H model, how many were actually true positives.")
    print("- 'Top1 15m Block Acc': When the hour was a true positive, how often did the #1 weighted 15m block contain the availability.")
    print("- 'Top2 15m Blocks Acc': How often did the availability fall within the Top 2 weighted 15m blocks (representing 50% of the polling budget).")

if __name__ == "__main__":
    main()
