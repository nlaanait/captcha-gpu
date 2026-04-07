import pandas as pd
from neuralprophet import NeuralProphet, save
import os
import sys
import warnings
import logging

logging.getLogger("NP").setLevel(logging.ERROR)

def train_and_save_all():
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('../lambda-cloud-api/availability_stats.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    gpu_types = df['type'].unique()
    print(f"Training models for {len(gpu_types)} GPU types...")
    
    for gpu in gpu_types:
        print(f"Training model for {gpu}...")
        gpu_data = df[df['type'] == gpu].copy()
        gpu_data = gpu_data[['timestamp', 'available']]
        gpu_data.columns = ['ds', 'y']
        gpu_data['y'] = gpu_data['y'].astype(float)
        
        # Resample to 1H blocks
        gpu_data.set_index('ds', inplace=True)
        gpu_data = gpu_data.resample('1H').max().reset_index()
        gpu_data['y'] = gpu_data['y'].fillna(0)
        gpu_data = gpu_data.sort_values('ds').reset_index(drop=True)
        
        m = NeuralProphet(epochs=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(gpu_data, freq='1H')
            
        save(m, f"models/{gpu}.np")
        print(f"Saved models/{gpu}.np")

if __name__ == '__main__':
    train_and_save_all()
