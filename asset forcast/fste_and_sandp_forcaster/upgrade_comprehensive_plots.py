#!/usr/bin/env python3
"""
Upgrade Comprehensive Plots with ML Forecasting
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Create ML features"""
    df = df.copy()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['roc_20'] = df['Close'].pct_change(20)
    df['volatility'] = df['Close'].rolling(20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def train_model(df):
    """Train ML model"""
    df_features = create_features(df)
    feature_cols = ['sma_20', 'roc_20', 'volatility', 'rsi']
    df_features['target'] = df_features['Close'].shift(-1)
    df_clean = df_features.dropna()
    
    if len(df_clean) < 50:
        return None
    
    X = df_clean[feature_cols].values
    y = df_clean['target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    return {'model': model, 'scaler': scaler, 'features': feature_cols}

def generate_forecast(ticker):
    """Generate forecast for ticker"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start='2020-01-01', end='2024-12-31')
        
        train = data[data.index < '2024-01-01']
        actual = data[data.index >= '2024-01-01']
        
        model_info = train_model(train)
        if not model_info:
            return None
        
        # Simple prediction
        df_feat = create_features(train)
        last_feat = df_feat[model_info['features']].tail(1).values
        last_scaled = model_info['scaler'].transform(last_feat)
        
        predictions = []
        for i in range(min(100, len(actual))):
            pred = model_info['model'].predict(last_scaled)[0]
            predictions.append(pred)
        
        return {'train': train, 'actual': actual, 'predictions': predictions}
    except:
        return None

def create_plot(ticker, data):
    """Create improved plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    recent = data['train'].tail(120)
    ax.plot(recent.index, recent['Close'], 'blue', label='Historical', linewidth=2)
    
    if len(data['actual']) > 0:
        ax.plot(data['actual'].index, data['actual']['Close'], 'green', 
               label='2024 Actual', linewidth=2)
    
    if data['predictions']:
        dates = data['actual'].index[:len(data['predictions'])]
        ax.plot(dates, data['predictions'], 'red', 
               label='ML Forecast', linewidth=2, linestyle='--')
    
    ax.set_title(f'{ticker} - ML Enhanced Forecast 2024', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to both directories
    Path('comprehensive_plots_2024/option1_standard').mkdir(parents=True, exist_ok=True)
    Path('comprehensive_plots_2024/option2_mape_enhanced').mkdir(parents=True, exist_ok=True)
    
    plt.savefig(f'comprehensive_plots_2024/option1_standard/{ticker}_2024_vs_actual.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'comprehensive_plots_2024/option2_mape_enhanced/{ticker}_unified_mape_forecast_2024.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main upgrade function"""
    try:
        with open('primary_stocks.txt') as f:
            primary = [line.strip() for line in f if line.strip()]
        with open('fallback_stocks.txt') as f:
            fallback = [line.strip() for line in f if line.strip()]
        stocks = list(set(primary + fallback))
    except:
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    
    print(f"🚀 Upgrading {len(stocks)} plots with ML forecasting...")
    
    for ticker in stocks:
        print(f"Processing {ticker}...", end=' ')
        data = generate_forecast(ticker)
        if data:
            create_plot(ticker, data)
            print("✅")
        else:
            print("❌")
    
    print("🎯 Upgrade complete!")

if __name__ == "__main__":
    main()
