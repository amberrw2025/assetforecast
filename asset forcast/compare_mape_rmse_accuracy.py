#!/usr/bin/env python3
"""
MAPE vs RMSE Accuracy Comparison
===============================

Compares which metric (MAPE or RMSE) is actually following 
the actual 2024 prices more closely across multiple stocks.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def calculate_both_metrics(actual, predicted):
    """Calculate both MAPE and RMSE for comparison"""
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return None, None
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((actual_clean - predicted_clean)**2))
    
    return mape, rmse

def simple_forecast(prices, days=126):
    """Simple linear trend forecast like our plots use"""
    if len(prices) >= 30:
        trend = (prices[-1] - prices[-30]) / 30
    else:
        trend = (prices[-1] - prices[0]) / len(prices)
    
    forecast = []
    for i in range(1, days + 1):
        predicted_price = prices[-1] + (trend * i)
        forecast.append(max(0.01, predicted_price))
    
    return np.array(forecast)

def main():
    print('🔍 COMPARING MAPE vs RMSE ACCURACY FOR 2024 PREDICTIONS')
    print('=' * 70)

    # Test on key stocks from your comprehensive set
    test_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AZN.L', 'BP.L']

    results = []

    for ticker in test_tickers:
        print(f'\n📊 Testing {ticker}...')
        try:
            # Get training data (2020-2024)
            train_data = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)
            # Get actual 2024 data
            actual_2024 = yf.download(ticker, start='2024-01-01', end='2024-06-30', progress=False)
            
            if train_data.empty or actual_2024.empty:
                print(f'   ❌ No data for {ticker}')
                continue
                
            # Handle new yfinance format
            if isinstance(train_data['Close'], pd.DataFrame):
                prices = train_data['Close'].iloc[:, 0].values
                actual_prices = actual_2024['Close'].iloc[:, 0].values
            else:
                prices = train_data['Close'].values
                actual_prices = actual_2024['Close'].values
            
            # Generate forecast
            forecast = simple_forecast(prices, len(actual_prices))
            
            # Calculate both metrics
            mape, rmse = calculate_both_metrics(actual_prices, forecast)
            
            if mape is not None and rmse is not None:
                # Determine which is better (lower is better for both)
                # For comparison, convert RMSE to percentage of average price
                avg_price = np.mean(actual_prices)
                rmse_percentage = (rmse / avg_price) * 100
                
                better_metric = 'MAPE' if mape < rmse_percentage else 'RMSE'
                
                print(f'   📈 Avg Price: ${avg_price:.2f}')
                print(f'   📉 MAPE: {mape:.2f}%')
                print(f'   📉 RMSE: ${rmse:.2f} ({rmse_percentage:.2f}%)')
                print(f'   🏆 Better predictor: {better_metric}')
                
                results.append({
                    'ticker': ticker,
                    'mape': mape,
                    'rmse': rmse,
                    'rmse_pct': rmse_percentage,
                    'avg_price': avg_price,
                    'better': better_metric
                })
            else:
                print(f'   ❌ Could not calculate metrics for {ticker}')
                
        except Exception as e:
            print(f'   ❌ Error with {ticker}: {e}')

    print('\n' + '=' * 70)
    print('📊 SUMMARY: MAPE vs RMSE ACCURACY COMPARISON')
    print('=' * 70)

    if results:
        mape_wins = sum(1 for r in results if r['better'] == 'MAPE')
        rmse_wins = sum(1 for r in results if r['better'] == 'RMSE')
        
        avg_mape = np.mean([r['mape'] for r in results])
        avg_rmse_pct = np.mean([r['rmse_pct'] for r in results])
        
        print(f'📈 Average MAPE across stocks: {avg_mape:.2f}%')
        print(f'📈 Average RMSE (as %): {avg_rmse_pct:.2f}%')
        print()
        print(f'🏆 MAPE was better: {mape_wins}/{len(results)} stocks')
        print(f'🏆 RMSE was better: {rmse_wins}/{len(results)} stocks')
        print()
        
        if avg_mape < avg_rmse_pct:
            print('🎯 CONCLUSION: MAPE is following actual 2024 prices more closely!')
            print(f'   MAPE shows {avg_mape:.2f}% average error vs RMSE showing {avg_rmse_pct:.2f}%')
        else:
            print('🎯 CONCLUSION: RMSE is following actual 2024 prices more closely!')
            print(f'   RMSE shows {avg_rmse_pct:.2f}% average error vs MAPE showing {avg_mape:.2f}%')
            
        print('\n📋 DETAILED BREAKDOWN:')
        for r in results:
            print(f'   {r["ticker"]:8s}: MAPE={r["mape"]:5.1f}% | RMSE={r["rmse_pct"]:5.1f}% | Winner: {r["better"]}')
            
        print('\n💡 INTERPRETATION:')
        print('   • Lower percentage = better accuracy')
        print('   • MAPE = Mean Absolute Percentage Error')
        print('   • RMSE = Root Mean Square Error (converted to %)')
        print('   • Both measure forecast vs actual 2024 price accuracy')
        
    else:
        print('❌ No successful comparisons completed')

if __name__ == "__main__":
    main()
