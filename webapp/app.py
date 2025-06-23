"""
Flask web application for asset forecasting.
Multi-model stock prediction interface.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def create_app():
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/forecast-asset', methods=['POST'])
    def forecast_asset():
        try:
            data = request.get_json()
            ticker = data.get('ticker', '').upper().strip()
            days = int(data.get('days', 30))
            
            if not ticker:
                return jsonify({'success': False, 'error': 'Ticker required'})
            
            # Get stock data
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period='1y')
            
            if hist_data.empty:
                return jsonify({'success': False, 'error': f'No data for {ticker}'})
            
            # Prepare data
            dates = [d.strftime('%Y-%m-%d') for d in hist_data.index]
            prices = hist_data['Close'].tolist()
            
            # Store all forecasts and model info
            all_forecasts = {}
            model_info = {}
            
            # 1. Linear Regression Model
            try:
                recent = hist_data.tail(60)
                X = np.arange(len(recent)).reshape(-1, 1)
                y = recent['Close'].values
                
                linear_model = LinearRegression()
                linear_model.fit(X, y)
                
                future_X = np.arange(len(recent), len(recent) + days).reshape(-1, 1)
                linear_forecast = linear_model.predict(future_X)
                linear_forecast = np.maximum(linear_forecast, 0.01)
                
                all_forecasts['Linear Regression'] = linear_forecast.tolist()
                model_info['Linear Regression'] = {
                    'status': 'success',
                    'type': f'Linear trend on {len(recent)} days',
                    'slope': float(linear_model.coef_[0])
                }
            except Exception as e:
                model_info['Linear Regression'] = {'status': 'failed', 'error': str(e)[:50]}
            
            # 2. Moving Average Model
            try:
                prices_array = hist_data['Close'].values
                short_ma = np.mean(prices_array[-7:])  # 7-day MA
                long_ma = np.mean(prices_array[-30:])   # 30-day MA
                
                if long_ma > 0:
                    trend = (short_ma - long_ma) / long_ma
                else:
                    trend = 0
                
                last_price = prices_array[-1]
                ma_forecast = []
                
                for i in range(days):
                    trend_factor = trend * (0.95 ** i)  # Diminishing trend
                    next_price = last_price * (1 + trend_factor)
                    next_price = max(next_price, 0.01)
                    ma_forecast.append(next_price)
                    last_price = next_price
                
                all_forecasts['Moving Average'] = ma_forecast
                model_info['Moving Average'] = {
                    'status': 'success',
                    'type': '7-day vs 30-day MA trend',
                    'trend': f'{trend:.4f}'
                }
            except Exception as e:
                model_info['Moving Average'] = {'status': 'failed', 'error': str(e)[:50]}
            
            # 3. Exponential Smoothing Model
            try:
                prices_array = hist_data['Close'].values
                alpha = 0.3  # Smoothing parameter
                smoothed = [prices_array[0]]
                
                for price in prices_array[1:]:
                    smoothed.append(alpha * price + (1 - alpha) * smoothed[-1])
                
                # Calculate trend from smoothed data
                recent_smoothed = smoothed[-10:]
                if len(recent_smoothed) >= 2:
                    trend_per_day = (recent_smoothed[-1] - recent_smoothed[0]) / (len(recent_smoothed) - 1)
                else:
                    trend_per_day = 0
                
                last_smoothed = smoothed[-1]
                exp_forecast = []
                
                for i in range(days):
                    trend_effect = trend_per_day * (i + 1) * (0.98 ** i)
                    next_value = last_smoothed + trend_effect
                    next_value = max(next_value, 0.01)
                    exp_forecast.append(next_value)
                
                all_forecasts['Exponential Smoothing'] = exp_forecast
                model_info['Exponential Smoothing'] = {
                    'status': 'success',
                    'type': f'Exponential smoothing (α={alpha})',
                    'trend_per_day': f'{trend_per_day:.4f}'
                }
            except Exception as e:
                model_info['Exponential Smoothing'] = {'status': 'failed', 'error': str(e)[:50]}
            
            # 4. Polynomial Regression Model
            try:
                recent = hist_data.tail(50)
                if len(recent) >= 10:
                    X = np.arange(len(recent)).reshape(-1, 1)
                    y = recent['Close'].values
                    
                    # 2nd degree polynomial
                    poly_features = PolynomialFeatures(degree=2)
                    X_poly = poly_features.fit_transform(X)
                    
                    poly_model = LinearRegression()
                    poly_model.fit(X_poly, y)
                    
                    future_X = np.arange(len(recent), len(recent) + days).reshape(-1, 1)
                    future_X_poly = poly_features.transform(future_X)
                    poly_forecast = poly_model.predict(future_X_poly)
                    
                    # Constrain predictions to reasonable bounds
                    current_price = y[-1]
                    poly_forecast = np.clip(poly_forecast, current_price * 0.5, current_price * 2.0)
                    poly_forecast = np.maximum(poly_forecast, 0.01)
                    
                    all_forecasts['Polynomial Regression'] = poly_forecast.tolist()
                    model_info['Polynomial Regression'] = {
                        'status': 'success',
                        'type': f'2nd degree polynomial on {len(recent)} days',
                        'r2_score': 'fitted'
                    }
                else:
                    model_info['Polynomial Regression'] = {'status': 'failed', 'error': 'Insufficient data'}
            except Exception as e:
                model_info['Polynomial Regression'] = {'status': 'failed', 'error': str(e)[:50]}
            
            # 5. Seasonal Trend Model
            try:
                prices_array = hist_data['Close'].values
                
                if len(prices_array) >= 30:
                    # Calculate weekly seasonality (7-day pattern)
                    weekly_pattern = []
                    for day in range(7):
                        day_prices = [prices_array[i] for i in range(day, len(prices_array), 7)]
                        if day_prices:
                            weekly_pattern.append(np.mean(day_prices))
                        else:
                            weekly_pattern.append(prices_array[-1])
                    
                    # Overall trend
                    trend_slope = (prices_array[-1] - prices_array[-30]) / 30
                    
                    seasonal_forecast = []
                    last_price = prices_array[-1]
                    
                    for i in range(days):
                        # Apply trend
                        trend_component = last_price + trend_slope * (i + 1)
                        
                        # Apply seasonality
                        seasonal_factor = weekly_pattern[i % 7] / np.mean(weekly_pattern)
                        
                        forecast_value = trend_component * seasonal_factor
                        forecast_value = max(forecast_value, 0.01)
                        seasonal_forecast.append(forecast_value)
                    
                    all_forecasts['Seasonal Model'] = seasonal_forecast
                    model_info['Seasonal Model'] = {
                        'status': 'success',
                        'type': 'Trend + 7-day seasonality',
                        'trend_slope': f'{trend_slope:.4f}'
                    }
                else:
                    model_info['Seasonal Model'] = {'status': 'failed', 'error': 'Need 30+ days for seasonality'}
            except Exception as e:
                model_info['Seasonal Model'] = {'status': 'failed', 'error': str(e)[:50]}
            
            # 6. Create Ensemble Model
            successful_forecasts = {name: forecast for name, forecast in all_forecasts.items()}
            
            if len(successful_forecasts) >= 2:
                try:
                    # Weighted ensemble - give more weight to certain models
                    weights = {
                        'Linear Regression': 0.20,
                        'Moving Average': 0.25,
                        'Exponential Smoothing': 0.25,
                        'Polynomial Regression': 0.15,
                        'Seasonal Model': 0.15
                    }
                    
                    ensemble_forecast = np.zeros(days)
                    total_weight = 0
                    
                    for name, forecast in successful_forecasts.items():
                        weight = weights.get(name, 0.2)  # Default weight
                        ensemble_forecast += np.array(forecast) * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        ensemble_forecast /= total_weight
                        
                        all_forecasts['Ensemble'] = ensemble_forecast.tolist()
                        model_info['Ensemble'] = {
                            'status': 'success',
                            'type': f'Weighted average of {len(successful_forecasts)} models',
                            'models_combined': list(successful_forecasts.keys())
                        }
                except Exception as e:
                    model_info['Ensemble'] = {'status': 'failed', 'error': str(e)[:50]}
            
            # Generate future dates (skip weekends)
            future_dates = []
            current_date = hist_data.index[-1]
            for i in range(days):
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:  # Skip weekends
                    current_date += timedelta(days=1)
                future_dates.append(current_date.strftime('%Y-%m-%d'))
            
            # Choose primary forecast (prefer ensemble, then best available)
            if 'Ensemble' in all_forecasts:
                primary_forecast = all_forecasts['Ensemble']
                primary_model = 'Ensemble'
            elif 'Moving Average' in all_forecasts:
                primary_forecast = all_forecasts['Moving Average']
                primary_model = 'Moving Average'
            elif 'Exponential Smoothing' in all_forecasts:
                primary_forecast = all_forecasts['Exponential Smoothing']
                primary_model = 'Exponential Smoothing'
            elif successful_forecasts:
                primary_model = list(successful_forecasts.keys())[0]
                primary_forecast = list(successful_forecasts.values())[0]
            else:
                return jsonify({'success': False, 'error': 'All forecasting models failed'})
            
            # Calculate summary statistics
            current_price = float(prices[-1])
            predicted_price = float(primary_forecast[-1])
            change_percent = ((predicted_price - current_price) / current_price) * 100
            
            # Calculate confidence interval based on model agreement
            if len(successful_forecasts) > 1:
                final_prices = [forecast[-1] for forecast in successful_forecasts.values()]
                price_std = np.std(final_prices)
                confidence_interval = float(price_std * 1.96)  # 95% confidence
            else:
                # Use historical volatility for single model
                returns = hist_data['Close'].pct_change().dropna()
                volatility = returns.std()
                confidence_interval = float(volatility * np.sqrt(days) * current_price)
            
            return jsonify({
                'success': True,
                'ticker': ticker,
                'historical_dates': dates[-60:],
                'historical_prices': prices[-60:],
                'forecast_dates': future_dates,
                'all_forecasts': all_forecasts,
                'primary_forecast': primary_forecast,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'change_percent': change_percent,
                'confidence_interval': confidence_interval,
                'model_info': model_info,
                'summary': {
                    'primary_model': primary_model,
                    'models_used': len(successful_forecasts),
                    'trend': 'Bullish' if change_percent > 0 else 'Bearish'
                },
                'ensemble_info': {
                    'primary_model': primary_model,
                    'models_used': len(successful_forecasts)
                }
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)