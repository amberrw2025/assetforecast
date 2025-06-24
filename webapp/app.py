"""
Flask web application for asset forecasting.
Multi-model stock prediction interface.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
import yfinance as yf
from dotenv import load_dotenv
from pathlib import Path
import traceback

# Load environment variables from .env file
load_dotenv()

# Import advanced models
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.arima_model import ARIMAModel
    from models.prophet_model import ProphetModel
    from models.lstm_model import LSTMModel
    ADVANCED_MODELS_AVAILABLE = True
    print("✅ Advanced models imported successfully")
except ImportError as e:
    print(f"⚠️  Advanced models not available: {e}")
    ADVANCED_MODELS_AVAILABLE = False

# --- Global Cache Initialization ---
cache = Cache(config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
})

@cache.memoize(timeout=300)
def get_yfinance_data(ticker, period="2y"):
    """Cached function to fetch data from yfinance."""
    yf_ticker = yf.Ticker(ticker)
    data = yf_ticker.history(period=period)
    if data.empty:
        return None
    # Ensure the DatetimeIndex is timezone-naive
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    return data

def create_app():
    app = Flask(__name__)
    cache.init_app(app)
    
    @app.route('/')
    def index():
        return render_template('commodities.html')

    @app.route('/commodities')
    def commodities():
        return render_template('commodities.html')
    
    @app.route('/api/commodity-data/<string:commodity>')
    def get_commodity_data(commodity):
        """Get commodity overview data."""
        try:
            # Fetch data for a longer period to calculate all metrics
            hist_data = get_yfinance_data(commodity, period="2y")
            if hist_data is None or hist_data.empty:
                return jsonify({'success': False, 'error': f'No data available for {commodity}'})

            close_prices = hist_data['Close']
            
            # Calculate metrics
            current_price = close_prices.iloc[-1]
            previous_price = close_prices.iloc[-2]
            change_percent = ((current_price - previous_price) / previous_price) * 100
            
            returns = close_prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100 # Annualized volatility

            # Date range
            date_range = {
                'start': hist_data.index.min().strftime('%Y-%m-%d'),
                'end': hist_data.index.max().strftime('%Y-%m-%d')
            }
            
            # 52-week high/low
            year_high = close_prices.last('365D').max()
            year_low = close_prices.last('365D').min()
            
            # Volatility over 30 and 90 days
            vol_30d = returns.last('30D').std() * np.sqrt(252) * 100
            vol_90d = returns.last('90D').std() * np.sqrt(252) * 100
            
            # Max Drawdown
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns/peak) - 1
            max_drawdown = drawdown.min() * 100

            # Sharpe Ratio (risk-free rate is assumed to be 0)
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

            return jsonify({
                'success': True,
                'data': {
                    'current_price': current_price,
                    'change_percent': change_percent,
                    'volatility': volatility,
                    'records': len(hist_data),
                    'date_range': date_range,
                    'year_high': year_high,
                    'year_low': year_low,
                    'avg_price': close_prices.mean(),
                    'vol_30d': vol_30d,
                    'vol_90d': vol_90d,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio
                }
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Error fetching commodity data: {str(e)}'})

    @app.route('/api/commodity-historical/<commodity>')
    def get_commodity_historical(commodity):
        """Get historical commodity price data for charting"""
        try:
            hist_data = get_yfinance_data(commodity, period="1y")
            if hist_data is None:
                return jsonify({'success': False, 'error': f'No historical data available for {commodity}'})
            
            return jsonify({
                'success': True,
                'data': {
                    'dates': hist_data.index.strftime('%Y-%m-%d').tolist(),
                    'prices': hist_data['Close'].tolist(),
                }
            })
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error fetching historical data: {str(e)}'})

    @app.route('/api/forecast-commodity', methods=['POST'])
    def forecast_commodity():
        """Generate commodity price forecast using an advanced model ensemble."""
        try:
            data = request.get_json()
            commodity = data.get('commodity', '').strip()
            days = int(data.get('days', 30))

            if not commodity:
                return jsonify({'success': False, 'error': 'Commodity symbol required'})

            hist_data = get_yfinance_data(commodity, period="3y")
            
            if hist_data is None or len(hist_data) < 100:
                return jsonify({'success': False, 'error': f'Insufficient historical data for {commodity}'})

            # --- Advanced Model Forecasting ---
            forecasts = {}
            model_info = []

            df_model = hist_data.reset_index().rename(columns={'Date': 'date', 'Close': 'close_price'})
            
            # 1. ARIMA Model
            if ADVANCED_MODELS_AVAILABLE:
                try:
                    arima_model = ARIMAModel(order=(5, 1, 0))
                    arima_model.fit(df_model)
                    forecasts['ARIMA'] = arima_model.predict(df_model, steps=days).tolist()
                    model_info.append('ARIMA')
                except Exception as e:
                    print(f"Commodity ARIMA failed: {e}")

                # 2. Prophet Model
                try:
                    prophet_model = ProphetModel(yearly_seasonality=True, weekly_seasonality=False)
                    prophet_model.fit(df_model)
                    forecasts['Prophet'] = prophet_model.predict(df_model, periods=days).tolist()
                    model_info.append('Prophet')
                except Exception as e:
                    print(f"Commodity Prophet failed: {e}")

                # 3. LSTM Model
                try:
                    lstm_model = LSTMModel(sequence_length=60, units=50, layers=2)
                    lstm_model.fit(df_model, epochs=50)
                    forecasts['LSTM'] = lstm_model.predict(df_model, steps=days).tolist()
                    model_info.append('LSTM')
                except Exception as e:
                    print(f"Commodity LSTM failed: {e}")

            if not forecasts:
                return jsonify({'success': False, 'error': 'All advanced forecasting models failed.'})

            # Create a weighted ensemble forecast
            ensemble_forecast = np.zeros(days)
            weights = {'ARIMA': 0.3, 'Prophet': 0.4, 'LSTM': 0.3}
            total_weight = 0

            for model_name, forecast_data in forecasts.items():
                if model_name in weights and len(forecast_data) == days:
                    weight = weights[model_name]
                    ensemble_forecast += np.array(forecast_data) * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_forecast /= total_weight
            
            current_price = hist_data['Close'].iloc[-1]
            predicted_price = ensemble_forecast[-1]
            price_change = ((predicted_price - current_price) / current_price) * 100
            
            trend = "Neutral"
            if price_change > 1: trend = "Bullish"
            if price_change > 3: trend = "Strong Bullish"
            if price_change < -1: trend = "Bearish"
            if price_change < -3: trend = "Strong Bearish"

            forecast_dates = [ (hist_data.index[-1] + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days + 1) ]

            valid_forecasts = [np.array(f) for f in forecasts.values() if len(f) == days]
            if len(valid_forecasts) > 1:
                forecast_matrix = np.array(valid_forecasts)
                std_dev = np.std(forecast_matrix, axis=0)
                upper_band = (ensemble_forecast + std_dev * 1.96).tolist()
                lower_band = (ensemble_forecast - std_dev * 1.96).tolist()
            else: 
                std_dev = np.std(ensemble_forecast) * 0.1
                upper_band = (ensemble_forecast + std_dev).tolist()
                lower_band = (ensemble_forecast - std_dev).tolist()
                
            return jsonify({
                'success': True,
                'forecast': {
                    'predicted_price': predicted_price,
                    'confidence': min(95, 70 + len(model_info) * 10),
                    'trend': trend,
                    'models_used': len(model_info),
                    'model_details': ', '.join(model_info),
                    'insights': [f"Forecast combines {', '.join(model_info)} models for robustness."],
                    'price_change_percent': price_change,
                },
                'historical_data': {
                    'dates': hist_data.index[-60:].strftime('%Y-%m-%d').tolist(),
                    'prices': hist_data['Close'][-60:].tolist()
                },
                'forecast_data': {
                    'dates': forecast_dates,
                    'values': ensemble_forecast.tolist(),
                    'upper_band': upper_band,
                    'lower_band': lower_band
                }
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Forecast generation failed: {str(e)}'})

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)
