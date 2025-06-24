"""
Flask web application for asset forecasting.
Multi-model stock prediction interface.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import joblib
from pathlib import Path

# Import advanced models
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.arima_model import ARIMAModel
    from models.prophet_model import ProphetModel
    from models.lstm_model import LSTMModel
    from models.enhanced_forecast_model import EnhancedForecastModel
    ADVANCED_MODELS_AVAILABLE = True
    print("✅ Advanced models imported successfully")
except ImportError as e:
    print(f"⚠️  Advanced models not available: {e}")
    ADVANCED_MODELS_AVAILABLE = False

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # --- Cache Busting ---
    @app.after_request
    def add_header(response):
        """
        Add headers to both force latest content and prevent caching.
        """
        response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
        response.headers['Cache-Control'] = 'public, max-age=0, no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}
    
    def load_uploaded_data(ticker):
        """Try to load uploaded data for a ticker"""
        upload_file = os.path.join(app.config['UPLOAD_FOLDER'], f'{ticker.upper()}_data.csv')
        if os.path.exists(upload_file):
            try:
                df = pd.read_csv(upload_file)
                # Standardize column names
                df.columns = df.columns.str.lower()
                
                # Convert date column
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                # Rename columns to match yfinance format
                column_mapping = {
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        df[new_col] = df[old_col]
                
                return df
            except Exception as e:
                print(f"Error loading uploaded data for {ticker}: {e}")
                return None
        return None
    
    @app.route('/')
    def index():
        import time
        return render_template('index.html', current_timestamp=int(time.time()))
    
    @app.route('/dashboard')
    def dashboard():
        return render_template('dashboard.html')
    
    @app.route('/stocks')
    def stocks():
        return render_template('stocks.html')
    
    @app.route('/upload')
    def upload_page():
        return render_template('upload.html')
    
    @app.route('/forecast_plots')
    def forecast_plots():
        plots_dir = Path(os.path.dirname(__file__)).parent / 'forecast_plots_2024'
        if not plots_dir.exists():
            return "Plots directory not found.", 404
        
        plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])
        return render_template('forecast_plots.html', plot_files=plot_files)

    @app.route('/plots/<filename>')
    def get_plot_image(filename):
        plots_dir = Path(os.path.dirname(__file__)).parent / 'forecast_plots_2024'
        return send_from_directory(plots_dir, filename)
    
    @app.route('/api/upload-stock', methods=['POST'])
    def upload_stock():
        """Handle stock data file upload"""
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file uploaded'})
            
            file = request.files['file']
            ticker = request.form.get('ticker', '').upper().strip()
            
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            
            if not ticker:
                return jsonify({'success': False, 'error': 'Ticker symbol required'})
            
            if file and allowed_file(file.filename):
                # Save file with ticker name
                filename = f'{ticker}_data.csv'
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Validate the uploaded data
                try:
                    df = pd.read_csv(filepath)
                    required_columns = ['date', 'close']
                    df.columns = df.columns.str.lower()
                    
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        os.remove(filepath)  # Clean up invalid file
                        return jsonify({
                            'success': False, 
                            'error': f'Missing required columns: {missing_columns}. Required: Date, Close (minimum)'
                        })
                    
                    # Check data quality
                    if len(df) < 100:
                        os.remove(filepath)
                        return jsonify({
                            'success': False,
                            'error': 'File must contain at least 100 data points for reliable forecasting'
                        })
                    
                    return jsonify({
                        'success': True,
                        'message': f'Successfully uploaded data for {ticker}',
                        'data_points': len(df),
                        'date_range': {
                            'start': df['date'].min(),
                            'end': df['date'].max()
                        }
                    })
                    
                except Exception as e:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return jsonify({'success': False, 'error': f'Invalid CSV format: {str(e)}'})
            
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file.'})
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})
    
    @app.route('/api/uploaded-stocks')
    def get_uploaded_stocks():
        """Get list of uploaded stock data files"""
        try:
            uploaded_files = []
            upload_dir = app.config['UPLOAD_FOLDER']
            
            for filename in os.listdir(upload_dir):
                if filename.endswith('_data.csv'):
                    ticker = filename.replace('_data.csv', '')
                    filepath = os.path.join(upload_dir, filename)
                    
                    try:
                        df = pd.read_csv(filepath)
                        df.columns = df.columns.str.lower()
                        
                        uploaded_files.append({
                            'ticker': ticker,
                            'filename': filename,
                            'data_points': len(df),
                            'date_range': {
                                'start': df['date'].min(),
                                'end': df['date'].max()
                            },
                            'upload_time': os.path.getmtime(filepath)
                        })
                    except:
                        continue
            
            return jsonify({'success': True, 'uploaded_stocks': uploaded_files})
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/stocks')
    def api_stocks():
        """Get available stocks from the dataset"""
        try:
            # Load the combined dataset
            data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'combined_ftse_sp500_data.csv')
            df = pd.read_csv(data_file)
            
            # Get unique stocks with their info
            stock_info = df[['ticker', 'company_name', 'index']].drop_duplicates().sort_values('ticker')
            
            stocks_list = []
            for _, row in stock_info.iterrows():
                # Map index names to standard format
                market = 'FTSE 100' if row['index'] == 'FTSE100' else 'S&P 500'
                
                stocks_list.append({
                    'ticker': row['ticker'],
                    'name': row['company_name'],
                    'market': market
                })
            
            return jsonify({
                'success': True,
                'stocks': stocks_list
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to load stocks: {str(e)}'
            })
    
    @app.route('/api/stock/<ticker>')
    def api_stock_data(ticker):
        """Get detailed data for a specific stock"""
        try:
            # Load the combined dataset
            data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'combined_ftse_sp500_data.csv')
            df = pd.read_csv(data_file)
            
            # Filter for the specific ticker
            stock_data = df[df['ticker'] == ticker.upper()].copy()
            
            if stock_data.empty:
                return jsonify({
                    'success': False,
                    'error': f'No data found for ticker {ticker}'
                })
            
            # Sort by date
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data = stock_data.sort_values('date')
            
            # Get basic info
            company_name = stock_data['company_name'].iloc[0]
            market = 'FTSE 100' if stock_data['index'].iloc[0] == 'FTSE100' else 'S&P 500'
            
            # Calculate basic metrics
            current_price = float(stock_data['close_price'].iloc[-1])
            first_price = float(stock_data['close_price'].iloc[0])
            total_return = ((current_price - first_price) / first_price) * 100
            
            # Prepare historical data
            historical_data = {
                'dates': stock_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': stock_data['close_price'].tolist(),
                'volumes': stock_data['volume'].tolist(),
                'highs': stock_data['high'].tolist(),
                'lows': stock_data['low'].tolist()
            }
            
            return jsonify({
                'success': True,
                'ticker': ticker.upper(),
                'name': company_name,
                'market': market,
                'current_price': current_price,
                'total_return': total_return,
                'data_points': len(stock_data),
                'date_range': {
                    'start': stock_data['date'].min().strftime('%Y-%m-%d'),
                    'end': stock_data['date'].max().strftime('%Y-%m-%d')
                },
                'historical_data': historical_data
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to load data for {ticker}: {str(e)}'
            })
    
    @app.route('/api/forecast-asset', methods=['POST'])
    def forecast_asset():
        try:
            data = request.get_json()
            ticker = data.get('ticker', '').upper().strip()
            days = int(data.get('days', 30))
            advanced = data.get('advanced', True)  # Use advanced models by default
            
            if not ticker:
                return jsonify({'success': False, 'error': 'Ticker required'})
            
            # Try to load uploaded data first, fallback to yfinance
            hist_data = load_uploaded_data(ticker)
            data_source = 'uploaded'
            
            if hist_data is None:
                # Fallback to yfinance - use more data for better models
                stock = yf.Ticker(ticker)
                hist_data = stock.history(period='5y')  # Use 5 years for better forecasting
                data_source = 'yfinance'
            
            if hist_data.empty:
                return jsonify({'success': False, 'error': f'No data found for {ticker}. Try uploading data or check ticker symbol.'})
            
            # Prepare data
            dates = [d.strftime('%Y-%m-%d') for d in hist_data.index]
            prices = hist_data['Close'].tolist()
            
            # Store all forecasts and model info
            all_forecasts = {}
            model_info = {}
            
            # Use advanced models if available and requested
            if advanced and ADVANCED_MODELS_AVAILABLE and len(hist_data) >= 100:
                try:
                    # Prepare data for advanced models
                    df_advanced = hist_data.reset_index()
                    
                    # Fix timezone issues for Prophet model
                    # Check if the original index had timezone info before reset_index()
                    if hasattr(hist_data.index, 'tz') and hist_data.index.tz is not None:
                        # The first column after reset_index() contains the original index
                        if hasattr(df_advanced.iloc[:, 0], 'dt') and df_advanced.iloc[:, 0].dt.tz is not None:
                            df_advanced.iloc[:, 0] = df_advanced.iloc[:, 0].dt.tz_localize(None)
                    
                    # Fix column naming to handle variable number of columns from yfinance
                    available_columns = list(df_advanced.columns)
                    
                    # Create proper column mapping
                    if len(available_columns) == 6:  # Date + OHLCV
                        df_advanced.columns = ['date', 'open', 'high', 'low', 'close_price', 'volume']
                    elif len(available_columns) == 7:  # Date + OHLCV + Dividends
                        df_advanced.columns = ['date', 'open', 'high', 'low', 'close_price', 'volume', 'dividends']
                    elif len(available_columns) == 8:  # Date + OHLCV + Dividends + Splits
                        df_advanced.columns = ['date', 'open', 'high', 'low', 'close_price', 'volume', 'dividends', 'stock_splits']
                    else:
                        # Fallback: manually map the essential columns
                        column_mapping = {}
                        column_mapping[available_columns[0]] = 'date'  # First column is always date/index
                        
                        # Find the main OHLC columns
                        for i, col in enumerate(available_columns[1:], 1):
                            col_lower = col.lower()
                            if 'open' in col_lower:
                                column_mapping[col] = 'open'
                            elif 'high' in col_lower:
                                column_mapping[col] = 'high'
                            elif 'low' in col_lower:
                                column_mapping[col] = 'low'
                            elif 'close' in col_lower:
                                column_mapping[col] = 'close_price'
                            elif 'volume' in col_lower:
                                column_mapping[col] = 'volume'
                        
                        df_advanced = df_advanced.rename(columns=column_mapping)
                    
                    # Ensure we have the minimum required columns
                    if 'close_price' not in df_advanced.columns:
                        # Try to find close price in any remaining column
                        for col in df_advanced.columns:
                            if 'close' in col.lower():
                                df_advanced = df_advanced.rename(columns={col: 'close_price'})
                                break
                    
                    # Remove timezone from date column if it exists
                    if 'date' in df_advanced.columns and hasattr(df_advanced['date'], 'dt'):
                        if df_advanced['date'].dt.tz is not None:
                            df_advanced['date'] = df_advanced['date'].dt.tz_localize(None)
                    
                    print(f"✅ Data prepared for advanced models: {df_advanced.columns.tolist()}")
                    
                    # 1. ARIMA Model
                    try:
                        arima_model = ARIMAModel(order=(2, 1, 2))
                        arima_model.fit(df_advanced)
                        arima_forecast = arima_model.predict(df_advanced, steps=days)
                        arima_forecast = np.maximum(arima_forecast, 0.01)
                        
                        all_forecasts['ARIMA'] = arima_forecast.tolist()
                        model_info['ARIMA'] = {
                            'status': 'success',
                            'type': f'ARIMA({arima_model.order}) on {len(hist_data)} days',
                            'aic': arima_model.training_history.get('aic', 'N/A'),
                            'confidence': 'high'
                        }
                    except Exception as e:
                        model_info['ARIMA'] = {'status': 'failed', 'error': str(e)[:50]}
                    
                    # 2. Prophet Model
                    try:
                        prophet_model = ProphetModel(
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False
                        )
                        prophet_model.fit(df_advanced)
                        prophet_forecast = prophet_model.predict(df_advanced, periods=days)
                        prophet_forecast = np.maximum(prophet_forecast, 0.01)
                        
                        all_forecasts['Prophet'] = prophet_forecast.tolist()
                        model_info['Prophet'] = {
                            'status': 'success',
                            'type': f'Prophet with seasonality on {len(hist_data)} days',
                            'seasonality': 'yearly + weekly',
                            'confidence': 'high'
                        }
                    except Exception as e:
                        model_info['Prophet'] = {'status': 'failed', 'error': str(e)[:50]}
                    
                    # 3. LSTM Model (if enough data)
                    if len(hist_data) >= 200:
                        try:
                            lstm_model = LSTMModel(
                                sequence_length=60,
                                units=50,
                                layers=2,
                                dropout=0.2,
                                learning_rate=0.001
                            )
                            lstm_model.fit(df_advanced, epochs=50, batch_size=32)
                            lstm_forecast = lstm_model.predict(df_advanced, steps=days)
                            lstm_forecast = np.maximum(lstm_forecast, 0.01)
                            
                            all_forecasts['LSTM'] = lstm_forecast.tolist()
                            model_info['LSTM'] = {
                                'status': 'success',
                                'type': f'Deep LSTM on {len(hist_data)} days',
                                'architecture': '2-layer LSTM with 50 units',
                                'confidence': 'very high'
                            }
                        except Exception as e:
                            model_info['LSTM'] = {'status': 'failed', 'error': str(e)[:50]}
                
                except Exception as e:
                    print(f"❌ Advanced models failed: {e}")
                    print(f"Data shape: {hist_data.shape if hist_data is not None else 'None'}")
                    print(f"Data columns: {hist_data.columns.tolist() if hist_data is not None else 'None'}")
                    import traceback
                    print(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback to basic models if advanced models failed or not available
            if not all_forecasts:
                advanced = False
            
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
                    # Create a matrix of all successful forecasts
                    forecast_matrix = np.array(list(successful_forecasts.values()))
                    
                    # Calculate standard deviation across models for each day
                    daily_std = np.std(forecast_matrix, axis=0)
                    
                    # Dynamic weights based on model sophistication and data availability
                    weights = {}
                    if advanced:
                        # Advanced model weights (prioritize sophisticated models)
                        weights = {
                            'ARIMA': 0.30,
                            'Prophet': 0.30, 
                            'LSTM': 0.35,
                            'Linear Regression': 0.05,
                            'Moving Average': 0.10,
                            'Exponential Smoothing': 0.10,
                            'Polynomial Regression': 0.05,
                            'Seasonal Model': 0.10
                        }
                    else:
                        # Basic model weights
                        weights = {
                            'Linear Regression': 0.20,
                            'Moving Average': 0.25,
                            'Exponential Smoothing': 0.25,
                            'Polynomial Regression': 0.15,
                            'Seasonal Model': 0.15
                        }
                    
                    # Add feature engineering - Technical indicators
                    prices_array = hist_data['Close'].values
                    
                    # Calculate technical indicators for context
                    if len(prices_array) >= 50:
                        # RSI (Relative Strength Index)
                        delta = np.diff(prices_array)
                        gain = np.where(delta > 0, delta, 0)
                        loss = np.where(delta < 0, -delta, 0)
                        avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
                        avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
                        rsi = 100 - (100 / (1 + (avg_gain / avg_loss if avg_loss > 0 else 1)))
                        
                        # Bollinger Bands
                        sma_20 = np.mean(prices_array[-20:])
                        std_20 = np.std(prices_array[-20:])
                        upper_band = sma_20 + (2 * std_20)
                        lower_band = sma_20 - (2 * std_20)
                        current_price = prices_array[-1]
                        bb_position = (current_price - lower_band) / (upper_band - lower_band)
                        
                        # MACD
                        ema_12 = prices_array[-1]  # Simplified
                        ema_26 = np.mean(prices_array[-26:]) if len(prices_array) >= 26 else prices_array[-1]
                        macd = ema_12 - ema_26
                        
                        # Volatility adjustment
                        volatility = np.std(prices_array[-30:]) / np.mean(prices_array[-30:])
                        
                        # Adjust ensemble weights based on technical indicators
                        if rsi > 70:  # Overbought - reduce bullish weights
                            for model in ['Linear Regression', 'LSTM']:
                                if model in weights:
                                    weights[model] *= 0.8
                        elif rsi < 30:  # Oversold - increase bullish weights
                            for model in ['Prophet', 'ARIMA']:
                                if model in weights:
                                    weights[model] *= 1.2
                        
                        # High volatility - favor more conservative models
                        if volatility > 0.05:
                            for model in ['Moving Average', 'Exponential Smoothing']:
                                if model in weights:
                                    weights[model] *= 1.3
                    
                    ensemble_forecast = np.zeros(days)
                    total_weight = 0
                    
                    for name, forecast in successful_forecasts.items():
                        weight = weights.get(name, 0.1)  # Default weight
                        ensemble_forecast += np.array(forecast) * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        ensemble_forecast /= total_weight
                        
                        # Enhanced confidence interval calculation
                        # Use model agreement + historical volatility
                        model_disagreement = np.std(forecast_matrix, axis=0)
                        historical_volatility = np.std(prices_array[-60:]) if len(prices_array) >= 60 else np.std(prices_array)
                        
                        confidence_factor = 1.96  # 95% confidence
                        upper_bound = ensemble_forecast + (model_disagreement + historical_volatility * 0.1) * confidence_factor
                        lower_bound = ensemble_forecast - (model_disagreement + historical_volatility * 0.1) * confidence_factor
                        lower_bound = np.maximum(lower_bound, 0.01).tolist()
                        
                        all_forecasts['Ensemble'] = ensemble_forecast.tolist()
                        model_info['Ensemble'] = {
                            'status': 'success',
                            'type': f'Advanced weighted ensemble of {len(successful_forecasts)} models',
                            'models_combined': list(successful_forecasts.keys()),
                            'upper_bound': upper_bound.tolist(),
                            'lower_bound': lower_bound,
                            'weighting': 'dynamic based on technical indicators',
                            'confidence': 'very high' if advanced else 'high'
                        }
                        
                        # Create realistic forecast with market volatility
                        try:
                            # Calculate historical daily volatility
                            returns = hist_data['Close'].pct_change().dropna()
                            daily_volatility = returns.std()
                            
                            # Create realistic forecast by adding volatility to ensemble
                            realistic_forecast = []
                            current_realistic_price = ensemble_forecast[0]
                            
                            for i in range(len(ensemble_forecast)):
                                # Get the base trend from ensemble
                                if i == 0:
                                    trend_change = 0
                                else:
                                    trend_change = (ensemble_forecast[i] - ensemble_forecast[i-1]) / ensemble_forecast[i-1]
                                
                                # Add realistic daily volatility (random walk)
                                np.random.seed(42 + i)  # Reproducible randomness
                                random_shock = np.random.normal(0, daily_volatility)
                                
                                # Apply trend + volatility
                                current_realistic_price = current_realistic_price * (1 + trend_change + random_shock * 0.7)
                                
                                # Ensure price stays positive
                                current_realistic_price = max(current_realistic_price, 0.01)
                                realistic_forecast.append(current_realistic_price)
                            
                            all_forecasts['Realistic Forecast'] = realistic_forecast
                            model_info['Realistic Forecast'] = {
                                'status': 'success',
                                'type': f'Ensemble + market volatility simulation',
                                'volatility': f'{daily_volatility:.4f}',
                                'description': 'Adds realistic market noise to smooth ensemble forecast',
                                'confidence': 'medium-high'
                            }
                        except Exception as e:
                            model_info['Realistic Forecast'] = {'status': 'failed', 'error': str(e)[:50]}
                except Exception as e:
                    model_info['Ensemble'] = {'status': 'failed', 'error': str(e)[:50]}
            
            # === ENHANCED FORECASTING WITH EXTERNAL DATA ===
            try:
                if successful_forecasts and ADVANCED_MODELS_AVAILABLE:
                    enhanced_model = EnhancedForecastModel()
                    
                    # Generate enhanced forecasts using external data
                    enhanced_result = enhanced_model.generate_enhanced_forecast(
                        ticker=ticker,
                        hist_data=hist_data,
                        base_forecasts=successful_forecasts,
                        days=days
                    )
                    
                    if enhanced_result['forecasts']:
                        # Add enhanced forecasts to the main forecasts
                        all_forecasts.update(enhanced_result['forecasts'])
                        
                        # Add enhanced model info
                        for model_name in enhanced_result['forecasts']:
                            model_info[model_name] = {
                                'status': 'success',
                                'type': 'Enhanced with external data',
                                'data_sources': ['Economic indicators', 'Sentiment analysis', 'Fundamental metrics', 'Technical indicators'],
                                'confidence': 'very high',
                                'external_bias': f"{enhanced_result['external_summary'].get('overall_bias', 0.0):.3f}",
                                'color': '#9c27b0' if 'Enhanced_Ensemble' in model_name else '#6f42c1'
                            }
                        
                        # Update color mapping for enhanced models
                        model_colors = {
                            'Linear Regression': '#28a745',
                            'Moving Average': '#dc3545', 
                            'Exponential Smoothing': '#ffc107',
                            'Polynomial Regression': '#17a2b8',
                            'Seasonal Model': '#6c757d',
                            'Ensemble': '#007bff',
                            'Realistic Forecast': '#fd7e14',
                            'Enhanced_Ensemble': '#9c27b0',
                            'Linear Regression_Enhanced': '#20c997',
                            'Moving Average_Enhanced': '#e83e8c',
                            'Exponential Smoothing_Enhanced': '#ffc107',
                            'ARIMA_Enhanced': '#6610f2',
                            'Prophet_Enhanced': '#fd7e14',
                            'LSTM_Enhanced': '#6f42c1'
                        }
                        
                        # Log external data impact
                        external_summary = enhanced_result.get('external_summary', {})
                        print(f"📊 Enhanced forecasting for {ticker}:")
                        print(f"   Economic impact: {external_summary.get('economic_impact', 0.0):.3f}")
                        print(f"   Sentiment impact: {external_summary.get('sentiment_impact', 0.0):.3f}")
                        print(f"   Overall bias: {external_summary.get('overall_bias', 0.0):.3f}")
                        
            except Exception as e:
                print(f"❌ Enhanced forecasting failed: {e}")
                # Continue with regular forecasting
            
            # Generate future dates (skip weekends, but ensure continuous business days)
            future_dates = []
            current_date = hist_data.index[-1]
            
            # Generate exactly 'days' number of business days
            days_generated = 0
            while days_generated < days:
                current_date += timedelta(days=1)
                # Skip weekends (Saturday=5, Sunday=6)
                if current_date.weekday() < 5:  # Monday=0 to Friday=4
                    future_dates.append(current_date.strftime('%Y-%m-%d'))
                    days_generated += 1
            
            # Choose primary forecast (prefer enhanced ensemble, then ensemble, then best available)
            if 'Enhanced_Ensemble' in all_forecasts:
                primary_forecast = all_forecasts['Enhanced_Ensemble']
                primary_model = 'Enhanced_Ensemble'
            elif 'Ensemble' in all_forecasts:
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
            
            # Create seamless connection between historical and forecast data
            # Add the last historical price as the first point of the forecast
            connected_forecast_dates = [dates[-1]] + future_dates
            connected_forecast_prices = [current_price] + primary_forecast
            
            # Also create connected versions for all forecasts
            connected_all_forecasts = {}
            for model_name, forecast_data in all_forecasts.items():
                connected_all_forecasts[model_name] = [current_price] + forecast_data
            
            # Create historical data in the format expected by frontend
            historical_data = [
                {'date': date, 'price': price} 
                for date, price in zip(dates[-60:], prices[-60:])
            ]
            
            # Create confidence bounds for the frontend
            if 'Ensemble' in model_info and 'upper_bound' in model_info['Ensemble']:
                upper_bound = model_info['Ensemble']['upper_bound']
                lower_bound = model_info['Ensemble']['lower_bound']
            else:
                # Fallback confidence bounds
                primary_array = np.array(primary_forecast)
                std_dev = np.std(primary_array) if len(primary_array) > 1 else primary_array[0] * 0.05
                upper_bound = (primary_array + std_dev).tolist()
                lower_bound = (primary_array - std_dev).tolist()
                lower_bound = np.maximum(lower_bound, 0.01).tolist()
            
            # Add colors for each model (enhanced models already have colors set above)
            if 'model_colors' not in locals():
                model_colors = {
                    'Linear Regression': '#28a745',
                    'Moving Average': '#dc3545', 
                    'Exponential Smoothing': '#ffc107',
                    'Polynomial Regression': '#17a2b8',
                    'Seasonal Model': '#6c757d',
                    'Ensemble': '#007bff',
                    'Realistic Forecast': '#fd7e14',  # Orange color for realistic forecast
                    'Enhanced_Ensemble': '#9c27b0',
                    'ARIMA': '#6610f2',
                    'Prophet': '#20c997',
                    'LSTM': '#e83e8c'
                }
            
            for model_name in model_info:
                if model_name in model_colors:
                    model_info[model_name]['color'] = model_colors[model_name]
                else:
                    model_info[model_name]['color'] = '#343a40'

            return jsonify({
                'ticker': ticker,
                'historical': historical_data,
                'forecasts': all_forecasts,
                'future_dates': future_dates,
                'model_info': model_info,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'change_percent': change_percent,
                'summary': {
                    'primary_model': primary_model,
                    'models_used': len(successful_forecasts),
                    'trend': 'Bullish' if change_percent > 0 else 'Bearish',
                    'data_source': data_source,
                    'data_points': len(hist_data)
                }
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/external-data/<ticker>')
    def get_external_data(ticker):
        """Get external data factors for a given ticker"""
        try:
            if not ADVANCED_MODELS_AVAILABLE:
                return jsonify({'success': False, 'error': 'Enhanced models not available'})
            
            enhanced_model = EnhancedForecastModel()
            
            # Get sample historical data for the ticker
            hist_data = yf.Ticker(ticker).history(period="1y")
            if hist_data.empty:
                return jsonify({'success': False, 'error': 'No data available for ticker'})
            
            # Get external data
            start_date = (hist_data.index[0] - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = hist_data.index[-1].strftime('%Y-%m-%d')
            
            economic_data = enhanced_model.get_economic_indicators(start_date, end_date)
            sentiment_score = enhanced_model.get_sentiment_score(ticker)
            fundamental_metrics = enhanced_model.get_fundamental_metrics(ticker)
            technical_indicators = enhanced_model.calculate_technical_indicators(hist_data['Close'])
            
            # Calculate current impacts
            adjustments = enhanced_model._calculate_adjustments(
                economic_data, sentiment_score, fundamental_metrics, technical_indicators
            )
            
            return jsonify({
                'success': True,
                'ticker': ticker,
                'external_factors': {
                    'economic': {
                        'current_data': economic_data.tail(1).to_dict('records')[0] if not economic_data.empty else {},
                        'impact': adjustments.get('economic_factor', 0.0),
                        'description': 'Federal funds rate, unemployment, treasury rates, market volatility (VIX)'
                    },
                    'sentiment': {
                        'score': sentiment_score,
                        'impact': adjustments.get('sentiment_factor', 0.0),
                        'description': 'Social media sentiment, news analysis, market trends'
                    },
                    'fundamental': {
                        'metrics': fundamental_metrics,
                        'description': 'P/E ratio, P/B ratio, ROE, Beta and other financial health indicators'
                    },
                    'technical': {
                        'indicators': technical_indicators,
                        'impact': adjustments.get('technical_factor', 0.0),
                        'description': 'RSI, moving averages, volatility, momentum indicators'
                    },
                    'overall_bias': adjustments.get('overall_bias', 0.0),
                    'interpretation': {
                        'bias_direction': 'Bullish' if adjustments.get('overall_bias', 0.0) > 0 else 'Bearish' if adjustments.get('overall_bias', 0.0) < 0 else 'Neutral',
                        'confidence': 'High' if abs(adjustments.get('overall_bias', 0.0)) > 0.05 else 'Medium' if abs(adjustments.get('overall_bias', 0.0)) > 0.02 else 'Low'
                    }
                }
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/dashboard-data')
    def api_dashboard_data():
        """Get data for the main dashboard"""
        try:
            # Load the main processed dataset
            data_file = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'processed_data.csv')
            df = pd.read_csv(data_file, parse_dates=['date'])
            
            # 1. Price Time Series data (for a representative stock)
            sample_stock = df[df['ticker'] == 'AZN.L'].copy()
            time_series_data = {
                'dates': sample_stock['date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': sample_stock['close_price'].tolist()
            }
            
            # Placeholder for model performance
            model_performance = {
                'models': ['ARIMA', 'Prophet', 'LSTM', 'Ensemble'],
                'mape': [5.2, 4.8, 6.5, 4.1],  # Example data
                'rmse': [120.5, 115.2, 130.0, 110.8] # Example data
            }
            
            # Placeholder for forecast accuracy
            forecast_accuracy = {
                'horizon': ['1-day', '7-day', '30-day'],
                'accuracy': [98.5, 92.1, 85.4] # Example data
            }

            return jsonify({
                'success': True,
                'time_series': time_series_data,
                'model_performance': model_performance,
                'forecast_accuracy': forecast_accuracy
            })

        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to load dashboard data: {str(e)}'})
    
    @app.route('/api/search-tickers')
    def search_tickers():
        """Search for stock tickers"""
        try:
            query = request.args.get('q', '').upper()
            if not query:
                return jsonify([])

            data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'combined_ftse_sp500_data.csv')
            df = pd.read_csv(data_file)
            
            # Get unique stocks
            all_tickers = df[['ticker', 'company_name']].drop_duplicates()
            
            # Filter based on query (ticker or company name)
            mask = all_tickers['ticker'].str.contains(query) | all_tickers['company_name'].str.contains(query, case=False)
            matched_tickers = all_tickers[mask]
            
            # Format for response with correct field names
            results = []
            for _, row in matched_tickers.head(10).iterrows():
                results.append({
                    'symbol': row['ticker'],
                    'name': row['company_name']
                })
            
            return jsonify(results)

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/backtest-asset', methods=['POST'])
    def backtest_asset():
        """Run a backtest for a given ticker."""
        try:
            data = request.get_json()
            ticker = data.get('ticker', '').upper().strip()
            days = int(data.get('days', 30))

            if not ticker:
                return jsonify({'success': False, 'error': 'Ticker required'})

            # Get stock data, but for a period ending `days` ago
            end_date = datetime.now() - timedelta(days=days)
            start_date = end_date - timedelta(days=365) # 1 year of data for backtesting
            
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            
            if hist_data.empty:
                return jsonify({'success': False, 'error': f'Not enough historical data for backtest on {ticker}'})

            # The "actual" data is from the end of our backtest period onwards
            actual_data = stock.history(start=end_date.strftime('%Y-%m-%d'), period=f"{days+5}d") # get a bit extra
            
            if actual_data.empty or len(actual_data) < days:
                return jsonify({'success': False, 'error': f'Not enough actual data for {days}-day backtest'})
                
            actual_dates = [d.strftime('%Y-%m-%d') for d in actual_data.index[:days]]
            actual_prices = actual_data['Close'].tolist()[:days]

            # Now, run the forecast logic on the historical slice of data
            # Prepare data for models (same format as forecast_asset)
            dates = [d.strftime('%Y-%m-%d') for d in hist_data.index]
            prices = hist_data['Close'].tolist()
            
            if len(prices) < 30:
                return jsonify({'success': False, 'error': 'Not enough historical data for backtesting'})

            # Run forecasting models on historical data
            all_forecasts = {}
            model_info = {}
            successful_forecasts = {}
            
            # Basic models
            try:
                from models.base_model import LinearRegressionModel, MovingAverageModel, ExponentialSmoothingModel
                
                # Linear Regression
                lr_model = LinearRegressionModel()
                lr_forecast = lr_model.forecast(prices, days)
                if lr_forecast:
                    all_forecasts['Linear Regression'] = lr_forecast
                    successful_forecasts['Linear Regression'] = lr_forecast
                    model_info['Linear Regression'] = {'method': 'Statistical', 'color': '#28a745'}
                
                # Moving Average
                ma_model = MovingAverageModel()
                ma_forecast = ma_model.forecast(prices, days)
                if ma_forecast:
                    all_forecasts['Moving Average'] = ma_forecast
                    successful_forecasts['Moving Average'] = ma_forecast
                    model_info['Moving Average'] = {'method': 'Statistical', 'color': '#dc3545'}
                
                # Exponential Smoothing
                es_model = ExponentialSmoothingModel()
                es_forecast = es_model.forecast(prices, days)
                if es_forecast:
                    all_forecasts['Exponential Smoothing'] = es_forecast
                    successful_forecasts['Exponential Smoothing'] = es_forecast
                    model_info['Exponential Smoothing'] = {'method': 'Statistical', 'color': '#ffc107'}
                    
            except Exception as e:
                print(f"Basic models failed in backtest: {e}")

            # Advanced models
            try:
                from models.arima_model import ARIMAModel
                from models.prophet_model import ProphetModel
                from models.lstm_model import LSTMModel
                
                # ARIMA
                try:
                    arima_model = ARIMAModel()
                    arima_forecast = arima_model.forecast(prices, days)
                    if arima_forecast and len(arima_forecast) == days:
                        all_forecasts['ARIMA'] = arima_forecast
                        successful_forecasts['ARIMA'] = arima_forecast
                        model_info['ARIMA'] = {'method': 'Advanced', 'color': '#6f42c1'}
                except Exception as e:
                    print(f"ARIMA backtest failed: {e}")
                
                # Prophet
                try:
                    prophet_model = ProphetModel()
                    # Prophet needs dates and prices
                    df_prophet = pd.DataFrame({
                        'ds': pd.to_datetime(dates),
                        'y': prices
                    })
                    prophet_forecast = prophet_model.forecast_dataframe(df_prophet, days)
                    if prophet_forecast and len(prophet_forecast) == days:
                        all_forecasts['Prophet'] = prophet_forecast
                        successful_forecasts['Prophet'] = prophet_forecast
                        model_info['Prophet'] = {'method': 'Advanced', 'color': '#20c997'}
                except Exception as e:
                    print(f"Prophet backtest failed: {e}")
                
                # LSTM
                try:
                    lstm_model = LSTMModel()
                    lstm_forecast = lstm_model.forecast(prices, days)
                    if lstm_forecast and len(lstm_forecast) == days:
                        all_forecasts['LSTM'] = lstm_forecast
                        successful_forecasts['LSTM'] = lstm_forecast
                        model_info['LSTM'] = {'method': 'Advanced', 'color': '#fd7e14'}
                except Exception as e:
                    print(f"LSTM backtest failed: {e}")
                    
            except Exception as e:
                print(f"Advanced models import failed in backtest: {e}")
            
            # Create ensemble if we have successful forecasts
            if successful_forecasts:
                try:
                    from models.ensemble_model import EnsembleModel
                    ensemble_model = EnsembleModel()
                    ensemble_forecast = ensemble_model.forecast(successful_forecasts, days)
                    if ensemble_forecast:
                        all_forecasts['Ensemble'] = ensemble_forecast
                        model_info['Ensemble'] = {'method': 'Ensemble', 'color': '#007bff'}
                except Exception as e:
                    print(f"Ensemble backtest failed: {e}")
            
            # If no models worked, create a simple fallback
            if not all_forecasts:
                # Simple linear trend as fallback
                if len(prices) >= 2:
                    slope = (prices[-1] - prices[-30]) / 30 if len(prices) >= 30 else (prices[-1] - prices[0]) / len(prices)
                    fallback_forecast = [prices[-1] + slope * (i + 1) for i in range(days)]
                    all_forecasts['Linear Trend'] = fallback_forecast
                    model_info['Linear Trend'] = {'method': 'Fallback', 'color': '#6c757d'}

            return jsonify({
                'success': True,
                'dates': actual_dates,
                'actual': actual_prices,
                'forecasts': all_forecasts,
                'model_info': model_info
            })

        except Exception as e:
            return jsonify({'success': False, 'error': f'Backtest failed: {str(e)}'})
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)