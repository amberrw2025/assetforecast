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
from pathlib import Path

# Import advanced models
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.arima_model import ARIMAModel
    from models.prophet_model import ProphetModel
    from models.enhanced_forecast_model import EnhancedForecastModel
    ADVANCED_MODELS_AVAILABLE = True
    print("✅ Advanced models imported successfully")
except ImportError as e:
    print(f"⚠️  Advanced models not available: {e}")
    ADVANCED_MODELS_AVAILABLE = False

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-default-secret-key')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    @app.after_request
    def add_header(response):
        response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
        response.headers['Cache-Control'] = 'public, max-age=0, no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}
    
    @app.route('/')
    def index():
        return render_template('index.html')

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

    @app.route('/api/forecast-asset', methods=['POST'])
    def forecast_asset():
        """Main endpoint to generate and return asset forecasts."""
        try:
            # 1. Get and validate request data
            req_data = request.get_json()
            ticker = req_data.get('ticker', '').upper()
            forecast_days = int(req_data.get('forecast_days', 30))
            use_enhanced = req_data.get('use_enhanced', False)
            
            if not ticker or not ADVANCED_MODELS_AVAILABLE:
                return jsonify({'success': False, 'error': 'Invalid request or models not available'}), 400

            # 2. Fetch and prepare data
            hist_data = yf.download(ticker, period='5y', progress=False)
            if hist_data.empty:
                return jsonify({'success': False, 'error': f'No data for {ticker}'}), 404
            
            df_for_models = hist_data[['Close']].copy().reset_index()
            df_for_models.columns = ['date', 'y']
            df_for_models['date'] = pd.to_datetime(df_for_models['date'])

            # 3. Run selected models
            successful_forecasts = {}
            
            # --- ARIMA ---
            try:
                arima_model = ARIMAModel()
                arima_model.fit(df_for_models)
                forecast = arima_model.predict(df=df_for_models, steps=forecast_days)
                successful_forecasts['arima'] = np.array(forecast).tolist()
            except Exception as e:
                print(f"❌ ARIMA model failed: {e}")

            # --- Prophet ---
            try:
                prophet_model = ProphetModel()
                prophet_model.fit(df_for_models)
                forecast = prophet_model.predict(df=df_for_models, periods=forecast_days)
                successful_forecasts['prophet'] = np.array(forecast).tolist()
            except Exception as e:
                print(f"❌ Prophet model failed: {e}")

            # 4. Create Ensemble Forecast
            ensemble_forecast = []
            valid_forecasts = [f for f in successful_forecasts.values() if f]
            if len(valid_forecasts) > 1:
                min_len = min(len(f) for f in valid_forecasts)
                trimmed = [f[:min_len] for f in valid_forecasts]
                ensemble_forecast = np.mean(trimmed, axis=0).tolist()
            
            # 5. Run Enhanced (VIX Simulation) Forecast if requested
            enhanced_forecast = []
            if use_enhanced:
                try:
                    enhanced_model = EnhancedForecastModel()
                    forecast_start_date = hist_data.index[-1] + timedelta(days=1)
                    forecast_dates_pd = pd.date_range(start=forecast_start_date, periods=forecast_days, freq='D')
                    
                    economic_data = enhanced_model.get_economic_indicators(
                        start_date=forecast_start_date.strftime('%Y-%m-%d'),
                        end_date=forecast_dates_pd[-1].strftime('%Y-%m-%d'),
                        ticker=ticker
                    )
                    
                    enhanced_results = enhanced_model.generate_enhanced_forecast(
                        ticker=ticker, hist_data=hist_data, forecast_dates=forecast_dates_pd,
                        vix_forecast_period_data=economic_data
                    )
                    sim_path = enhanced_results.get('forecasts', {}).get('VIX_Simulated_Path')
                    if sim_path:
                        enhanced_forecast = sim_path
                except Exception as e:
                    print(f"❌ Enhanced simulation failed: {e}")
            
            # 6. Prepare final JSON response
            last_date = hist_data.index[-1]
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

            response = {
                'success': True,
                'ticker': ticker,
                'historical_data': {
                    'dates': hist_data.index.strftime('%Y-%m-%d').tolist(),
                    'prices': hist_data['Close'].values.flatten().tolist()
                },
                'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
                'arima_forecast': successful_forecasts.get('arima', []),
                'prophet_forecast': successful_forecasts.get('prophet', []),
                'ensemble_forecast': ensemble_forecast,
            }
            if enhanced_forecast:
                response['enhanced_forecast'] = enhanced_forecast
            
            return jsonify(response)

        except Exception as e:
            print(f"🔥 Unhandled exception in forecast_asset: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': 'A critical error occurred on the server.'}), 500
    
    @app.route('/api/search-tickers')
    def search_tickers():
        """Provides ticker suggestions for the search bar."""
        query = request.args.get('q', '').upper()
        if not query:
            return jsonify([])
        
        # This is a placeholder. For a real app, use a database or a pre-compiled list.
        # For now, we'll use a simple hardcoded list of popular tickers.
        all_tickers = {
            "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com, Inc.", "TSLA": "Tesla, Inc.", "NVDA": "NVIDIA Corporation",
            "JPM": "JPMorgan Chase & Co.", "JNJ": "Johnson & Johnson", "V": "Visa Inc.",
            "WMT": "Walmart Inc.", "PG": "Procter & Gamble Company",
            "AZN.L": "AstraZeneca PLC", "HSBA.L": "HSBC Holdings PLC", "ULVR.L": "Unilever PLC"
        }
        
        matched = {t: n for t, n in all_tickers.items() if query in t or query in n.upper()}
        results = [{'symbol': t, 'name': n} for t, n in matched.items()]
        return jsonify(results[:10])

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)