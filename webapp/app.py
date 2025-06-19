"""
Flask web application for forecast accuracy assessment model.
Provides interactive web interface for model predictions and analysis.
"""

import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objs as go
import plotly.utils
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io
import base64
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models import ARIMAModel, ProphetModel, LSTMModel, ModelEvaluator
from config import PROCESSED_DATA_DIR, MODELS_DIR

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    
    # Create upload directory
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize models
    models = {}
    
    def load_models():
        """Load trained models."""
        try:
            # Load ARIMA model
            arima = ARIMAModel()
            arima.load_model(str(MODELS_DIR / 'arima'))
            models['ARIMA'] = arima
            
            # Load Prophet model
            prophet = ProphetModel()
            prophet.load_model(str(MODELS_DIR / 'prophet'))
            models['Prophet'] = prophet
            
            # Load LSTM model
            lstm = LSTMModel()
            lstm.load_model(str(MODELS_DIR / 'lstm'))
            models['LSTM'] = lstm
            
            print("✅ All models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def allowed_file(filename):
        """Check if file extension is allowed."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def prepare_data(df):
        """Prepare data for forecasting."""
        # Ensure date column exists
        if 'date' not in df.columns:
            # Try to find date-like column
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                df['date'] = pd.to_datetime(df[date_columns[0]])
            else:
                # Create dummy dates if none found
                df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Ensure target column exists
        if 'close_price' not in df.columns:
            # Try to find price-like column
            price_columns = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower()]
            if price_columns:
                df['close_price'] = df[price_columns[0]]
            else:
                # Use first numeric column
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    df['close_price'] = df[numeric_columns[0]]
                else:
                    raise ValueError("No suitable target column found")
        
        # Clean data
        df = df.dropna(subset=['close_price'])
        df = df.sort_values('date')
        
        return df
    
    @app.route('/')
    def index():
        """Main dashboard page."""
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file upload."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read and prepare data
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                
                df = prepare_data(df)
                
                # Save processed data
                processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
                df.to_csv(processed_filepath, index=False)
                
                # Generate data summary
                summary = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'date_range': {
                        'start': df['date'].min().strftime('%Y-%m-%d'),
                        'end': df['date'].max().strftime('%Y-%m-%d')
                    },
                    'target_stats': {
                        'mean': float(df['close_price'].mean()),
                        'std': float(df['close_price'].std()),
                        'min': float(df['close_price'].min()),
                        'max': float(df['close_price'].max())
                    }
                }
                
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and processed successfully',
                    'summary': summary
                })
                
            except Exception as e:
                return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    @app.route('/forecast', methods=['POST'])
    def generate_forecast():
        """Generate forecasts using trained models."""
        try:
            data = request.get_json()
            steps = data.get('steps', 30)
            selected_models = data.get('models', ['ARIMA', 'Prophet', 'LSTM'])
            
            # Load data
            data_file = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
            if not os.path.exists(data_file):
                return jsonify({'error': 'No data uploaded. Please upload data first.'}), 400
            
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Generate forecasts
            forecasts = {}
            for model_name in selected_models:
                if model_name in models:
                    try:
                        model = models[model_name]
                        if model_name == 'Prophet':
                            forecast = model.predict(df, periods=steps)
                        else:
                            forecast = model.predict(df, steps=steps)
                        
                        forecasts[model_name] = forecast.tolist()
                    except Exception as e:
                        forecasts[model_name] = {'error': str(e)}
            
            # Create forecast dates
            last_date = df['date'].max()
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=steps,
                freq='D'
            ).strftime('%Y-%m-%d').tolist()
            
            return jsonify({
                'success': True,
                'forecasts': forecasts,
                'dates': forecast_dates,
                'actual_data': {
                    'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
                    'values': df['close_price'].tolist()
                }
            })
            
        except Exception as e:
            return jsonify({'error': f'Error generating forecast: {str(e)}'}), 500
    
    @app.route('/evaluate', methods=['POST'])
    def evaluate_models():
        """Evaluate model performance."""
        try:
            # Load data
            data_file = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
            if not os.path.exists(data_file):
                return jsonify({'error': 'No data uploaded. Please upload data first.'}), 400
            
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Split data for evaluation
            test_size = int(len(df) * 0.2)
            test_data = df.tail(test_size).copy()
            
            # Evaluate models
            evaluator = ModelEvaluator()
            for name, model in models.items():
                evaluator.add_model(model, name)
            
            results = evaluator.evaluate_all_models(test_data)
            
            # Convert numpy types to native Python types
            for model_name, metrics in results.items():
                for metric, value in metrics.items():
                    if isinstance(value, (np.integer, np.floating)):
                        results[model_name][metric] = float(value)
            
            return jsonify({
                'success': True,
                'results': results,
                'test_size': len(test_data)
            })
            
        except Exception as e:
            return jsonify({'error': f'Error evaluating models: {str(e)}'}), 500
    
    @app.route('/dashboard')
    def dashboard():
        """Dashboard with charts and visualizations."""
        return render_template('dashboard.html')
    
    @app.route('/api/data-summary')
    def data_summary():
        """Get data summary for dashboard."""
        try:
            data_file = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
            if not os.path.exists(data_file):
                return jsonify({'error': 'No data available'}), 404
            
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'])
            
            summary = {
                'total_records': len(df),
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d')
                },
                'price_stats': {
                    'mean': float(df['close_price'].mean()),
                    'std': float(df['close_price'].std()),
                    'min': float(df['close_price'].min()),
                    'max': float(df['close_price'].max())
                },
                'models_loaded': len(models)
            }
            
            return jsonify(summary)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/chart-data')
    def chart_data():
        """Get data for charts."""
        try:
            data_file = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
            if not os.path.exists(data_file):
                return jsonify({'error': 'No data available'}), 404
            
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Prepare chart data
            chart_data = {
                'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': df['close_price'].tolist(),
                'rolling_mean': df['close_price'].rolling(window=30).mean().tolist(),
                'rolling_std': df['close_price'].rolling(window=30).std().tolist()
            }
            
            return jsonify(chart_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/download-forecast')
    def download_forecast():
        """Download forecast results as CSV."""
        try:
            # This would generate and return a CSV file
            # For now, return a sample
            data = {
                'Date': pd.date_range(start='2025-01-01', periods=30, freq='D'),
                'ARIMA_Forecast': np.random.randn(30) * 100 + 1000,
                'Prophet_Forecast': np.random.randn(30) * 100 + 1000,
                'LSTM_Forecast': np.random.randn(30) * 100 + 1000
            }
            
            df = pd.DataFrame(data)
            
            # Create CSV in memory
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name='forecast_results.csv'
            )
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Load models on startup
    load_models()
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000) 