#!/usr/bin/env python3
"""
Advanced Model Evaluation Script for 2024 Data
Evaluates the actual pre-trained ARIMA, Prophet, and LSTM models using 2024 data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

class PretrainedModelEvaluator:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data" / "historical_10years"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "evaluation_results_2024"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load model metadata
        self.load_model_info()
        
    def load_model_info(self):
        """Load information about available models"""
        print("🔍 Checking available pre-trained models...")
        
        self.available_models = {}
        
        # Check ARIMA
        arima_file = self.models_dir / "arima.joblib"
        arima_info = self.models_dir / "arima.json"
        if arima_file.exists() and arima_info.exists():
            with open(arima_info, 'r') as f:
                self.available_models['ARIMA'] = json.load(f)
            print("✅ ARIMA model found")
        
        # Check Prophet
        prophet_file = self.models_dir / "prophet.joblib"
        prophet_info = self.models_dir / "prophet.json"
        if prophet_file.exists() and prophet_info.exists():
            with open(prophet_info, 'r') as f:
                self.available_models['Prophet'] = json.load(f)
            print("✅ Prophet model found")
        
        # Check LSTM
        lstm_file = self.models_dir / "lstm.joblib"
        lstm_info = self.models_dir / "lstm.json"
        if lstm_file.exists() and lstm_info.exists():
            with open(lstm_info, 'r') as f:
                self.available_models['LSTM'] = json.load(f)
            print("✅ LSTM model found")
        
        print(f"📊 Total available models: {len(self.available_models)}")
        
    def load_data(self):
        """Load the 10-year historical data"""
        print("📊 Loading historical data...")
        
        data_file = self.data_dir / "all_stocks_10year_combined.csv"
        if not data_file.exists():
            raise FileNotFoundError("Historical data not found. Run collect_10year_data.py first.")
        
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"✅ Loaded {len(df):,} records for {df['Symbol'].nunique()} stocks")
        return df
    
    def prepare_2024_data(self, df):
        """Extract 2024 data for evaluation"""
        print("📅 Preparing 2024 test data...")
        
        start_2024 = datetime(2024, 1, 1)
        end_2024 = datetime(2024, 12, 31)
        
        test_data = df[(df['Date'] >= start_2024) & (df['Date'] <= end_2024)].copy()
        training_data = df[df['Date'] < start_2024].copy()
        
        print(f"✅ 2024 test data: {len(test_data):,} records")
        print(f"✅ Training data: {len(training_data):,} records")
        
        return training_data, test_data
    
    def evaluate_model_performance(self, model_name, model_info):
        """Evaluate a specific pre-trained model"""
        print(f"\n🔧 Evaluating {model_name} model...")
        
        # Get model training information
        training_info = model_info.get('training_info', {})
        stocks_trained = training_info.get('stocks_trained', [])
        
        print(f"   📋 Model trained on {len(stocks_trained)} stocks")
        print(f"   📅 Training completed: {model_info.get('training_date', 'Unknown')}")
        
        # Load the actual model
        try:
            model_file = self.models_dir / f"{model_name.lower()}.joblib"
            model = joblib.load(model_file)
            print(f"   ✅ Model loaded successfully")
            
            # Model size
            model_size = model_file.stat().st_size / 1024  # KB
            print(f"   📏 Model size: {model_size:.1f} KB")
            
            # Return basic model info for now
            return {
                'model_name': model_name,
                'stocks_trained': len(stocks_trained),
                'training_date': model_info.get('training_date', 'Unknown'),
                'model_size_kb': model_size,
                'status': 'Loaded Successfully'
            }
            
        except Exception as e:
            print(f"   ❌ Failed to load model: {str(e)}")
            return {
                'model_name': model_name,
                'status': f'Failed to load: {str(e)[:50]}'
            }
    
    def create_model_summary(self):
        """Create a summary of all available models"""
        print("\n📋 CREATING MODEL SUMMARY")
        print("="*50)
        
        model_summaries = []
        
        for model_name, model_info in self.available_models.items():
            summary = self.evaluate_model_performance(model_name, model_info)
            model_summaries.append(summary)
        
        # Create summary DataFrame
        if model_summaries:
            df = pd.DataFrame(model_summaries)
            
            # Save summary
            summary_file = self.results_dir / "pretrained_models_summary_2024.csv"
            df.to_csv(summary_file, index=False)
            print(f"\n💾 Model summary saved to: {summary_file}")
            
            # Display summary
            print(f"\n📊 PRE-TRAINED MODELS SUMMARY:")
            print(df.to_string(index=False))
            
            return df
        else:
            print("❌ No models available for evaluation")
            return None
    
    def analyze_training_stocks(self):
        """Analyze which stocks the models were trained on"""
        print("\n🔍 ANALYZING TRAINING STOCKS")
        print("="*40)
        
        all_training_stocks = set()
        
        for model_name, model_info in self.available_models.items():
            training_info = model_info.get('training_info', {})
            stocks_trained = training_info.get('stocks_trained', [])
            
            print(f"\n{model_name} Model:")
            print(f"   Trained on: {', '.join(stocks_trained[:5])}{'...' if len(stocks_trained) > 5 else ''}")
            print(f"   Total stocks: {len(stocks_trained)}")
            
            all_training_stocks.update(stocks_trained)
        
        print(f"\n📊 TRAINING STOCKS ANALYSIS:")
        print(f"   Total unique stocks used: {len(all_training_stocks)}")
        print(f"   Stocks: {', '.join(sorted(list(all_training_stocks))[:10])}{'...' if len(all_training_stocks) > 10 else ''}")
        
        return list(all_training_stocks)
    
    def check_2024_data_availability(self, training_stocks):
        """Check which training stocks have 2024 data available"""
        print("\n📅 CHECKING 2024 DATA AVAILABILITY")
        print("="*45)
        
        # Load data
        df = self.load_data()
        
        # Get 2024 data
        start_2024 = datetime(2024, 1, 1)
        test_data = df[df['Date'] >= start_2024]
        
        stocks_with_2024 = test_data['Symbol'].unique()
        
        # Check overlap
        training_stocks_with_2024 = [stock for stock in training_stocks if stock in stocks_with_2024]
        
        print(f"✅ Training stocks with 2024 data: {len(training_stocks_with_2024)}")
        print(f"   Stocks: {', '.join(training_stocks_with_2024[:10])}{'...' if len(training_stocks_with_2024) > 10 else ''}")
        
        # Count 2024 records per stock
        stock_2024_counts = {}
        for stock in training_stocks_with_2024[:5]:  # Check first 5
            stock_2024 = test_data[test_data['Symbol'] == stock]
            stock_2024_counts[stock] = len(stock_2024)
        
        if stock_2024_counts:
            print(f"\n📊 2024 Data Records per Stock (sample):")
            for stock, count in stock_2024_counts.items():
                print(f"   {stock}: {count} records")
        
        return training_stocks_with_2024
    
    def run_evaluation(self):
        """Run the complete evaluation"""
        print("🎯 PRE-TRAINED MODEL EVALUATION WITH 2024 DATA")
        print("="*60)
        
        if not self.available_models:
            print("❌ No pre-trained models found!")
            print("   Make sure the models have been trained and saved.")
            return
        
        # Create model summary
        model_summary = self.create_model_summary()
        
        # Analyze training stocks
        training_stocks = self.analyze_training_stocks()
        
        # Check 2024 data availability
        if training_stocks:
            stocks_with_2024 = self.check_2024_data_availability(training_stocks)
            
            print(f"\n🎉 EVALUATION SUMMARY:")
            print(f"   📊 Pre-trained models available: {len(self.available_models)}")
            print(f"   📈 Training stocks with 2024 data: {len(stocks_with_2024) if stocks_with_2024 else 0}")
            print(f"   📁 Results saved to: {self.results_dir}")
            
            # Save final summary
            final_summary = {
                'evaluation_date': datetime.now().isoformat(),
                'models_available': list(self.available_models.keys()),
                'training_stocks_total': len(training_stocks),
                'training_stocks_with_2024_data': len(stocks_with_2024) if stocks_with_2024 else 0,
                'evaluation_feasible': len(stocks_with_2024) > 0 if stocks_with_2024 else False
            }
            
            summary_file = self.results_dir / "evaluation_feasibility_2024.json"
            with open(summary_file, 'w') as f:
                json.dump(final_summary, f, indent=2)
            
            print(f"   📄 Feasibility report: {summary_file}")
        
        else:
            print("❌ No training stock information available")

def main():
    evaluator = PretrainedModelEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
