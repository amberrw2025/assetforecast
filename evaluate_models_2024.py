#!/usr/bin/env python3
"""
Model Evaluation Script for 2024 Data
Evaluates trained models using 2024 data as test set
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator2024:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data" / "historical_10years"
        self.results_dir = self.project_root / "evaluation_results_2024"
        self.results_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load the 10-year historical data"""
        print("📊 Loading historical data...")
        
        data_file = self.data_dir / "all_stocks_10year_combined.csv"
        if not data_file.exists():
            raise FileNotFoundError("Historical data not found. Run collect_10year_data.py first.")
        
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        
        print(f"✅ Loaded {len(df):,} records for {df['Symbol'].nunique()} stocks")
        return df
    
    def prepare_2024_data(self, df):
        """Extract 2024 data for evaluation"""
        print("📅 Preparing 2024 test data...")
        
        # Dates are already converted to timezone-naive in load_data()
        
        start_2024 = datetime(2024, 1, 1)
        end_2024 = datetime(2024, 12, 31)
        
        test_data = df[(df['Date'] >= start_2024) & (df['Date'] <= end_2024)].copy()
        training_data = df[df['Date'] < start_2024].copy()
        
        print(f"✅ 2024 test data: {len(test_data):,} records")
        print(f"✅ Training data: {len(training_data):,} records")
        
        return training_data, test_data
    
    def calculate_metrics(self, actual, predicted):
        """Calculate evaluation metrics"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return None
        
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        # MAPE
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        
        # Directional accuracy
        if len(actual) > 1:
            actual_direction = np.sign(np.diff(actual))
            predicted_direction = np.sign(np.diff(predicted))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            directional_accuracy = 0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def evaluate_baseline_models(self, training_data, test_data):
        """Evaluate baseline forecasting models"""
        print("\n🎯 EVALUATING BASELINE MODELS WITH 2024 DATA")
        print("="*60)
        
        results = {}
        
        # Get stocks with sufficient 2024 data
        stocks_2024 = []
        for symbol in test_data['Symbol'].unique():
            stock_test = test_data[test_data['Symbol'] == symbol]
            if len(stock_test) >= 50:  # At least 50 trading days
                stocks_2024.append(symbol)
        
        print(f"📊 Evaluating {len(stocks_2024)} stocks with sufficient 2024 data")
        
        for symbol in stocks_2024[:10]:  # Evaluate first 10 stocks
            print(f"\n📈 Evaluating {symbol}...")
            
            stock_train = training_data[training_data['Symbol'] == symbol].tail(200)
            stock_test = test_data[test_data['Symbol'] == symbol].head(50)
            
            if len(stock_train) < 50 or len(stock_test) < 20:
                continue
            
            actual_prices = stock_test['Close'].values
            stock_results = {}
            
            # 1. Linear Trend Model
            try:
                X = np.arange(len(stock_train)).reshape(-1, 1)
                y = stock_train['Close'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                future_X = np.arange(len(stock_train), len(stock_train) + len(stock_test)).reshape(-1, 1)
                linear_pred = model.predict(future_X)
                
                metrics = self.calculate_metrics(actual_prices, linear_pred)
                if metrics:
                    stock_results['Linear Trend'] = metrics
                    print(f"   ✅ Linear Trend: RMSE={metrics['rmse']:.2f}")
            except Exception as e:
                print(f"   ❌ Linear Trend failed: {str(e)[:30]}")
            
            # 2. Moving Average
            try:
                ma_pred = [stock_train['Close'].tail(20).mean()] * len(stock_test)
                metrics = self.calculate_metrics(actual_prices, ma_pred)
                if metrics:
                    stock_results['Moving Average'] = metrics
                    print(f"   ✅ Moving Average: RMSE={metrics['rmse']:.2f}")
            except Exception as e:
                print(f"   ❌ Moving Average failed: {str(e)[:30]}")
            
            # 3. Naive (Last Price)
            try:
                naive_pred = [stock_train['Close'].iloc[-1]] * len(stock_test)
                metrics = self.calculate_metrics(actual_prices, naive_pred)
                if metrics:
                    stock_results['Naive'] = metrics
                    print(f"   ✅ Naive: RMSE={metrics['rmse']:.2f}")
            except Exception as e:
                print(f"   ❌ Naive failed: {str(e)[:30]}")
            
            if stock_results:
                results[symbol] = stock_results
        
        return results
    
    def save_results(self, results):
        """Save evaluation results"""
        print(f"\n�� SAVING EVALUATION RESULTS")
        print("="*40)
        
        # Create summary DataFrame
        summary_data = []
        for symbol, models in results.items():
            for model_name, metrics in models.items():
                summary_data.append({
                    'Stock': symbol,
                    'Model': model_name,
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'MAPE': metrics['mape'],
                    'R²': metrics['r2'],
                    'Directional_Accuracy': metrics['directional_accuracy']
                })
        
        if not summary_data:
            print("❌ No results to save")
            return
        
        df = pd.DataFrame(summary_data)
        
        # Save detailed results
        results_file = self.results_dir / "evaluation_results_2024.csv"
        df.to_csv(results_file, index=False)
        print(f"💾 Results saved to: {results_file}")
        
        # Calculate averages
        avg_performance = df.groupby('Model').agg({
            'RMSE': 'mean',
            'MAE': 'mean',
            'MAPE': 'mean',
            'R²': 'mean',
            'Directional_Accuracy': 'mean'
        }).round(3)
        
        print(f"\n📊 AVERAGE PERFORMANCE BY MODEL:")
        print(avg_performance.to_string())
        
        # Best models
        best_rmse = df.loc[df['RMSE'].idxmin()]
        best_r2 = df.loc[df['R²'].idxmax()]
        
        print(f"\n🏆 BEST PERFORMING MODELS:")
        print(f"   Best RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.3f})")
        print(f"   Best R²: {best_r2['Model']} ({best_r2['R²']:.3f})")
        
        # Create simple plot
        self.create_plots(df)
        
        return df
    
    def create_plots(self, df):
        """Create evaluation plots"""
        print("📈 Creating evaluation plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # RMSE comparison
        rmse_avg = df.groupby('Model')['RMSE'].mean().sort_values()
        rmse_avg.plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Average RMSE by Model (2024 Test)')
        axes[0].set_ylabel('RMSE')
        
        # R² comparison
        r2_avg = df.groupby('Model')['R²'].mean().sort_values(ascending=False)
        r2_avg.plot(kind='bar', ax=axes[1], color='lightgreen')
        axes[1].set_title('Average R² by Model (2024 Test)')
        axes[1].set_ylabel('R²')
        
        plt.tight_layout()
        
        plot_file = self.results_dir / "evaluation_plots_2024.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Plots saved to: {plot_file}")
    
    def run_evaluation(self):
        """Run the complete evaluation"""
        print("🎯 MODEL EVALUATION WITH 2024 DATA")
        print("="*50)
        
        # Load data
        df = self.load_data()
        
        # Prepare 2024 test data
        training_data, test_data = self.prepare_2024_data(df)
        
        # Evaluate models
        results = self.evaluate_baseline_models(training_data, test_data)
        
        # Save results
        if results:
            self.save_results(results)
            print(f"\n🎉 Evaluation complete!")
            print(f"📁 Results saved to: {self.results_dir}")
        else:
            print("❌ No results generated")

def main():
    evaluator = ModelEvaluator2024()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
