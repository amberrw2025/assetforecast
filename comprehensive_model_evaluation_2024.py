#!/usr/bin/env python3
"""
Comprehensive Model Evaluation with 2024 Data
Evaluates all available forecasting approaches using 2024 as test data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveModelEvaluator:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data" / "historical_10years"
        self.results_dir = self.project_root / "comprehensive_evaluation_2024"
        self.results_dir.mkdir(exist_ok=True)
        
        # Stock categorization
        self.ftse_stocks = ['AZN.L', 'LSEG.L', 'RKT.L', 'OCDO.L', 'CRH.L', 
                           'BT-A.L', 'VOD.L', 'SSE.L', 'GLEN.L', 'TSCO.L']
        self.sp500_stocks = ['NVDA', 'TSLA', 'MRNA', 'ZM', 'NFLX', 
                            'WBA', 'INTC', 'PARA', 'PAYC', 'F']
        
    def load_data(self):
        """Load the historical data"""
        print("📊 Loading 10-year historical data...")
        
        data_file = self.data_dir / "all_stocks_10year_combined.csv"
        if not data_file.exists():
            raise FileNotFoundError("Historical data not found. Run collect_10year_data.py first.")
        
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        
        print(f"✅ Loaded {len(df):,} records for {df['Symbol'].nunique()} stocks")
        return df
    
    def prepare_evaluation_data(self, df):
        """Prepare training and test data"""
        print("📅 Preparing evaluation datasets...")
        
        # Split: Use 2015-2023 for training, 2024 for testing
        start_2024 = datetime(2024, 1, 1)
        
        training_data = df[df['Date'] < start_2024].copy()
        test_data = df[df['Date'] >= start_2024].copy()
        
        print(f"✅ Training data (2015-2023): {len(training_data):,} records")
        print(f"✅ Test data (2024): {len(test_data):,} records")
        
        return training_data, test_data
    
    def calculate_comprehensive_metrics(self, actual, predicted, model_name="Model"):
        """Calculate comprehensive evaluation metrics"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return None
        
        # Basic metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        
        # Directional accuracy
        if len(actual) > 1:
            actual_direction = np.sign(np.diff(actual))
            predicted_direction = np.sign(np.diff(predicted))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            directional_accuracy = 0
        
        # Normalized RMSE (NRMSE)
        nrmse = rmse / (np.max(actual) - np.min(actual)) * 100 if np.max(actual) != np.min(actual) else 0
        
        # Mean Bias Error
        mbe = np.mean(predicted - actual)
        
        return {
            'model': model_name,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'nrmse': nrmse,
            'mbe': mbe,
            'directional_accuracy': directional_accuracy,
            'n_predictions': len(actual)
        }
    
    def evaluate_single_stock(self, symbol, training_data, test_data):
        """Evaluate multiple models for a single stock"""
        print(f"📈 Evaluating {symbol}...")
        
        # Get stock data
        stock_train = training_data[training_data['Symbol'] == symbol].sort_values('Date')
        stock_test = test_data[test_data['Symbol'] == symbol].sort_values('Date')
        
        if len(stock_train) < 100 or len(stock_test) < 50:
            print(f"   ⚠️  Insufficient data for {symbol}")
            return None
        
        # Use last 500 days of training data for context
        context_data = stock_train.tail(500)
        actual_prices = stock_test['Close'].values
        
        results = {}
        
        # 1. Linear Trend Model
        try:
            X = np.arange(len(context_data)).reshape(-1, 1)
            y = context_data['Close'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_X = np.arange(len(context_data), len(context_data) + len(stock_test)).reshape(-1, 1)
            linear_pred = model.predict(future_X)
            
            metrics = self.calculate_comprehensive_metrics(actual_prices, linear_pred, "Linear Trend")
            if metrics:
                results['Linear Trend'] = metrics
                print(f"   ✅ Linear Trend: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        except Exception as e:
            print(f"   ❌ Linear Trend failed: {str(e)[:40]}")
        
        # 2. Polynomial Trend (2nd degree)
        try:
            X = np.arange(len(context_data)).reshape(-1, 1)
            y = context_data['Close'].values
            
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            future_X = np.arange(len(context_data), len(context_data) + len(stock_test)).reshape(-1, 1)
            future_X_poly = poly_features.transform(future_X)
            poly_pred = model.predict(future_X_poly)
            
            metrics = self.calculate_comprehensive_metrics(actual_prices, poly_pred, "Polynomial Trend")
            if metrics:
                results['Polynomial Trend'] = metrics
                print(f"   ✅ Polynomial Trend: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        except Exception as e:
            print(f"   ❌ Polynomial Trend failed: {str(e)[:40]}")
        
        # 3. Moving Average (20-day)
        try:
            ma_20 = context_data['Close'].tail(20).mean()
            ma_pred = [ma_20] * len(stock_test)
            
            metrics = self.calculate_comprehensive_metrics(actual_prices, ma_pred, "Moving Average (20d)")
            if metrics:
                results['Moving Average (20d)'] = metrics
                print(f"   ✅ Moving Average: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        except Exception as e:
            print(f"   ❌ Moving Average failed: {str(e)[:40]}")
        
        # 4. Exponential Smoothing
        try:
            alpha = 0.3
            prices = context_data['Close'].values
            smoothed = [prices[0]]
            
            for i in range(1, len(prices)):
                smoothed.append(alpha * prices[i] + (1 - alpha) * smoothed[-1])
            
            exp_pred = [smoothed[-1]] * len(stock_test)
            
            metrics = self.calculate_comprehensive_metrics(actual_prices, exp_pred, "Exponential Smoothing")
            if metrics:
                results['Exponential Smoothing'] = metrics
                print(f"   ✅ Exponential Smoothing: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        except Exception as e:
            print(f"   ❌ Exponential Smoothing failed: {str(e)[:40]}")
        
        # 5. Naive (Last Price)
        try:
            last_price = context_data['Close'].iloc[-1]
            naive_pred = [last_price] * len(stock_test)
            
            metrics = self.calculate_comprehensive_metrics(actual_prices, naive_pred, "Naive (Last Price)")
            if metrics:
                results['Naive (Last Price)'] = metrics
                print(f"   ✅ Naive: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        except Exception as e:
            print(f"   ❌ Naive failed: {str(e)[:40]}")
        
        # 6. Trend + Seasonality (Weekly pattern)
        try:
            # Extract trend
            X = np.arange(len(context_data)).reshape(-1, 1)
            y = context_data['Close'].values
            trend_model = LinearRegression()
            trend_model.fit(X, y)
            
            # Calculate weekly seasonality
            context_data_copy = context_data.copy()
            context_data_copy['weekday'] = context_data_copy['Date'].dt.dayofweek
            weekly_pattern = context_data_copy.groupby('weekday')['Close'].mean()
            
            # Predict
            future_X = np.arange(len(context_data), len(context_data) + len(stock_test)).reshape(-1, 1)
            trend_pred = trend_model.predict(future_X)
            
            # Add seasonality
            test_weekdays = stock_test['Date'].dt.dayofweek
            seasonal_pred = []
            for i, weekday in enumerate(test_weekdays):
                seasonal_adjustment = weekly_pattern.get(weekday, 0) - context_data['Close'].mean()
                seasonal_pred.append(trend_pred[i] + seasonal_adjustment * 0.1)  # Dampened seasonality
            
            metrics = self.calculate_comprehensive_metrics(actual_prices, seasonal_pred, "Trend + Seasonality")
            if metrics:
                results['Trend + Seasonality'] = metrics
                print(f"   ✅ Trend + Seasonality: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        except Exception as e:
            print(f"   ❌ Trend + Seasonality failed: {str(e)[:40]}")
        
        return results
    
    def run_comprehensive_evaluation(self):
        """Run evaluation on all stocks"""
        print("🎯 COMPREHENSIVE MODEL EVALUATION WITH 2024 DATA")
        print("="*70)
        
        # Load data
        df = self.load_data()
        training_data, test_data = self.prepare_evaluation_data(df)
        
        # Evaluate all stocks
        all_results = {}
        
        print(f"\n📊 Evaluating all {len(df['Symbol'].unique())} stocks...")
        
        for symbol in sorted(df['Symbol'].unique()):
            stock_results = self.evaluate_single_stock(symbol, training_data, test_data)
            if stock_results:
                all_results[symbol] = stock_results
        
        # Compile and analyze results
        self.analyze_and_save_results(all_results)
        
        return all_results
    
    def analyze_and_save_results(self, all_results):
        """Analyze and save comprehensive results"""
        print(f"\n📋 ANALYZING COMPREHENSIVE RESULTS")
        print("="*60)
        
        # Create detailed results DataFrame
        detailed_data = []
        for symbol, models in all_results.items():
            market = "FTSE 100" if symbol in self.ftse_stocks else "S&P 500"
            for model_name, metrics in models.items():
                row = {
                    'Stock': symbol,
                    'Market': market,
                    'Model': model_name,
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'MAPE': metrics['mape'],
                    'NRMSE': metrics['nrmse'],
                    'R²': metrics['r2'],
                    'MBE': metrics['mbe'],
                    'Directional_Accuracy': metrics['directional_accuracy'],
                    'N_Predictions': metrics['n_predictions']
                }
                detailed_data.append(row)
        
        if not detailed_data:
            print("❌ No results to analyze")
            return
        
        results_df = pd.DataFrame(detailed_data)
        
        # Save detailed results
        detailed_file = self.results_dir / "comprehensive_evaluation_2024_detailed.csv"
        results_df.to_csv(detailed_file, index=False)
        print(f"💾 Detailed results saved: {detailed_file}")
        
        # Analysis by model
        print(f"\n📊 MODEL PERFORMANCE ANALYSIS:")
        print("-" * 50)
        
        model_performance = results_df.groupby('Model').agg({
            'RMSE': ['mean', 'std', 'min', 'max'],
            'MAE': ['mean', 'std'],
            'MAPE': ['mean', 'std'],
            'R²': ['mean', 'std'],
            'Directional_Accuracy': ['mean', 'std']
        }).round(3)
        
        print("Average Performance by Model:")
        print(model_performance.to_string())
        
        # Analysis by market
        print(f"\n📈 MARKET COMPARISON:")
        print("-" * 30)
        
        market_performance = results_df.groupby(['Market', 'Model'])['RMSE'].mean().unstack()
        print("Average RMSE by Market and Model:")
        print(market_performance.round(2).to_string())
        
        # Best models per stock
        print(f"\n🏆 BEST MODEL PER STOCK (by RMSE):")
        print("-" * 40)
        
        best_models = results_df.loc[results_df.groupby('Stock')['RMSE'].idxmin()]
        for _, row in best_models.iterrows():
            print(f"   {row['Stock']:8} | {row['Model']:20} | RMSE: {row['RMSE']:7.2f} | R²: {row['R²']:6.3f}")
        
        # Overall best models
        overall_best = results_df.groupby('Model')['RMSE'].mean().sort_values()
        print(f"\n🥇 OVERALL MODEL RANKING (by average RMSE):")
        for i, (model, rmse) in enumerate(overall_best.items(), 1):
            print(f"   {i}. {model:20} | Avg RMSE: {rmse:7.2f}")
        
        # Create comprehensive visualizations
        self.create_comprehensive_plots(results_df)
        
        # Save summary statistics
        summary_stats = {
            'evaluation_date': datetime.now().isoformat(),
            'test_period': '2024',
            'stocks_evaluated': len(all_results),
            'models_tested': results_df['Model'].nunique(),
            'ftse_stocks': len([s for s in all_results.keys() if s in self.ftse_stocks]),
            'sp500_stocks': len([s for s in all_results.keys() if s in self.sp500_stocks]),
            'best_overall_model': overall_best.index[0],
            'average_rmse_best_model': overall_best.iloc[0],
            'model_ranking': overall_best.to_dict()
        }
        
        summary_file = self.results_dir / "comprehensive_evaluation_summary_2024.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\n📄 Summary statistics: {summary_file}")
        print(f"📁 All results saved to: {self.results_dir}")
    
    def create_comprehensive_plots(self, results_df):
        """Create comprehensive visualization plots"""
        print("📈 Creating comprehensive evaluation plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model Performance Comparison (RMSE)
        ax1 = plt.subplot(3, 3, 1)
        model_rmse = results_df.groupby('Model')['RMSE'].mean().sort_values()
        model_rmse.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('Average RMSE by Model', fontweight='bold')
        ax1.set_xlabel('RMSE')
        
        # 2. R² Comparison
        ax2 = plt.subplot(3, 3, 2)
        model_r2 = results_df.groupby('Model')['R²'].mean().sort_values(ascending=False)
        model_r2.plot(kind='barh', ax=ax2, color='lightgreen')
        ax2.set_title('Average R² by Model', fontweight='bold')
        ax2.set_xlabel('R²')
        
        # 3. MAPE Comparison
        ax3 = plt.subplot(3, 3, 3)
        model_mape = results_df.groupby('Model')['MAPE'].mean().sort_values()
        model_mape.plot(kind='barh', ax=ax3, color='lightcoral')
        ax3.set_title('Average MAPE by Model', fontweight='bold')
        ax3.set_xlabel('MAPE (%)')
        
        # 4. Market Comparison
        ax4 = plt.subplot(3, 3, 4)
        market_comparison = results_df.groupby(['Market', 'Model'])['RMSE'].mean().unstack()
        market_comparison.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('RMSE by Market and Model', fontweight='bold')
        ax4.set_ylabel('RMSE')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)
        
        # 5. Directional Accuracy
        ax5 = plt.subplot(3, 3, 5)
        dir_acc = results_df.groupby('Model')['Directional_Accuracy'].mean().sort_values(ascending=False)
        dir_acc.plot(kind='barh', ax=ax5, color='orange')
        ax5.set_title('Average Directional Accuracy', fontweight='bold')
        ax5.set_xlabel('Directional Accuracy (%)')
        
        # 6. RMSE Distribution by Model
        ax6 = plt.subplot(3, 3, 6)
        models_for_box = results_df['Model'].unique()[:4]  # Top 4 models
        box_data = [results_df[results_df['Model'] == model]['RMSE'].values for model in models_for_box]
        ax6.boxplot(box_data, labels=[m[:15] for m in models_for_box])
        ax6.set_title('RMSE Distribution (Top 4 Models)', fontweight='bold')
        ax6.set_ylabel('RMSE')
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
        
        # 7. Performance by Stock (FTSE vs S&P)
        ax7 = plt.subplot(3, 3, 7)
        stock_performance = results_df.groupby('Stock')['RMSE'].mean().sort_values()
        colors = ['red' if stock in self.ftse_stocks else 'blue' for stock in stock_performance.index]
        stock_performance.plot(kind='bar', ax=ax7, color=colors)
        ax7.set_title('RMSE by Stock (Red=FTSE, Blue=S&P)', fontweight='bold')
        ax7.set_ylabel('Average RMSE')
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)
        
        # 8. R² vs RMSE Scatter
        ax8 = plt.subplot(3, 3, 8)
        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]
            ax8.scatter(model_data['RMSE'], model_data['R²'], label=model[:15], alpha=0.7)
        ax8.set_xlabel('RMSE')
        ax8.set_ylabel('R²')
        ax8.set_title('R² vs RMSE by Model', fontweight='bold')
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 9. Model Consistency (Std Dev of RMSE)
        ax9 = plt.subplot(3, 3, 9)
        model_consistency = results_df.groupby('Model')['RMSE'].std().sort_values()
        model_consistency.plot(kind='barh', ax=ax9, color='purple')
        ax9.set_title('Model Consistency (Lower = More Consistent)', fontweight='bold')
        ax9.set_xlabel('RMSE Standard Deviation')
        
        plt.tight_layout()
        
        # Save plots
        plot_file = self.results_dir / "comprehensive_evaluation_plots_2024.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comprehensive plots saved: {plot_file}")

def main():
    print("🎯 COMPREHENSIVE MODEL EVALUATION WITH 2024 DATA")
    print("="*70)
    print("⚡ Evaluating 6 different forecasting models on all 20 stocks")
    print("📅 Using 2015-2023 for training, 2024 for testing")
    print("="*70)
    
    evaluator = ComprehensiveModelEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    
    print(f"\n🎉 COMPREHENSIVE EVALUATION COMPLETE!")
    print(f"📊 Evaluated {len(results)} stocks with multiple models")
    print(f"📁 Results saved to: {evaluator.results_dir}")

if __name__ == "__main__":
    main() 