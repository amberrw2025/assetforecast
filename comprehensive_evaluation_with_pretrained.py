#!/usr/bin/env python3
"""
Comprehensive Model Evaluation with Pre-trained Models
Evaluates both baseline and pre-trained models using 2024 data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing, AutoReg
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEvaluator:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data" / "historical_10years"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "comprehensive_evaluation_2024"
        self.results_dir.mkdir(exist_ok=True)
        
        self.ftse_stocks = ['AZN.L', 'LSEG.L', 'RKT.L', 'OCDO.L', 'CRH.L', 
                           'BT-A.L', 'VOD.L', 'SSE.L', 'GLEN.L', 'TSCO.L']
        self.sp500_stocks = ['NVDA', 'TSLA', 'MRNA', 'ZM', 'NFLX', 
                            'WBA', 'INTC', 'PARA', 'PAYC', 'F']
    
    def load_data(self):
        """Load historical data"""
        print("📊 Loading historical data...")
        
        data_file = self.data_dir / "all_stocks_10year_combined.csv"
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        
        # Split data
        start_2024 = datetime(2024, 1, 1)
        training_data = df[df['Date'] < start_2024]
        test_data = df[df['Date'] >= start_2024]
        
        print(f"✅ Training data: {len(training_data):,} records")
        print(f"✅ Test data (2024): {len(test_data):,} records")
        
        return training_data, test_data
    
    def load_pretrained_models(self):
        """Load pre-trained models"""
        print("\n🔧 Loading pre-trained models...")
        
        models = {}
        
        # ARIMA
        try:
            models['ARIMA'] = joblib.load(self.models_dir / "arima.joblib")
            print("✅ ARIMA model loaded")
        except Exception as e:
            print(f"❌ ARIMA failed: {e}")
        
        # Prophet
        try:
            models['Prophet'] = joblib.load(self.models_dir / "prophet.joblib")
            print("✅ Prophet model loaded")
        except Exception as e:
            print(f"❌ Prophet failed: {e}")
        
        # LSTM
        try:
            models['LSTM'] = joblib.load(self.models_dir / "lstm.joblib")
            print("✅ LSTM model loaded")
        except Exception as e:
            print(f"❌ LSTM failed: {e}")
        
        return models
    
    def evaluate_stock(self, symbol, training_data, test_data, pretrained_models):
        """Evaluate all models for a single stock"""
        print(f"📈 Evaluating {symbol}...")
        
        stock_train = training_data[training_data['Symbol'] == symbol].tail(200)
        stock_test = test_data[test_data['Symbol'] == symbol].head(50)
        
        if len(stock_train) < 50 or len(stock_test) < 20:
            return None
        
        actual = stock_test['Close'].values
        market = "FTSE 100" if symbol in self.ftse_stocks else "S&P 500"
        results = {}
        
        # 1. Linear Trend Baseline
        try:
            X = np.arange(len(stock_train)).reshape(-1, 1)
            y = stock_train['Close'].values
            model = LinearRegression()
            model.fit(X, y)
            
            future_X = np.arange(len(stock_train), len(stock_train) + len(stock_test)).reshape(-1, 1)
            pred = model.predict(future_X)
            
            rmse = np.sqrt(mean_squared_error(actual, pred))
            r2 = r2_score(actual, pred)
            
            results['Linear Trend'] = {'rmse': rmse, 'r2': r2, 'type': 'baseline'}
            print(f"   ✅ Linear Trend: RMSE={rmse:.2f}")
        except Exception as e:
            print(f"   ❌ Linear Trend failed: {str(e)[:30]}")
        
        # 2. Moving Average Baseline
        try:
            # Tune window on a validation set (last 50 days of training)
            ma_train_data = stock_train.iloc[:-50]
            ma_val_data = stock_train.iloc[-50:]
            best_window = 20
            best_rmse = float('inf')

            if len(ma_val_data) > 0:
                for window in [10, 20, 30, 50, 100]:
                    if len(ma_train_data) < window:
                        continue
                    
                    # Predict for validation set
                    ma_pred_val = [ma_train_data['Close'].tail(window).mean()] * len(ma_val_data)
                    rmse_val = np.sqrt(mean_squared_error(ma_val_data['Close'], ma_pred_val))
                    
                    if rmse_val < best_rmse:
                        best_rmse = rmse_val
                        best_window = window
            
            # Predict for the actual test set using the best window
            ma_pred = [stock_train['Close'].tail(best_window).mean()] * len(stock_test)
            rmse = np.sqrt(mean_squared_error(actual, ma_pred))
            r2 = r2_score(actual, ma_pred)
            
            results[f'Moving Average (w={best_window})'] = {'rmse': rmse, 'r2': r2, 'type': 'baseline'}
            print(f"   ✅ Moving Average (w={best_window}): RMSE={rmse:.2f}")
        except Exception as e:
            print(f"   ❌ Moving Average failed: {str(e)[:30]}")
        
        # 3. LSTM Model (if available)
        if 'LSTM' in pretrained_models:
            try:
                # Prepare LSTM input
                prices = stock_train['Close'].values
                sequence_length = 30
                
                if len(prices) >= sequence_length:
                    # Use last sequence for prediction
                    last_sequence = prices[-sequence_length:].reshape(1, sequence_length, 1)
                    
                    # Make predictions for each test point
                    lstm_predictions = []
                    current_sequence = last_sequence.copy()
                    
                    for _ in range(min(len(stock_test), 10)):  # Limit predictions
                        pred = pretrained_models['LSTM'].predict(current_sequence, verbose=0)[0][0]
                        lstm_predictions.append(pred)
                        
                        # Update sequence (simple approach)
                        current_sequence = np.roll(current_sequence, -1, axis=1)
                        current_sequence[0, -1, 0] = pred
                    
                    if len(lstm_predictions) > 0:
                        # Pad predictions if needed
                        while len(lstm_predictions) < len(actual):
                            lstm_predictions.append(lstm_predictions[-1])
                        
                        lstm_predictions = lstm_predictions[:len(actual)]
                        
                        rmse = np.sqrt(mean_squared_error(actual, lstm_predictions))
                        r2 = r2_score(actual, lstm_predictions)
                        
                        results['LSTM'] = {'rmse': rmse, 'r2': r2, 'type': 'pretrained'}
                        print(f"   ✅ LSTM: RMSE={rmse:.2f}")
                    else:
                        print(f"   ❌ LSTM: No predictions generated")
                else:
                    print(f"   ❌ LSTM: Insufficient data for sequence")
            except Exception as e:
                print(f"   ❌ LSTM failed: {str(e)[:30]}")
        
        # 4. Optimized Exponential Smoothing
        try:
            prices = stock_train['Close'].astype(float)
            # Fit the model. `fit` will automatically find the optimal alpha.
            ses_model = SimpleExpSmoothing(prices, initialization_method="estimated").fit()
            ses_pred = ses_model.forecast(len(stock_test))
            
            rmse = np.sqrt(mean_squared_error(actual, ses_pred))
            r2 = r2_score(actual, ses_pred)
            
            alpha_val = ses_model.params.get('smoothing_level', 'N/A')
            if isinstance(alpha_val, float):
                alpha_val = f"{alpha_val:.2f}"
            
            results[f'Exp. Smoothing (α={alpha_val})'] = {'rmse': rmse, 'r2': r2, 'type': 'baseline'}
            print(f"   ✅ Exp. Smoothing (α={alpha_val}): RMSE={rmse:.2f}")
        except Exception as e:
            print(f"   ❌ Exponential Smoothing failed: {str(e)[:30]}")
        
        # 5. Autoregressive (AR) Model
        try:
            # Select optimal lag 'p' using a validation set
            ar_train_data = stock_train.iloc[:-50]['Close'].astype(float)
            ar_val_data = stock_train.iloc[-50:]['Close'].astype(float)
            best_lag = 5
            best_rmse = float('inf')

            if len(ar_val_data) > 0:
                for lag in [1, 5, 10, 15, 25]:
                    if len(ar_train_data) < lag:
                        continue
                    try:
                        ar_model_val = AutoReg(ar_train_data, lags=lag, old_names=False).fit()
                        ar_pred_val = ar_model_val.predict(start=len(ar_train_data), end=len(ar_train_data) + len(ar_val_data) - 1, dynamic=False)
                        rmse_val = np.sqrt(mean_squared_error(ar_val_data, ar_pred_val))
                        if rmse_val < best_rmse:
                            best_rmse = rmse_val
                            best_lag = lag
                    except Exception:
                        continue
            
            prices = stock_train['Close'].astype(float)
            ar_model = AutoReg(prices, lags=best_lag, old_names=False).fit()
            ar_pred = ar_model.predict(start=len(prices), end=len(prices) + len(stock_test) - 1, dynamic=False)

            rmse = np.sqrt(mean_squared_error(actual, ar_pred))
            r2 = r2_score(actual, ar_pred)
            
            results[f'AR (lags={best_lag})'] = {'rmse': rmse, 'r2': r2, 'type': 'baseline'}
            print(f"   ✅ AR (lags={best_lag}): RMSE={rmse:.2f}")
        except Exception as e:
            print(f"   ❌ AR failed: {str(e)[:30]}")

        # 6. Holt-Winters Exponential Smoothing
        try:
            prices = stock_train['Close'].astype(float)
            seasonal_periods = 252

            if len(prices) >= 2 * seasonal_periods:
                hw_model = ExponentialSmoothing(prices, trend='mul', seasonal='mul', seasonal_periods=seasonal_periods).fit()
            else:
                hw_model = ExponentialSmoothing(prices, trend='mul', seasonal=None).fit()

            hw_pred = hw_model.forecast(len(stock_test))
            rmse = np.sqrt(mean_squared_error(actual, hw_pred))
            r2 = r2_score(actual, hw_pred)
            
            results['Holt-Winters'] = {'rmse': rmse, 'r2': r2, 'type': 'baseline'}
            print(f"   ✅ Holt-Winters: RMSE={rmse:.2f}")
        except Exception as e:
            print(f"   ❌ Holt-Winters failed: {str(e)[:30]}")
        
        return {'symbol': symbol, 'market': market, 'results': results}
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation"""
        print("🎯 COMPREHENSIVE EVALUATION WITH PRE-TRAINED MODELS")
        print("="*70)
        
        # Load data
        training_data, test_data = self.load_data()
        
        # Load pre-trained models
        pretrained_models = self.load_pretrained_models()
        
        # Evaluate all stocks
        all_results = []
        stocks = sorted(training_data['Symbol'].unique())
        
        print(f"\n📊 Evaluating {len(stocks)} stocks...")
        
        for symbol in stocks:
            stock_result = self.evaluate_stock(symbol, training_data, test_data, pretrained_models)
            if stock_result:
                all_results.append(stock_result)
        
        # Compile results
        self.compile_results(all_results)
        
        return all_results
    
    def compile_results(self, all_results):
        """Compile and analyze results"""
        print(f"\n📋 COMPILING COMPREHENSIVE RESULTS")
        print("="*50)
        
        # Create detailed results
        detailed_data = []
        
        for stock_result in all_results:
            symbol = stock_result['symbol']
            market = stock_result['market']
            
            for model_name, metrics in stock_result['results'].items():
                detailed_data.append({
                    'Stock': symbol,
                    'Market': market,
                    'Model': model_name,
                    'Type': metrics['type'],
                    'RMSE': metrics['rmse'],
                    'R²': metrics['r2']
                })
        
        if not detailed_data:
            print("❌ No results to compile")
            return
        
        results_df = pd.DataFrame(detailed_data)
        
        # Save results
        results_file = self.results_dir / "comprehensive_with_pretrained_2024.csv"
        results_df.to_csv(results_file, index=False)
        print(f"💾 Results saved: {results_file}")
        
        # Analysis
        print(f"\n📊 COMPREHENSIVE ANALYSIS:")
        print("-" * 40)
        
        # Model performance
        model_perf = results_df.groupby('Model').agg({
            'RMSE': ['mean', 'std', 'count'],
            'R²': ['mean', 'std']
        }).round(3)
        
        print("Model Performance Summary:")
        print(model_perf)
        
        # Best models by type
        print(f"\n BEST MODELS BY TYPE:")
        baseline_models = results_df[results_df['Type'] == 'baseline']
        pretrained_models = results_df[results_df['Type'] == 'pretrained']
        
        if not baseline_models.empty:
            best_baseline = baseline_models.loc[baseline_models['RMSE'].idxmin()]
            print(f"   Best Baseline: {best_baseline['Model']} (RMSE: {best_baseline['RMSE']:.2f})")
        
        if not pretrained_models.empty:
            best_pretrained = pretrained_models.loc[pretrained_models['RMSE'].idxmin()]
            print(f"   Best Pre-trained: {best_pretrained['Model']} (RMSE: {best_pretrained['RMSE']:.2f})")
        
        # Market comparison
        print(f"\n📈 MARKET COMPARISON:")
        market_perf = results_df.groupby(['Market', 'Model'])['RMSE'].mean().unstack()
        print("Average RMSE by Market and Model:")
        print(market_perf.round(2))
        
        # Create plots
        self.create_plots(results_df)
        
        print(f"\n🎉 COMPREHENSIVE EVALUATION COMPLETE!")
        print(f"📊 Evaluated {results_df['Stock'].nunique()} stocks")
        print(f"🔧 Tested {results_df['Model'].nunique()} models")
        print(f"📁 Results saved to: {self.results_dir}")
    
    def create_plots(self, results_df):
        """Create evaluation plots"""
        print("📈 Creating evaluation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comprehensive Model Evaluation - 2024 Data', fontsize=16, fontweight='bold')
        
        # 1. Model comparison
        ax1 = axes[0, 0]
        model_rmse = results_df.groupby('Model')['RMSE'].mean().sort_values()
        colors = ['red' if 'LSTM' in model else 'blue' for model in model_rmse.index]
        model_rmse.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Average RMSE by Model')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. R² comparison
        ax2 = axes[0, 1]
        model_r2 = results_df.groupby('Model')['R²'].mean().sort_values(ascending=False)
        model_r2.plot(kind='bar', ax=ax2, color='green')
        ax2.set_title('Average R² by Model')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Market performance
        ax3 = axes[1, 0]
        market_rmse = results_df.groupby('Market')['RMSE'].mean()
        market_rmse.plot(kind='bar', ax=ax3, color=['orange', 'purple'])
        ax3.set_title('Average RMSE by Market')
        ax3.set_ylabel('RMSE')
        ax3.tick_params(axis='x', rotation=0)
        
        # 4. Model type comparison
        ax4 = axes[1, 1]
        type_rmse = results_df.groupby('Type')['RMSE'].mean()
        type_rmse.plot(kind='bar', ax=ax4, color=['lightblue', 'lightcoral'])
        ax4.set_title('Average RMSE by Model Type')
        ax4.set_ylabel('RMSE')
        ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        
        plot_file = self.results_dir / "comprehensive_evaluation_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Plots saved: {plot_file}")

def main():
    evaluator = ComprehensiveEvaluator()
    evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()
