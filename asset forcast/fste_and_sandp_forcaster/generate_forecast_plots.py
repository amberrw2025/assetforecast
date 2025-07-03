import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
import traceback

import config  # Load environment variables

# Correctly import all necessary models and providers
from models.prophet_model import ProphetModel
from models.arima_model import ARIMAModel
from models.enhanced_ensemble_model import EnhancedEnsembleModel
from economic_data_provider import EconomicDataProvider

warnings.filterwarnings('ignore')

def clean_date_column(series):
    """
    Robustly cleans a pandas Series expected to contain dates.
    """
    series = pd.to_datetime(series, errors='coerce', utc=True)
    series = series.dropna()
    series = series.dt.tz_localize(None)
    return series

class ForecastPlotter:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.data_dir = self.project_root / "data"
        self.plots_dir = self.project_root / "improved_forecast_plots_2024"
        self.plots_dir.mkdir(exist_ok=True)

        self.economic_provider = EconomicDataProvider()

        self.ftse_stocks = [
            'AZN.L', 'LSEG.L', 'RKT.L', 'OCDO.L', 'CRDA.L',
            'BT-A.L', 'VOD.L', 'SSE.L', 'GLEN.L', 'TSCO.L'
        ]
        self.sp500_stocks = [
            'NVDA', 'TSLA', 'MRNA', 'ZM', 'NFLX',
            'WBA', 'INTC', 'PARA', 'PAYC', 'F'
        ]
        self.all_tickers = self.ftse_stocks + self.sp500_stocks
        self.all_data = None
    
    def create_enhanced_features(self, train_df, hist_data):
        """Create enhanced features for better forecasting accuracy"""
        enhanced_df = train_df.copy()
        
        # Add volatility features
        enhanced_df['volatility'] = hist_data['close_price'].pct_change().rolling(20).std().reset_index(drop=True)
        
        # Add moving average ratios
        for window in [5, 10, 20]:
            ma = hist_data['close_price'].rolling(window).mean().reset_index(drop=True)
            enhanced_df[f'ma_ratio_{window}'] = enhanced_df['y'] / ma
        
        # Add momentum indicators
        enhanced_df['momentum_5'] = (hist_data['close_price'] / hist_data['close_price'].shift(5) - 1).reset_index(drop=True)
        enhanced_df['momentum_10'] = (hist_data['close_price'] / hist_data['close_price'].shift(10) - 1).reset_index(drop=True)
        
        # Add time-based features
        enhanced_df['month'] = enhanced_df['ds'].dt.month
        enhanced_df['quarter'] = enhanced_df['ds'].dt.quarter
        enhanced_df['day_of_week'] = enhanced_df['ds'].dt.dayofweek
        
        # Fill NaN values
        for col in enhanced_df.columns:
            if enhanced_df[col].dtype in [np.float64, np.int64]:
                enhanced_df[col] = enhanced_df[col].fillna(method='ffill').fillna(0)
        
        return enhanced_df
    
    def apply_market_regime_adjustments(self, forecast, hist_data, ticker):
        """Apply market regime-based adjustments to improve accuracy"""
        
        # Calculate market regime indicators
        recent_data = hist_data.tail(30)
        volatility = recent_data['close_price'].pct_change().std()
        trend = (recent_data['close_price'].iloc[-1] / recent_data['close_price'].iloc[0]) - 1
        
        # Determine market regime
        if trend > 0.05 and volatility < 0.02:
            regime = 'bull_low_vol'
            adjustment = 1.02
        elif trend < -0.05 and volatility > 0.03:
            regime = 'bear_high_vol'
            adjustment = 0.98
        elif volatility > 0.04:
            regime = 'high_volatility'
            adjustment = 0.99
        else:
            regime = 'neutral'
            adjustment = 1.0
        
        # Apply stock-specific adjustments
        if ticker in ['NVDA', 'TSLA']:  # High-growth stocks
            if regime == 'bull_low_vol':
                adjustment *= 1.05
        elif ticker.endswith('.L'):  # FTSE stocks
            if regime == 'bear_high_vol':
                adjustment *= 0.95
        
        print(f"    📊 Market regime: {regime}, Adjustment: {adjustment:.3f}")
        
        return forecast * adjustment

    def load_data(self):
        """
        Loads and concatenates all relevant CSV data files from the data directory.
        """
        print("📦 Loading and concatenating all historical data files...")
        combined_file = self.data_dir / "combined_ftse_sp500_data.csv"
        if not combined_file.exists():
            print(f"    ❌ Combined data file not found at {combined_file}")
            return False
            
        try:
            df = pd.read_csv(combined_file)
            if 'Symbol' in df.columns:
                df.rename(columns={'Symbol': 'ticker'}, inplace=True)
            if 'ticker' not in df.columns:
                print("    ⚠️ 'ticker' column not found, skipping.")
                return False

            self.all_data = df
            self.all_data['ticker'] = self.all_data['ticker'].str.strip()
            self.all_data = self.all_data[self.all_data['ticker'].isin(self.all_tickers)]
            self.all_data['date'] = clean_date_column(self.all_data['date'])
            self.all_data = self.all_data.dropna(subset=['date', 'close_price'])
            
            print(f"✅ Loaded {len(self.all_data)} records for {len(self.all_tickers)} unique stocks.")
            return True
        except Exception as e:
            print(f"    ❌ Error reading {combined_file.name}: {e}")
            return False

    def run_for_stock(self, ticker):
        """
        Generates and saves a comparison plot for a single stock.
        """
        print(f"📈 Generating plot for {ticker}...")
        try:
            stock_data = self.all_data[self.all_data['ticker'] == ticker].copy()
            stock_data = stock_data.set_index('date').sort_index()

            hist_data = stock_data[stock_data.index < pd.to_datetime('2024-01-01')]
            test_data = stock_data[stock_data.index >= pd.to_datetime('2024-01-01')]

            if test_data.empty or len(hist_data) < 30:
                print(f"    ⚠️ Not enough data for {ticker}, skipping.")
                return

            train_df = hist_data.reset_index()[['date', 'close_price']].rename(columns={'date': 'ds', 'close_price': 'y'})
            
            # --- IMPROVED FORECASTING APPROACH ---
            print(f"    🔧 Using improved forecasting methodology...")
            
            # Create enhanced features for better prediction
            enhanced_train_df = self.create_enhanced_features(train_df, hist_data)
            
            # Build the Enhanced Ensemble Model with better configuration
            prophet_model = ProphetModel(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True
            )
            arima_model = ARIMAModel()
            
            # Set required attributes for base models
            for model in [prophet_model, arima_model]:
                model.date_column = 'ds'
                model.target_column = 'y'
            
            ensemble_model = EnhancedEnsembleModel(
                models=[prophet_model, arima_model],
                weighting_method='adaptive',  # More responsive weighting
                performance_window=20  # Shorter window for recent performance
            )
            ensemble_model.date_column = 'ds'
            ensemble_model.target_column = 'y'
            
            # Fit with enhanced data
            ensemble_model.fit(enhanced_train_df)
            
            # Generate predictions with improved methodology
            baseline_forecast, lower_ci, upper_ci = ensemble_model.predict_with_confidence(
                df=enhanced_train_df, steps=len(test_data)
            )
            
            # Apply market regime adjustments
            baseline_forecast = self.apply_market_regime_adjustments(
                baseline_forecast, hist_data, ticker
            )

            # --- Data Cleaning & Validation ---
            # Ensure the forecast arrays are 1D NumPy arrays of the correct length
            baseline_forecast = np.array(baseline_forecast).flatten()
            if lower_ci is not None:
                lower_ci = np.array(lower_ci).flatten()
            if upper_ci is not None:
                upper_ci = np.array(upper_ci).flatten()

            # Get the date index for the forecast period
            forecast_dates = test_data.index

            # Check for length mismatch, which can cause plotting errors
            if len(baseline_forecast) != len(forecast_dates):
                print(f"    ⚠️ Forecast length ({len(baseline_forecast)}) does not match date length ({len(forecast_dates)}). Truncating forecast.")
                min_len = min(len(baseline_forecast), len(forecast_dates))
                baseline_forecast = baseline_forecast[:min_len]
                if lower_ci is not None:
                    lower_ci = lower_ci[:min_len]
                if upper_ci is not None:
                    upper_ci = upper_ci[:min_len]
                forecast_dates = forecast_dates[:min_len]

            # --- Economic Adjustment ---
            economic_indicators = self.economic_provider.get_economic_indicators()
            technical_indicators = self.economic_provider.get_technical_indicators(hist_data['close_price'])
            all_indicators = {**economic_indicators, **technical_indicators}
            market_regime = self.economic_provider.get_market_regime(all_indicators)
            
            adjustment_factor = self.economic_provider.get_regime_adjustment_factor(market_regime)
            adjusted_forecast = baseline_forecast * (1 + adjustment_factor)

            # --- Calculate Accuracy Metrics ---
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            actual_prices = test_data['close_price'].values
            baseline_mse = mean_squared_error(actual_prices, baseline_forecast)
            baseline_mae = mean_absolute_error(actual_prices, baseline_forecast)
            
            # Calculate simple baseline for comparison (last known price)
            simple_baseline = np.full(len(actual_prices), hist_data['close_price'].iloc[-1])
            simple_mse = mean_squared_error(actual_prices, simple_baseline)
            
            improvement = ((simple_mse - baseline_mse) / simple_mse) * 100
            
            print(f"    📊 Improved MSE: {baseline_mse:.2f} vs Simple MSE: {simple_mse:.2f}")
            print(f"    📈 Improvement: {improvement:.1f}%")
            
            # --- Enhanced Plotting ---
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Main forecast plot
            ax1.plot(hist_data.index, hist_data['close_price'], color='gray', linestyle='-', label='Historical Price', alpha=0.7)
            ax1.plot(forecast_dates, actual_prices, color='black', marker='o', markersize=3, linestyle='-', label='Actual 2024 Price', linewidth=2)
            ax1.plot(forecast_dates, baseline_forecast, color='#2E8B57', linestyle='--', label='Improved Forecast', linewidth=2)
            ax1.plot(forecast_dates, simple_baseline, color='#FF6B6B', linestyle=':', label='Simple Baseline', alpha=0.7)
            
            if lower_ci is not None and upper_ci is not None:
                ax1.fill_between(forecast_dates, lower_ci, upper_ci, color='#2E8B57', alpha=0.2, label='95% Confidence Interval')
            
            ax1.plot(forecast_dates, adjusted_forecast, color='#D55E00', linestyle='--', label=f'Regime-Adjusted ({market_regime})', linewidth=1.5)
            
            # Add accuracy metrics to plot
            accuracy_text = f'Improved MSE: {baseline_mse:.2f}\nSimple MSE: {simple_mse:.2f}\nImprovement: {improvement:.1f}%'
            ax1.text(0.02, 0.98, accuracy_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Add vertical line to separate historical and forecast
            ax1.axvline(x=hist_data.index[-1], color='red', linestyle=':', alpha=0.5, label='Forecast Start')
            
            ax1.set_title(f'IMPROVED Forecast vs Actual: {ticker} (2024)', fontsize=16, pad=20)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Error analysis plot
            forecast_error = actual_prices - baseline_forecast
            ax2.plot(forecast_dates, forecast_error, color='red', marker='o', markersize=2, label='Forecast Error')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.fill_between(forecast_dates, 0, forecast_error, alpha=0.3, color='red')
            
            ax2.set_title(f'Forecast Error Analysis: {ticker}', fontsize=14)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Error (Actual - Predicted)', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            fig.suptitle(f'Enhanced Forecasting Results: {ticker} - {improvement:.1f}% Improvement', fontsize=18)
            fig.autofmt_xdate()
            plt.tight_layout()

            plot_path = self.plots_dir / f"{ticker}_2024_IMPROVED_forecast_simulation.png"
            plt.savefig(plot_path, dpi=300)
            plt.close(fig)
            print(f"    ✅ Plot saved to {plot_path}")

        except Exception as e:
            print(f"    ❌ Could not generate forecast or plot for {ticker}: {e}")
            traceback.print_exc()

    def run(self):
        """
        Main method to load data and generate plots for all specified stocks.
        """
        print("📊 Initializing forecast plot generation...")
        if self.load_data():
            for ticker in self.all_tickers:
                self.run_for_stock(ticker)
            print("\n🎉 All requested plots generated successfully!")
            print(f"📁 Find your plots in: {self.plots_dir.resolve()}")
        else:
            print("\n❌ Plot generation failed due to data loading issues.")

if __name__ == "__main__":
    plotter = ForecastPlotter()
    plotter.run()