import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')

class ForecastPlotter:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data" / "historical_10years"
        self.plots_dir = self.project_root / "forecast_plots_2024"
        self.plots_dir.mkdir(exist_ok=True)

        self.ftse_stocks = ['AZN.L', 'LSEG.L', 'RKT.L', 'OCDO.L', 'CRH.L',
                           'BT-A.L', 'VOD.L', 'SSE.L', 'GLEN.L', 'TSCO.L']
        self.sp500_stocks = ['NVDA', 'TSLA', 'MRNA', 'ZM', 'NFLX',
                            'WBA', 'INTC', 'PARA', 'PAYC', 'F']
        self.all_stocks = self.ftse_stocks + self.sp500_stocks

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
        start_2024 = datetime(2024, 1, 1)
        training_data = df[df['Date'] < start_2024].copy()
        test_data = df[df['Date'] >= start_2024].copy()
        
        print(f"✅ Training data (2015-2023): {len(training_data):,} records")
        print(f"✅ Test data (2024): {len(test_data):,} records")
        return training_data, test_data

    def generate_comparison_plot(self, symbol, training_data, test_data):
        """Generates and saves a comparison plot for a single stock."""
        print(f"📈 Generating plot for {symbol}...")
        
        stock_train = training_data[training_data['Symbol'] == symbol].sort_values('Date')
        stock_test = test_data[test_data['Symbol'] == symbol].sort_values('Date')

        if stock_test.empty:
            print(f"   ⚠️  No 2024 data for {symbol}, skipping plot.")
            return

        # Use last 500 days of training data for context
        context_data = stock_train.tail(500)
        actual_prices = stock_test['Close'].values
        actual_dates = stock_test['Date']

        # Generate Polynomial forecast
        try:
            # Prepare data for polynomial regression
            X = np.arange(len(context_data)).reshape(-1, 1)
            y = context_data['Close'].values
            
            # Create polynomial features (degree 2 for a curve)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Predict future values
            future_X = np.arange(len(context_data), len(context_data) + len(stock_test)).reshape(-1, 1)
            future_X_poly = poly.transform(future_X)
            predicted_prices = model.predict(future_X_poly)

        except Exception as e:
            print(f"   ❌ Could not generate forecast for {symbol}: {e}")
            return

        # Plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 7))

        ax.plot(actual_dates, actual_prices, label='Actual 2024 Prices', color='royalblue', marker='.', linestyle='-', markersize=4)
        ax.plot(actual_dates, predicted_prices, label='Predicted 2024 Prices (Polynomial Trend)', color='darkorange', linestyle='--', marker='', markersize=4)

        ax.set_title(f'2024 Actual vs. Predicted Prices for {symbol}', fontsize=16, weight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Closing Price (USD)', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        fig.autofmt_xdate()
        
        plot_path = self.plots_dir / f"{symbol}_2024_forecast_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Plot saved to {plot_path}")

    def run(self):
        """Load data, prepare it, and generate plots for all stocks."""
        try:
            df = self.load_data()
            training_data, test_data = self.prepare_evaluation_data(df)
            
            for symbol in self.all_stocks:
                self.generate_comparison_plot(symbol, training_data, test_data)
                
            print("\n🎉 All plots generated successfully!")
            print(f"📁 Find your plots in: {self.plots_dir}")

        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            print("   Please ensure the 10-year data file exists.")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    plotter = ForecastPlotter()
    plotter.run() 