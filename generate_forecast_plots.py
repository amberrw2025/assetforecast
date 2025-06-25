import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime # Kept from Amber, useful for explicit date handling
from sklearn.preprocessing import MinMaxScaler # Kept for potential general scaling needs, though not used directly in this merged plot script
# from tensorflow.keras.models import Sequential, load_model # Removed as LSTM model is not the primary focus here
# from tensorflow.keras.layers import LSTM, Dense, Dropout # Removed as LSTM model is not the primary focus here

# Imports from the 'new' branch for advanced models
from models.enhanced_forecast_model import EnhancedForecastModel
from models.prophet_model import ProphetModel
from models.arima_model import ARIMAModel
from models.ensemble_model import EnsembleModel

import warnings
import joblib # Kept from Amber, useful for saving/loading general models/scalers

warnings.filterwarnings('ignore')

# --- Utility from 'new' branch: Robust Date Cleaning ---
def clean_date_column(series):
    """
    Robustly cleans a pandas Series expected to contain dates.
    - Converts to datetime, coercing errors to NaT.
    - Converts to UTC to handle mixed timezones, then removes timezone info.
    - Drops any rows that failed conversion.
    """
    series = pd.to_datetime(series, errors='coerce', utc=True)
    series = series.dropna()
    series = series.dt.tz_localize(None)
    return series

class ForecastPlotter:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent # Adjusted path to go up two levels for project root
        self.data_dir = self.project_root / "data"
        self.plots_dir = self.project_root / "forecast_plots_2024"
        # self.models_dir = self.project_root / "models_mv" # Not strictly needed if models are imported directly
        self.plots_dir.mkdir(exist_ok=True)
        # self.models_dir.mkdir(exist_ok=True) # Not strictly needed

        # Initialize the EnhancedForecastModel as it's used to fetch economic indicators and run simulations
        self.enhanced_model = EnhancedForecastModel()

        # Comprehensive list of stocks as per the project plan (5 top/5 bottom from each index)
        # These lists should ideally be dynamically loaded or managed based on the project's data acquisition
        self.ftse_stocks = [
            'AZN.L', 'LSEG.L', 'RKT.L', 'OCDO.L', 'CRH.L', # Example top/bottom from FTSE
            'BT-A.L', 'VOD.L', 'SSE.L', 'GLEN.L', 'TSCO.L' # Example top/bottom from FTSE
        ]
        self.sp500_stocks = [
            'NVDA', 'TSLA', 'MRNA', 'ZM', 'NFLX', # Example top/bottom from S&P 500
            'WBA', 'INTC', 'PARA', 'PAYC', 'F' # Example top/bottom from S&P 500
        ]
        self.all_tickers = self.ftse_stocks + self.sp500_stocks
        self.all_data = None # Will store the concatenated data from all CSVs

    def load_data(self):
        """
        Loads and concatenates all relevant CSV data files from the data directory.
        Filters data for the specified tickers and cleans the date column.
        """
        print("📦 Loading and concatenating all historical data files...")
        all_files = self.data_dir.glob("*.csv")
        df_list = []
        for f in all_files:
            try:
                df = pd.read_csv(f)
                # Standardize 'Symbol' to 'Ticker' for consistency
                if 'Symbol' in df.columns:
                    if 'Ticker' in df.columns: # Avoid duplicate columns if both exist
                        del df['Ticker']
                    df.rename(columns={'Symbol': 'Ticker'}, inplace=True)

                if 'Ticker' in df.columns:
                    df_list.append(df)
                else:
                    print(f"    ⚠️ 'Ticker' or 'Symbol' column not found in {f.name}, skipping.")
            except Exception as e:
                print(f"    ❌ Error reading {f.name}: {e}")

        if not df_list:
            print("    ❌ No valid data files found to load.")
            return

        self.all_data = pd.concat(df_list, ignore_index=True)
        
        # Clean the 'Ticker' column to remove any leading/trailing whitespace
        if 'Ticker' in self.all_data.columns:
            self.all_data['Ticker'] = self.all_data['Ticker'].str.strip()
            
        # Filter for only the tickers we are interested in for this project
        self.all_data = self.all_data[self.all_data['Ticker'].isin(self.all_tickers)]
        
        # Use the robust cleaning function for the Date column
        self.all_data['Date'] = clean_date_column(self.all_data['Date'])
        self.all_data = self.all_data.dropna(subset=['Date']) # Drop rows where Date could not be parsed
        
        print(f"✅ Loaded {len(self.all_data)} records for {len(self.all_tickers)} unique stocks.")

    def run_for_stock(self, ticker):
        """
        Generates and saves a comparison plot for a single stock,
        including actuals, baseline ensemble forecast, and VIX-simulated path.
        """
        print(f"📈 Generating plot for {ticker}...")
        try:
            # Prepare data and set index
            stock_data = self.all_data[self.all_data['Ticker'] == ticker].copy()
            stock_data = stock_data.set_index('Date').sort_index() # Ensure sorted by date

            # Split data into training (pre-2024) and test (2024 onward)
            hist_data = stock_data[stock_data.index < pd.to_datetime('2024-01-01')]
            test_data = stock_data[stock_data.index >= pd.to_datetime('2024-01-01')]

            # Check for sufficient data
            if test_data.empty or len(hist_data) < 20: # Need enough history for models
                print(f"    ⚠️ Not enough historical or test data for {ticker}, skipping plot.")
                return

            # Prepare data for modeling (Prophet expects 'ds' and 'y')
            train_df = hist_data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            # --- Build the Ensemble Model ---
            
            # 1. Initialize Prophet Model
            prophet_model = ProphetModel()
            prophet_model.date_column = 'ds'
            prophet_model.target_column = 'y'
            
            # 2. Initialize ARIMA Model
            arima_model = ARIMAModel(order=(5,1,0)) # Example ARIMA order
            arima_model.date_column = 'ds'
            arima_model.target_column = 'y'
            
            # 3. Create, fit, and predict with Ensemble Model
            ensemble_model = EnsembleModel(models=[prophet_model, arima_model], weights=[0.5, 0.5])
            ensemble_model.fit(train_df)
            
            # Predict for the test period (2024 onwards)
            baseline_forecast_df = ensemble_model.predict(df=train_df, periods=len(test_data))
            baseline_forecast = baseline_forecast_df['yhat'].values if 'yhat' in baseline_forecast_df.columns else baseline_forecast_df.values

            # Fetch relevant economic data for the actual forecast period (2024)
            forecast_start = test_data.index.min()
            forecast_end = test_data.index.max()
            economic_data = self.enhanced_model.get_economic_indicators(
                start_date=forecast_start.strftime('%Y-%m-%d'),
                end_date=forecast_end.strftime('%Y-%m-%d'),
                ticker=ticker # Pass ticker for specific VIX/VFTSE fetching
            )

            # Generate the VIX-Simulated Path using the EnhancedForecastModel
            simulation_data = self.enhanced_model.generate_enhanced_forecast(
                ticker=ticker,
                hist_data=hist_data, # Provide historical data for context
                forecast_dates=test_data.index, # Dates for which to simulate
                vix_forecast_period_data=economic_data # External economic data
            )
            simulated_path = simulation_data.get('forecasts', {}).get('VIX_Simulated_Path')

            # Plotting
            plt.style.use('seaborn-v0_8-whitegrid') # Use a good style
            fig, ax = plt.subplots(figsize=(14, 8))

            # Ensure all plottable data are numpy arrays for consistency
            plot_test_data = test_data['Close'].values
            plot_baseline_forecast = np.array(baseline_forecast)
            
            # Plot Actual Prices
            ax.plot(test_data.index, plot_test_data, color='black', marker='.', linestyle='-', markersize=4, zorder=5, label='Actual 2024 Prices')
            # Plot Baseline Forecast
            ax.plot(test_data.index, plot_baseline_forecast, color='royalblue', linestyle='--', label='Baseline Forecast (Ensemble)')
            
            # Plot VIX/VFTSE Simulated Path if available
            if simulated_path is not None and not pd.Series(simulated_path).empty:
                # Determine the correct label based on ticker's index
                sim_label = 'VFTSE-Simulated Path' if ticker.endswith('.L') else 'VIX-Simulated Path'
                plot_simulated_path = np.array(simulated_path)
                ax.plot(test_data.index, plot_simulated_path, color='darkorange', linewidth=2, label=sim_label)

            ax.set_title(f'2024 Forecast Simulation for {ticker}', fontsize=16, weight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            fig.autofmt_xdate() # Auto-format date labels for readability
            plt.tight_layout() # Adjust layout to prevent labels overlapping

            # Save the plot
            plot_path = self.plots_dir / f"{ticker}_2024_forecast_simulation.png"
            plt.savefig(plot_path, dpi=300) # Increased DPI for better quality
            plt.close(fig) # Close the figure to free memory
            print(f"    ✅ Plot saved to {plot_path}")

        except Exception as e:
            print(f"    ❌ Could not generate forecast or plot for {ticker}: {e}")
            traceback.print_exc() # Print full traceback for debugging

    def run(self):
        """
        Main method to load data and generate plots for all specified stocks.
        """
        print("📊 Initializing forecast plot generation...")
        self.load_data()

        if self.all_data is None or self.all_data.empty:
            print("❌ No data loaded. Cannot proceed with plot generation.")
            return
        
        # Iterate through all desired tickers and run the plotting logic
        for ticker in self.all_tickers:
            self.run_for_stock(ticker)

        print("\n🎉 All requested plots generated successfully!")
        print(f"📁 Find your plots in: {self.plots_dir.resolve()}")

if __name__ == "__main__":
    plotter = ForecastPlotter()
    plotter.run()