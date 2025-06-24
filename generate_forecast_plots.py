import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.enhanced_forecast_model import EnhancedForecastModel
from models.prophet_model import ProphetModel
from models.arima_model import ARIMAModel
from models.ensemble_model import EnsembleModel
import warnings

warnings.filterwarnings('ignore')

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
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data" / "historical_10years"
        self.plots_dir = self.project_root / "forecast_plots_2024"
        self.plots_dir.mkdir(exist_ok=True)
        self.enhanced_model = EnhancedForecastModel()

        self.ftse_stocks = ['AZN.L', 'LSEG.L', 'RKT.L', 'OCDO.L', 'CRH.L',
                           'BT-A.L', 'VOD.L', 'SSE.L', 'GLEN.L', 'TSCO.L']
        self.sp500_stocks = ['NVDA', 'TSLA', 'MRNA', 'ZM', 'NFLX',
                            'WBA', 'INTC', 'PARA', 'PAYC', 'F']
        self.all_tickers = self.ftse_stocks + self.sp500_stocks
        self.all_data = None

    def load_data(self):
        all_files = self.data_dir.glob("*.csv")
        df_list = []
        for f in all_files:
            try:
                df = pd.read_csv(f)
                # Standardize the ticker column
                if 'Symbol' in df.columns:
                    # Use 'Symbol' as the primary ticker identifier
                    if 'Ticker' in df.columns:
                        del df['Ticker'] # Drop incorrect 'Ticker' column if it exists
                    df.rename(columns={'Symbol': 'Ticker'}, inplace=True)

                if 'Ticker' in df.columns:
                    df_list.append(df)
                else:
                    print(f"   ⚠️ 'Ticker' column not found in {f.name}, skipping.")
            except Exception as e:
                print(f"   ❌ Error reading {f.name}: {e}")

        if not df_list:
            print("   ❌ No valid data files found.")
            return

        self.all_data = pd.concat(df_list, ignore_index=True)
        
        # Clean the 'Ticker' column to remove any leading/trailing whitespace
        if 'Ticker' in self.all_data.columns:
            self.all_data['Ticker'] = self.all_data['Ticker'].str.strip()
            
        self.all_data = self.all_data[self.all_data['Ticker'].isin(self.all_tickers)]
        # Use the robust cleaning function here
        self.all_data['Date'] = clean_date_column(self.all_data['Date'])
        self.all_data = self.all_data.dropna(subset=['Date'])
        
        print(f"✅ Loaded {len(self.all_data)} records for {len(self.all_tickers)} stocks")


    def run_for_stock(self, ticker):
        print(f"📈 Generating plot for {ticker}...")
        try:
            # Prepare data and set index
            stock_data = self.all_data[self.all_data['Ticker'] == ticker].copy()
            stock_data = stock_data.set_index('Date')

            # Split data
            hist_data = stock_data[stock_data.index < pd.to_datetime('2024-01-01')]
            test_data = stock_data[stock_data.index >= pd.to_datetime('2024-01-01')]

            if test_data.empty or len(hist_data) < 20:
                print(f"   ⚠️ Not enough data for {ticker}, skipping.")
                return

            # Prepare data for modeling
            train_df = hist_data.reset_index()[['Date', 'Close']]
            
            # --- Build the Ensemble Model ---
            
            # 1. Initialize Prophet Model
            prophet_model = ProphetModel()
            prophet_model.date_column = 'Date'
            prophet_model.target_column = 'Close'
            
            # 2. Initialize ARIMA Model
            arima_model = ARIMAModel(order=(5,1,0))
            arima_model.date_column = 'Date'
            arima_model.target_column = 'Close'
            
            # 3. Create, fit, and predict with Ensemble
            ensemble_model = EnsembleModel(models=[prophet_model, arima_model], weights=[0.5, 0.5])
            ensemble_model.fit(train_df)
            baseline_forecast = ensemble_model.predict(df=train_df, steps=len(test_data))

            # Fetch VIX data for the actual forecast period (2024)
            forecast_start = test_data.index.min()
            forecast_end = test_data.index.max()
            economic_data = self.enhanced_model.get_economic_indicators(
                start_date=forecast_start.strftime('%Y-%m-%d'),
                end_date=forecast_end.strftime('%Y-%m-%d'),
                ticker=ticker
            )

            # Generate the VIX-Simulated Path
            simulation_data = self.enhanced_model.generate_enhanced_forecast(
                ticker=ticker,
                hist_data=hist_data,
                forecast_dates=test_data.index,
                vix_forecast_period_data=economic_data
            )
            simulated_path = simulation_data.get('forecasts', {}).get('VIX_Simulated_Path')

            # Plotting
            plt.figure(figsize=(14, 8))

            # Ensure all plottable data are numpy arrays
            plot_test_data = test_data['Close'].values
            plot_baseline_forecast = np.array(baseline_forecast)
            
            plt.plot(test_data.index, plot_test_data, color='black', marker='.', linestyle='-', markersize=4, zorder=5)
            plt.plot(test_data.index, plot_baseline_forecast, color='royalblue', linestyle='--')
            
            legend_labels = ['Actual 2024 Prices', 'Baseline Forecast (Ensemble)']
            
            if simulated_path is not None and not pd.Series(simulated_path).empty:
                # Determine the correct label for the simulation
                sim_label = 'VFTSE-Simulated Path' if ticker.endswith('.L') else 'VIX-Simulated Path'
                legend_labels.append(sim_label)
                plot_simulated_path = np.array(simulated_path)
                plt.plot(test_data.index, plot_simulated_path, color='darkorange', linewidth=2)

            plt.title(f'2024 Forecast Simulation for {ticker}', fontsize=16)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend(legend_labels)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            plot_path = self.plots_dir / f"{ticker}_2024_forecast_simulation.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"   ✅ Plot saved to {plot_path}")

        except Exception as e:
            print(f"   ❌ Could not generate forecast for {ticker}: {e}")

    def run(self):
        print("📊 Loading 10-year historical data...")
        self.load_data()

        if self.all_data is None or self.all_data.empty:
            print("❌ No data loaded, cannot proceed.")
            return
        
        for ticker in self.all_tickers:
            self.run_for_stock(ticker)

        print("\n🎉 All plots generated successfully!")
        print(f"📁 Find your plots in: {self.plots_dir.resolve()}")

if __name__ == "__main__":
    plotter = ForecastPlotter()
    plotter.run()