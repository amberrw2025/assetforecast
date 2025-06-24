import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
import joblib

warnings.filterwarnings('ignore')

class MultivariateLSTMForecaster:
    """A class to handle training and prediction with a multivariate LSTM model."""
    def __init__(self, model_path, scaler_path):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

    def _build_model(self, input_shape):
        """Builds the LSTM model."""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fit(self, data, features, target_col='Close', sequence_length=60, epochs=50, batch_size=32):
        """Fits the model to the data."""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data[features])
        
        joblib.dump(self.scaler, self.scaler_path)

        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, features.index(target_col)])
        X, y = np.array(X), np.array(y)

        self.model = self._build_model(input_shape=(X.shape[1], X.shape[2]))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        self.model.save(self.model_path)

    def predict(self, data, features, sequence_length=60):
        """Makes predictions for the test data."""
        if self.model is None:
            self.model = load_model(self.model_path)
        if self.scaler is None:
            self.scaler = joblib.load(self.scaler_path)

        inputs = data[features].values
        inputs = self.scaler.transform(inputs)

        X_test = []
        for i in range(sequence_length, len(inputs)):
             X_test.append(inputs[i-sequence_length:i])
        X_test = np.array(X_test)

        predictions_scaled = self.model.predict(X_test)
        
        # To inverse transform, we need a dummy array with the right shape
        dummy_array = np.zeros((predictions_scaled.shape[0], self.scaler.n_features_in_))
        dummy_array[:, features.index('Close')] = predictions_scaled.flatten()
        
        return self.scaler.inverse_transform(dummy_array)[:, features.index('Close')]

class ForecastPlotter:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.plots_dir = self.project_root / "forecast_plots_2024"
        self.models_dir = self.project_root / "models_mv" # New dir for multivariate models
        self.plots_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        self.all_stocks = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN'] # Example stocks

    def load_and_prepare_data(self, symbol):
        """Loads and prepares the master dataset for a given stock."""
        print(f"📦 Loading and preparing data for {symbol}...")
        try:
            # 1. Load Base Historical Data
            hist_path = self.data_dir / "historical_10years" / f"{symbol}_10_years.csv"
            df = pd.read_csv(hist_path)
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

            # 2. Calculate Volatility
            df['Volatility'] = df['Close'].rolling(window=30).std() * np.sqrt(252)

            # 3. Load and Merge VIX Data
            vix_path = self.data_dir / "external" / "index_VIX.csv"
            vix_df = pd.read_csv(vix_path, parse_dates=['date']).rename(columns={'date': 'Date', 'close_price': 'VIX_Close'})
            df = pd.merge(df, vix_df[['Date', 'VIX_Close']], on='Date', how='left')
            
            # Forward-fill missing external data
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            df = df.dropna()
            
            print(f"✅ Data prepared for {symbol}. Records: {len(df)}")
            return df
        except FileNotFoundError as e:
            print(f"   ⚠️  Could not find data for {symbol}: {e}")
            return None

    def generate_comparison_plot(self, symbol, df, features):
        """Generates and saves a comparison plot for a single stock."""
        print(f"📈 Generating plot for {symbol}...")

        # Split data
        train_data = df[df['Date'] < datetime(2024, 1, 1)]
        test_data = df[df['Date'] >= datetime(2024, 1, 1)]

        if test_data.empty:
            print(f"   ⚠️  No 2024 data for {symbol}, skipping plot.")
            return

        # Initialize and fit model
        model_path = self.models_dir / f"{symbol}_mv_lstm.h5"
        scaler_path = self.models_dir / f"{symbol}_mv_scaler.joblib"
        forecaster = MultivariateLSTMForecaster(model_path, scaler_path)

        # For this example, we'll quickly fit the model. In a real scenario, you'd train this once.
        print(f"   Fitting new multivariate model for {symbol}...")
        forecaster.fit(train_data, features, target_col='Close', sequence_length=60, epochs=10, batch_size=32) # Fewer epochs for speed
        
        # Make predictions
        full_data_for_pred = pd.concat([train_data.tail(60), test_data])
        predicted_prices = forecaster.predict(full_data_for_pred, features, sequence_length=60)

        # Plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 7))

        ax.plot(test_data['Date'], test_data['Close'], label='Actual Price', color='black', marker='.', linestyle='-')
        ax.plot(test_data['Date'], predicted_prices, label='Multivariate LSTM Forecast', color='purple', linestyle='--')

        ax.set_title(f'2024 Multivariate Forecast vs. Actual for {symbol}', fontsize=16, weight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Closing Price (USD)', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        fig.autofmt_xdate()
        plot_path = self.plots_dir / f"{symbol}_2024_multivariate_forecast.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Plot saved to {plot_path}")

    def run(self):
        """Load data, prepare it, and generate plots for all stocks."""
        # Define features to use
        features = ['Close', 'Volume', 'Volatility', 'VIX_Close'] # Example feature set
        
        for symbol in self.all_stocks:
            df = self.load_and_prepare_data(symbol)
            if df is not None:
                self.generate_comparison_plot(symbol, df, features)
                
        print("\n🎉 All plots generated successfully!")
        print(f"📁 Find your plots in: {self.plots_dir}")

if __name__ == "__main__":
    plotter = ForecastPlotter()
    plotter.run()