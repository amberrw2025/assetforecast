#!/usr/bin/env python3
"""
Unified Forecast Plotter
=======================

Creates consistent forecast plots for ALL official tickers using the same
EnhancedEnsembleModel (Prophet + ARIMA).

Output directory: forecast_plots_2024/unified
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import TimeSeriesSplit

from config import get_official_tickers
from models.prophet_model import ProphetModel
from models.arima_model import ARIMAModel
from models.enhanced_ensemble_model import EnhancedEnsembleModel

# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering applied equally to every stock."""
    enhanced = df.copy()

    # Volatility (20-day rolling std of pct change)
    enhanced["volatility_20"] = df["y"].pct_change().rolling(20).std()

    # Moving-average ratios
    for win in (5, 10, 20):
        ma = df["y"].rolling(win).mean()
        enhanced[f"ma_ratio_{win}"] = df["y"] / ma

    # Momentum
    enhanced["momentum_5"] = df["y"].pct_change(5)
    enhanced["momentum_10"] = df["y"].pct_change(10)

    # Calendar features
    enhanced["month"] = df["ds"].dt.month
    enhanced["day_of_week"] = df["ds"].dt.dayofweek

    # Fill NaNs introduced by rolling windows
    enhanced.fillna(method="ffill", inplace=True)
    enhanced.fillna(method="bfill", inplace=True)

    return enhanced


# --------------------------------------------------------------------------------------
# Forecast plotter class
# --------------------------------------------------------------------------------------

class UnifiedForecastPlotter:
    TRAIN_START = "2020-01-01"
    TRAIN_END = "2024-01-01"
    # Predict at most this many business days into the future
    MAX_FORECAST_DAYS = 60

    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        # Save inside the project regardless of current working dir
        self.output_dir = Path(__file__).resolve().parent / "forecast_plots_2024" / "unified"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def run(self) -> None:
        print("🚀 Generating unified forecast plots…")
        for tk in self.tickers:
            try:
                self._process_ticker(tk)
            except Exception as exc:
                print(f"❌ {tk}: {exc}")
        print("🎉 Done. Plots saved to", self.output_dir.resolve())

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _download_data(self, ticker: str):
        """Download historical OHLCV data and ensure single-level column names.

        yfinance >=0.2 returns a MultiIndex column structure when a single ticker
        string is passed (level-0 = field, level-1 = ticker).  This breaks
        downstream code that expects plain column names like ``Close``.  We
        therefore collapse any MultiIndex to its first level so that
        ``df['Close']`` behaves as before regardless of the yfinance version.
        """

        def _flatten(df: pd.DataFrame) -> pd.DataFrame:
            if isinstance(df.columns, pd.MultiIndex):
                # Drop the redundant ticker level – keep only 'Close', 'Open', ...
                df.columns = df.columns.get_level_values(0)
            return df

        train = yf.download(
            ticker,
            start=self.TRAIN_START,
            end=self.TRAIN_END,
            progress=False,
        )
        test = yf.download(
            ticker,
            start=self.TRAIN_END,
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False,
        )

        train = _flatten(train).reset_index()
        test = _flatten(test).reset_index()

        return train, test

    def _process_ticker(self, ticker: str):
        print(f"\n📈 {ticker}…", flush=True)
        train_df, test_df = self._download_data(ticker)
        if train_df.empty or test_df.empty:
            print("   ⚠️  Missing data – skipped.")
            return

        # ------------------- prepare log-returns -------------------
        prophet_df = train_df.rename(columns={"Date": "ds", "Close": "y"})[["ds", "y"]]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], utc=True).dt.tz_localize(None)
        prophet_df['y'] = np.log(prophet_df['y']).diff()
        prophet_df = prophet_df.dropna().reset_index(drop=True)

        # Guard-rail: need enough history to build features & forecast
        if len(prophet_df) < 60:   # about 3 months of returns
            print("⚠️  Not enough data … skipping.")
            return

        features_df = create_features(prophet_df)

        prophet_model = ProphetModel(
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode="multiplicative"
        )
        arima_model = ARIMAModel()
        # set required attrs for base models
        for m in (prophet_model, arima_model):
            m.date_column = "ds"
            m.target_column = "y"

        ensemble = EnhancedEnsembleModel(
            models=[prophet_model, arima_model],
            weighting_method="adaptive",
            performance_window=20,
            confidence_level=0.80  # Tighter bands → narrower CI
        )
        ensemble.date_column = "ds"
        ensemble.target_column = "y"
        ensemble.fit(features_df)

        # Limit horizon so forecasts don't get unrealistically flat
        horizon = min(len(test_df), self.MAX_FORECAST_DAYS)
        pred_returns, lower, upper = ensemble.predict_with_confidence(df=features_df, steps=horizon)

        # reconstruct price series from returns
        last_price = train_df['Close'].iloc[-1]
        preds_price = last_price * np.exp(np.cumsum(pred_returns))
        if lower is not None and upper is not None:
            lower = last_price * np.exp(np.cumsum(lower))
            upper = last_price * np.exp(np.cumsum(upper))
        preds = preds_price

        self._plot(ticker, train_df, test_df, preds, lower, upper)
        print("   ✅ Plot created.")

    def _plot(self, ticker: str, train_df: pd.DataFrame, test_df: pd.DataFrame, preds: np.ndarray, lower: np.ndarray | None, upper: np.ndarray | None):
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot only 2024 data onward (actual + forecast)
        jan_start = pd.to_datetime("2024-01-01")

        # Align lengths to the shortest series to avoid length-mismatch errors
        min_len = min(len(preds), len(test_df))
        if min_len == 0:
            print(f"   ⚠️  No overlap between forecast and actual data for {ticker} – skipping plot.")
            plt.close(fig)
            return

        dates_forecast = test_df["Date"].iloc[:min_len]
        actual_prices = test_df["Close"].iloc[:min_len]
        preds = preds[:min_len]
        if lower is not None and upper is not None:
            lower = lower[:min_len]
            upper = upper[:min_len]

        ax.plot(dates_forecast, actual_prices, label="Actual 2024", color="black", linewidth=2)
        ax.plot(dates_forecast, preds, label="Unified Forecast", color="#1f77b4", linestyle="--", linewidth=2)

        if lower is not None and upper is not None:
            ax.fill_between(dates_forecast, lower, upper, color="#1f77b4", alpha=0.2, label="CI 80%")

        # Mark forecast start (Jan-01-2024)
        ax.axvline(pd.to_datetime(self.TRAIN_END), color="red", linestyle=":", alpha=0.5, label="Forecast start")

        # Limit x-axis to 2024 onward
        ax.set_xlim(left=jan_start)

        market = "🇬🇧 FTSE 100" if ticker.endswith(".L") else "🇺🇸 S&P 500"
        ax.set_title(f"{market} | {ticker} – Unified Forecast vs Actual (2024)", fontsize=16, pad=20)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")

        save_path = self.output_dir / f"{ticker}_2024_unified_forecast.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    ticker_list = get_official_tickers()
    UnifiedForecastPlotter(ticker_list).run() 