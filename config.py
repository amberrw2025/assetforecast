"""
Configuration file for the Forecast Accuracy Assessment Model Pipeline.
Contains all settings, API keys, and parameters for data acquisition and processing.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys (load from environment variables)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "ForecastModel/1.0")

# Data sources configuration
DATA_SOURCES = {
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",  # Will be set to current date
    "frequency": "D",  # Daily frequency
    "currency": "USD",  # Base currency for standardization
}

# Company selection configuration
COMPANIES = {
    "ftse100": {
        "top_performers": [
            "AAL.L", "ABF.L", "ADM.L", "AHT.L", "ANTO.L"  # Placeholder tickers
        ],
        "bottom_performers": [
            "BARC.L", "BDEV.L", "BKG.L", "BLND.L", "BT-A.L"  # Placeholder tickers
        ]
    },
    "sp500": {
        "top_performers": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"  # Placeholder tickers
        ],
        "bottom_performers": [
            "XOM", "CVX", "KO", "PEP", "WMT"  # Placeholder tickers
        ]
    }
}

# Financial data configuration
FINANCIAL_METRICS = [
    "profit_margin",
    "revenue",
    "net_income", 
    "headcount",
    "total_assets",
    "total_liabilities",
    "operating_cash_flow",
    "free_cash_flow"
]

# Economic indicators configuration
ECONOMIC_INDICATORS = {
    "interest_rate_us": {
        "source": "FRED",
        "series_id": "FEDFUNDS",
        "description": "Federal Funds Rate"
    },
    "interest_rate_uk": {
        "source": "FRED", 
        "series_id": "IR3TIB01GBM156N",
        "description": "UK Base Rate"
    },
    "oil_price": {
        "source": "EIA",
        "series_id": "RBRTE",
        "description": "Brent Crude Oil Price"
    },
    "unemployment_us": {
        "source": "FRED",
        "series_id": "UNRATE", 
        "description": "US Unemployment Rate"
    }
}

# Data cleaning configuration
CLEANING_CONFIG = {
    "missing_value_threshold": 0.3,  # Remove columns with >30% missing values
    "interpolation_method": "linear",
    "forward_fill_limit": 5,
    "backward_fill_limit": 5,
    "outlier_threshold": 3.0,  # Standard deviations for outlier detection
}

# DVC configuration
DVC_CONFIG = {
    "remote_storage": "s3://your-bucket/data",  # Replace with actual storage
    "data_files": [
        "raw/company_financials.csv",
        "raw/economic_indicators.csv", 
        "raw/market_data.csv",
        "processed/merged_dataset.csv"
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "file": LOGS_DIR / "pipeline.log"
}

# Model configuration (for future use)
MODEL_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "random_state": 42,
    "cv_folds": 5,
    "forecast_horizon": 30,  # days
} 