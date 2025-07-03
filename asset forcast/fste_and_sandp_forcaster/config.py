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
FRED_API_KEY = os.getenv("FRED_API_KEY", "f57a50634dba5f945b6cfbecc034a755")
EIA_API_KEY = os.getenv("EIA_API_KEY", "")  # Optional - will skip if not provided
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")  # Optional - will skip if not provided
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")  # Optional - will skip if not provided
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")  # Optional - will skip if not provided
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")  # Optional - will skip if not provided

# Reddit API keys (optional)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "")

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
            "AZN.L", "SHEL.L", "BP.L", "RR.L", "VOD.L"  # Major FTSE 100 stocks
        ],
        "bottom_performers": [
            "AAL.L", "FRES.L", "STJ.L", "BATS.L", "ENT.L"  # 2023 worst performers still in FTSE 100
        ]
    },
    "sp500": {
        "top_performers": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"  # Top performers
        ],
        "bottom_performers": [
            "XOM", "CVX", "KO", "PEP", "WMT"  # Bottom performers
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
    # Interest Rates (Core indicators)
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
    
    # Yield Curve (Critical for financial sector predictions)
    "treasury_3m": {
        "source": "FRED",
        "series_id": "GS3M",
        "description": "3-Month Treasury Rate"
    },
    "treasury_2y": {
        "source": "FRED",
        "series_id": "GS2", 
        "description": "2-Year Treasury Rate"
    },
    "treasury_5y": {
        "source": "FRED",
        "series_id": "GS5",
        "description": "5-Year Treasury Rate"
    },
    "treasury_10y": {
        "source": "FRED",
        "series_id": "GS10",
        "description": "10-Year Treasury Rate"  
    },
    "treasury_30y": {
        "source": "FRED",
        "series_id": "GS30",
        "description": "30-Year Treasury Rate"
    },
    
    # Inflation (Major market driver)
    "inflation_us": {
        "source": "FRED",
        "series_id": "CPIAUCSL",
        "description": "US Consumer Price Index"
    },
    "inflation_uk": {
        "source": "FRED",
        "series_id": "GBRCPIALLMINMEI", 
        "description": "UK Consumer Price Index"
    },
    
    # Labor Market
    "unemployment_us": {
        "source": "FRED",
        "series_id": "UNRATE", 
        "description": "US Unemployment Rate"
    },
    "unemployment_uk": {
        "source": "FRED",
        "series_id": "LRHUTTTTGBM156S",
        "description": "UK Unemployment Rate"
    },
    
    # Economic Activity
    "consumer_confidence": {
        "source": "FRED",
        "series_id": "UMCSENT",
        "description": "US Consumer Sentiment"
    },
    "industrial_production": {
        "source": "FRED", 
        "series_id": "INDPRO",
        "description": "US Industrial Production Index"
    },
    
    # Credit Markets (Early recession indicators)
    "credit_spread": {
        "source": "FRED",
        "series_id": "BAMLC0A0CM",
        "description": "US Investment Grade Corporate Bond Spread"
    },
    "high_yield_spread": {
        "source": "FRED",
        "series_id": "BAMLH0A0HYM2",
        "description": "US High Yield Corporate Bond Spread"
    },
    
    # Currency (Critical for FTSE 100 stocks)
    "gbp_usd": {
        "source": "FRED",
        "series_id": "DEXUSUK",
        "description": "GBP/USD Exchange Rate"
    },
    "eur_usd": {
        "source": "FRED",
        "series_id": "DEXUSEU", 
        "description": "EUR/USD Exchange Rate"
    }
}

# Sector ETFs for relative performance analysis (Yahoo Finance)
SECTOR_ETFS = {
    "technology": {"ticker": "XLK", "description": "Technology Select Sector SPDR Fund"},
    "financials": {"ticker": "XLF", "description": "Financial Select Sector SPDR Fund"}, 
    "healthcare": {"ticker": "XLV", "description": "Health Care Select Sector SPDR Fund"},
    "energy": {"ticker": "XLE", "description": "Energy Select Sector SPDR Fund"},
    "utilities": {"ticker": "XLU", "description": "Utilities Select Sector SPDR Fund"},
    "consumer_discretionary": {"ticker": "XLY", "description": "Consumer Discretionary SPDR Fund"},
    "consumer_staples": {"ticker": "XLP", "description": "Consumer Staples SPDR Fund"},
    "industrials": {"ticker": "XLI", "description": "Industrial Select Sector SPDR Fund"},
    "materials": {"ticker": "XLB", "description": "Materials Select Sector SPDR Fund"},
    "real_estate": {"ticker": "XLRE", "description": "Real Estate Select Sector SPDR Fund"},
    "small_cap": {"ticker": "IWM", "description": "Russell 2000 ETF"},
    "emerging_markets": {"ticker": "EEM", "description": "iShares MSCI Emerging Markets ETF"}
}

# Volatility indices for market regime detection (Yahoo Finance)
VOLATILITY_INDICES = {
    "vix": {"ticker": "^VIX", "description": "CBOE Volatility Index (S&P 500)"},
    "vix9d": {"ticker": "^VIX9D", "description": "CBOE 9-Day Volatility Index"},
    "vxn": {"ticker": "^VXN", "description": "CBOE NASDAQ Volatility Index"},
    "rvx": {"ticker": "^RVX", "description": "CBOE Russell 2000 Volatility Index"},
    "move": {"ticker": "^MOVE", "description": "CBOE Move Index (Bond Volatility)"}
}

# Currency and commodity data (Yahoo Finance)
CURRENCY_COMMODITIES = {
    "dxy": {"ticker": "DX-Y.NYB", "description": "US Dollar Index"},
    "gbp_usd_spot": {"ticker": "GBPUSD=X", "description": "GBP/USD Spot Rate"},
    "eur_usd_spot": {"ticker": "EURUSD=X", "description": "EUR/USD Spot Rate"}, 
    "gold": {"ticker": "GC=F", "description": "Gold Futures"},
    "oil_wti": {"ticker": "CL=F", "description": "WTI Crude Oil Futures"},
    "oil_brent": {"ticker": "BZ=F", "description": "Brent Crude Oil Futures"},
    "copper": {"ticker": "HG=F", "description": "Copper Futures"},
    "silver": {"ticker": "SI=F", "description": "Silver Futures"}
}

# Bond market indicators (Yahoo Finance)
BOND_INDICATORS = {
    "treasury_long": {"ticker": "TLT", "description": "20+ Year Treasury Bond ETF"},
    "treasury_intermediate": {"ticker": "IEF", "description": "7-10 Year Treasury Bond ETF"},
    "treasury_short": {"ticker": "SHY", "description": "1-3 Year Treasury Bond ETF"},
    "investment_grade": {"ticker": "LQD", "description": "Investment Grade Corporate Bond ETF"},
    "high_yield": {"ticker": "HYG", "description": "High Yield Corporate Bond ETF"},
    "emerging_debt": {"ticker": "EMB", "description": "Emerging Markets Bond ETF"}
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

# Column names
TICKER_COLUMN = "ticker"

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

# ========================================================================================
# EVALUATION CONFIGURATION
# ========================================================================================

# Primary evaluation metric (changed from RMSE to MAPE for better stock forecasting)
PRIMARY_EVALUATION_METRIC = 'mape'  # Mean Absolute Percentage Error
SECONDARY_METRICS = ['rmse', 'mae', 'r2', 'directional_accuracy']

# Metric optimization direction (lower is better for MAPE, RMSE, MAE; higher is better for R²)
METRIC_OPTIMIZATION = {
    'mape': 'minimize',      # Primary metric - percentage error
    'rmse': 'minimize',      # Raw error magnitude  
    'mae': 'minimize',       # Mean absolute error
    'r2': 'maximize',        # Coefficient of determination
    'directional_accuracy': 'maximize'  # Trading direction accuracy
}

# Performance thresholds for different metrics
PERFORMANCE_THRESHOLDS = {
    'mape': {
        'excellent': 5.0,    # < 5% error
        'good': 10.0,        # < 10% error  
        'acceptable': 15.0,  # < 15% error
        'poor': 25.0         # > 25% error
    },
    'rmse': {
        'ftse_excellent': 50.0,   # < £50 for FTSE
        'ftse_acceptable': 100.0,  # < £100 for FTSE
        'sp500_excellent': 15.0,   # < $15 for S&P 500
        'sp500_acceptable': 30.0   # < $30 for S&P 500
    },
    'directional_accuracy': {
        'excellent': 65.0,    # > 65% direction accuracy
        'good': 55.0,         # > 55% direction accuracy
        'acceptable': 50.0    # > 50% direction accuracy (random)
    }
}

# Evaluation reporting preferences
EVALUATION_DISPLAY = {
    'primary_metric': PRIMARY_EVALUATION_METRIC,
    'precision': {
        'mape': 2,      # 2 decimal places for MAPE (e.g., 5.23%)
        'rmse': 2,      # 2 decimal places for RMSE
        'mae': 2,       # 2 decimal places for MAE
        'r2': 3,        # 3 decimal places for R² (e.g., 0.842)
        'directional_accuracy': 1  # 1 decimal place (e.g., 65.2%)
    },
    'sort_ascending': True,  # True for MAPE (lower is better), False for R² (higher is better)
    'include_rmse_for_comparison': True  # Keep RMSE for legacy comparison
}

def get_official_tickers():
    companies = COMPANIES['ftse100']['top_performers'] \
              + COMPANIES['ftse100']['bottom_performers'] \
              + COMPANIES['sp500']['top_performers'] \
              + COMPANIES['sp500']['bottom_performers']
    return list(dict.fromkeys(companies))   # preserve order, ensure uniqueness 