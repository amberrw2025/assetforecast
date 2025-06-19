# Forecast Accuracy Assessment Model Pipeline

A comprehensive, modular Python pipeline for forecasting asset class performance and evaluating forecast accuracy over time. This project focuses on FTSE 100 and S&P 500 companies, utilizing both internal and external data sources to build robust forecasting models.

## 🎯 Project Overview

This pipeline implements the data acquisition, cleaning, and preparation phases of a forecast accuracy assessment model as outlined in the detailed project plan. The system is designed to:

- Collect financial data for selected companies using yfinance
- Gather macroeconomic indicators from FRED, EIA, and other sources
- Acquire sentiment data from social media platforms
- Clean, integrate, and prepare data for model training
- Provide comprehensive visualizations and analysis tools
- Support data versioning with DVC integration

## 📁 Project Structure

```
fste and S&P forcaster/
├── config.py                 # Configuration settings and parameters
├── main_pipeline.py          # Main orchestration script
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── dvc_integration.py       # DVC versioning integration
├── data_acquisition/        # Data collection modules
│   ├── __init__.py
│   ├── financial_data.py    # Company financial data collection
│   ├── economic_data.py     # Macroeconomic data collection
│   └── sentiment_data.py    # Social media sentiment collection
├── data_processing/         # Data cleaning and preprocessing
│   ├── __init__.py
│   └── data_cleaner.py      # Data cleaning and integration
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── logger.py           # Logging configuration
│   └── visualization.py    # Plotting and visualization tools
├── data/                   # Data storage (created automatically)
│   ├── raw/               # Raw collected data
│   └── processed/         # Cleaned and processed data
├── models/                # Model storage (for future use)
└── logs/                  # Log files (created automatically)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd "fste and S&P forcaster"

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root with your API keys:

```env
# FRED API (for economic data)
FRED_API_KEY=your_fred_api_key_here

# Twitter API (for sentiment data)
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret

# Reddit API (for sentiment data)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=ForecastModel/1.0
```

### 3. Run the Pipeline

```bash
# Run the complete pipeline
python main_pipeline.py
```

This will:
- Collect financial data for selected companies
- Gather economic indicators
- Acquire sentiment data
- Clean and integrate all datasets
- Generate visualizations and analysis
- Set up DVC tracking (optional)

## 📊 Data Sources

### Financial Data (yfinance)
- **Profit Margin**: Company profitability metrics
- **Revenue**: Total company revenue
- **Net Income**: Company net income
- **Headcount**: Employee count
- **Balance Sheet**: Assets and liabilities
- **Cash Flow**: Operating and free cash flow

### Economic Indicators (FRED)
- **Federal Funds Rate**: US interest rates
- **UK Base Rate**: UK interest rates
- **Unemployment Rate**: US employment data
- **Oil Prices**: Energy market data

### Sentiment Data
- **Twitter**: Social media sentiment analysis
- **Reddit**: Community sentiment from financial subreddits
- **Google Trends**: Search interest data

## 🔧 Configuration

The `config.py` file contains all configuration settings:

### Company Selection
```python
COMPANIES = {
    "ftse100": {
        "top_performers": ["AAL.L", "ABF.L", "ADM.L", "AHT.L", "ANTO.L"],
        "bottom_performers": ["BARC.L", "BDEV.L", "BKG.L", "BLND.L", "BT-A.L"]
    },
    "sp500": {
        "top_performers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "bottom_performers": ["XOM", "CVX", "KO", "PEP", "WMT"]
    }
}
```

### Data Cleaning Settings
```python
CLEANING_CONFIG = {
    "missing_value_threshold": 0.3,  # Remove columns with >30% missing values
    "interpolation_method": "linear",
    "outlier_threshold": 3.0,  # Standard deviations for outlier detection
}
```

## 📈 Output and Results

After running the pipeline, you'll find:

### Data Files
- `data/raw/company_financials.csv`: Raw financial data
- `data/raw/economic_indicators.csv`: Economic data
- `data/raw/sentiment_data.csv`: Sentiment data
- `data/processed/cleaned_dataset.csv`: Final integrated dataset

### Visualizations
- `data/processed/time_series_analysis.html`: Time series plots
- `data/processed/correlation_matrix.html`: Feature correlations
- `data/processed/missing_values_analysis.html`: Data quality analysis
- `data/processed/feature_distributions.html`: Distribution plots

### Analysis Reports
- `data/processed/summary_statistics.csv`: Statistical summaries
- `logs/pipeline.log`: Detailed execution logs

## 🔄 Data Versioning (DVC)

The pipeline includes DVC integration for data versioning:

```bash
# Initialize DVC (run automatically by pipeline)
dvc init

# Add remote storage (configure in dvc_integration.py)
dvc remote add default s3://your-bucket/data

# Track data files
dvc add data/raw/company_financials.csv
dvc add data/processed/cleaned_dataset.csv

# Push to remote storage
dvc push

# Pull from remote storage
dvc pull
```

## 🛠️ Customization

### Adding New Data Sources

1. Create a new collector class in `data_acquisition/`
2. Implement the required methods
3. Add configuration in `config.py`
4. Update the main pipeline

### Modifying Data Cleaning

1. Edit the `DataCleaner` class in `data_processing/data_cleaner.py`
2. Adjust cleaning parameters in `config.py`
3. Add new preprocessing steps as needed

### Custom Visualizations

1. Extend the `DataVisualizer` class in `utils/visualization.py`
2. Add new plotting methods
3. Integrate with the main pipeline

## 📋 API Requirements

### Required APIs
- **FRED API**: Free registration at https://fred.stlouisfed.org/
- **yfinance**: No API key required (uses Yahoo Finance)
- **Twitter API**: Developer account required
- **Reddit API**: Developer account required

### Optional APIs
- **EIA API**: For energy data (requires registration)
- **Google Trends**: No API key required (uses pytrends)

## 🐛 Troubleshooting

### Common Issues

1. **API Rate Limits**: The pipeline includes rate limiting, but you may need to adjust for your API limits
2. **Missing Data**: Some companies may not have complete data available
3. **DVC Setup**: DVC is optional and won't break the pipeline if not configured

### Debug Mode

Enable detailed logging by modifying the logging level in `config.py`:

```python
LOGGING_CONFIG = {
    "level": "DEBUG",  # Change from "INFO" to "DEBUG"
    # ... other settings
}
```

## 🔮 Next Steps

This pipeline covers the data acquisition and preparation phases. The next phases include:

1. **Model Training**: Implement forecasting models (Linear Regression, XGBoost, Prophet, etc.)
2. **Validation Pipeline**: Cross-validation and accuracy assessment
3. **Web Application**: Interactive dashboard using Streamlit or Dash
4. **Deployment**: Cloud deployment and monitoring

## 📄 License

This project is part of a forecast accuracy assessment model for financial analysis.

## 🤝 Contributing

1. Follow the modular structure
2. Add comprehensive logging
3. Include error handling
4. Update documentation
5. Test with sample data

## 📞 Support

For questions or issues:
1. Check the logs in `logs/pipeline.log`
2. Review the configuration in `config.py`
3. Ensure all dependencies are installed
4. Verify API keys are correctly configured

---

**Note**: This pipeline is designed for educational and research purposes. Always verify data accuracy and comply with relevant financial regulations when using for actual investment decisions. 