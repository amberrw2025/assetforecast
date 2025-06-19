# Setup Guide for macOS

This guide will help you set up the Forecast Accuracy Assessment Model Pipeline on macOS.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install minimal dependencies (recommended)
python3 -m pip install -r requirements_minimal.txt

# Or install all dependencies (optional)
python3 -m pip install -r requirements.txt
```

### 2. Test the Setup

```bash
# Run quick test to verify everything works
python3 quick_test.py
```

### 3. Run the Pipeline

```bash
# Run the complete pipeline
python3 run_pipeline.py

# Or run specific phases
python3 run_pipeline.py --data-only        # Data acquisition only
python3 run_pipeline.py --clean-only       # Data cleaning only
python3 run_pipeline.py --analysis-only    # Analysis only
```

## 🔧 Troubleshooting

### Issue: "pip command not found"
**Solution:** Use `python3 -m pip` instead of `pip`

```bash
# Instead of: pip install package_name
# Use: python3 -m pip install package_name
```

### Issue: Permission errors
**Solution:** Install packages for your user only

```bash
python3 -m pip install --user package_name
```

### Issue: SSL errors
**Solution:** Use trusted hosts

```bash
python3 -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org package_name
```

### Issue: Missing packages
**Solution:** Run the troubleshooting script

```bash
python3 troubleshoot.py
```

## 📦 What Gets Installed

### Essential Packages (Required)
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `yfinance` - Financial data
- `requests` - HTTP requests
- `plotly` - Data visualization
- `python-dotenv` - Environment variables
- `tqdm` - Progress bars
- `loguru` - Logging

### Optional Packages (Will use sample data if not available)
- `fredapi` - Economic data from FRED
- `pytrends` - Google Trends data
- `tweepy` - Twitter sentiment
- `praw` - Reddit sentiment
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `prophet` - Time series forecasting
- `statsmodels` - Statistical models
- `streamlit` - Web application

## 🎯 What the Pipeline Does

1. **Data Acquisition**: Collects financial, economic, and sentiment data
2. **Data Cleaning**: Handles missing values, outliers, and time alignment
3. **Data Integration**: Merges all datasets into a single clean dataset
4. **Analysis**: Creates visualizations and summary statistics
5. **Output**: Saves processed data and generates reports

## 📁 Output Files

After running the pipeline, you'll find:

- `data/raw/` - Raw collected data
- `data/processed/` - Cleaned and integrated data
- `data/processed/*.html` - Interactive visualizations
- `logs/pipeline.log` - Detailed execution logs

## 🔑 API Keys (Optional)

The pipeline works without API keys using sample data. To use real data:

1. Copy `env_example.txt` to `.env`
2. Edit `.env` with your API keys:
   - FRED API (free): https://fred.stlouisfed.org/docs/api/api_key.html
   - Twitter API: https://developer.twitter.com/
   - Reddit API: https://www.reddit.com/prefs/apps

## 🆘 Getting Help

If you encounter issues:

1. Run `python3 troubleshoot.py` for detailed diagnostics
2. Check the logs in `logs/pipeline.log`
3. Review the error messages for specific issues
4. Ensure you're using `python3` and `python3 -m pip` commands

## ✅ Success Indicators

You know the setup is working when:

- `python3 quick_test.py` shows "All tests passed"
- You can run `python3 run_pipeline.py` without errors
- The `data/` directory is created with subdirectories
- You see interactive HTML plots in `data/processed/`

---

**Note**: This pipeline is designed to work on macOS with Python 3.8+. If you're using a different operating system, the commands may vary slightly. 