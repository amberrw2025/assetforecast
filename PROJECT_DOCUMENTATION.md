# Comprehensive Project Documentation

This document provides a detailed overview of the FTSE & S&P 500 Forecasting Pipeline, a project designed to predict stock performance using various data sources and machine learning models.

## 1. Project Overview

This project implements a full machine learning pipeline for time-series forecasting of stock prices for companies in the FTSE 100 and S&P 500 indices. The pipeline automates data collection, preprocessing, model training, evaluation, and provides a web application for visualizing results.

The key objectives are:
- To collect and integrate financial, economic, and sentiment data.
- To build and train predictive models (e.g., LSTM) for forecasting.
- To evaluate model performance using a comprehensive suite of metrics.
- To provide an easy-to-use web interface for interacting with the forecasts.
- To ensure reproducibility and scalability using tools like DVC.

## 2. Project Structure

The project is organized into the following directory structure:

```
fste_and_sandp_forcaster/
│
├── data/
│   ├── external/             # External data (e.g., economic indicators)
│   ├── internal/             # Internal data (e.g., financial fundamentals)
│   ├── processed/            # Processed and cleaned data for modeling
│   └── ...                   # Other raw data categorizations
│
├── data_acquisition/
│   ├── financial_data.py     # Scripts to collect company financial data
│   ├── economic_data.py      # Scripts to collect macroeconomic data
│   └── sentiment_data.py     # Scripts to collect sentiment data
│
├── data_processing/
│   └── data_cleaner.py       # Scripts for cleaning and preprocessing data
│
├── models/
│   ├── lstm_model.py         # Long Short-Term Memory (LSTM) model
│   ├── arima_model.py        # ARIMA model implementation
│   ├── prophet_model.py      # Facebook Prophet model implementation
│   └── base_model.py         # Base model class
│
├── webapp/
│   ├── app.py                # Main Flask application file
│   └── templates/            # HTML templates for the web interface
│
├── comprehensive_evaluation_2024/ # Evaluation results and plots
├── evaluation_results_2024/       # More evaluation results
│
├── main_pipeline.py          # Main orchestrator for the entire pipeline
├── model_training_pipeline.py # Script to train the forecasting models
├── evaluate_models_2024.py   # Script to evaluate model performance
├── run_webapp.py             # Script to launch the web application
├── config.py                 # Project-wide configuration settings
├── requirements.txt          # Python dependencies
└── README.md                 # Original project README
```

This structure separates concerns, making the project modular and easier to maintain.

## 3. Core Components

This project is divided into several core components, each responsible for a specific part of the pipeline.

### 3.1. Data Acquisition

The pipeline collects data from a wide range of sources to build a comprehensive dataset for forecasting. The data acquisition logic is primarily located in the `data_acquisition/` directory.

#### Data Sources:

*   **Financial Data**:
    *   **Source**: `yfinance` library (Yahoo Finance).
    *   **Scripts**: `data_acquisition/financial_data.py`, `download_ftse_sp500_data.py`.
    *   **Data**: Historical stock prices (OHLCV), trading volume, and company fundamental data (e.g., revenue, net income, market cap).

*   **Economic Data**:
    *   **Source**: Federal Reserve Economic Data (FRED), Energy Information Administration (EIA), Google Trends.
    *   **Script**: `data_acquisition/economic_data.py`.
    *   **Data**: Macroeconomic indicators such as interest rates, unemployment rates, and oil prices, as well as search interest trends for relevant keywords.

*   **Sentiment Data**:
    *   **Source**: Twitter and Reddit APIs.
    *   **Script**: `data_acquisition/sentiment_data.py`.
    *   **Data**: Social media posts related to specific companies or market trends. The text data is further processed to calculate sentiment scores.

#### Key Features:

*   **Modularity**: Each data source is handled by a dedicated module, making it easy to add new sources.
*   **Resilience**: The scripts include error handling and can generate sample data if API credentials are not provided, allowing the pipeline to run in a test mode.
*   **Configuration**: The list of companies, keywords, and other parameters are managed in the `config.py` file.
*   **Storage**: All collected raw data is stored in the `data/` directory, organized into subdirectories like `internal/`, `external/`, and `sentiment/`.

### 3.2. Data Processing

Once the data is collected, it undergoes a thorough cleaning and preprocessing phase to prepare it for modeling. This is managed by the `DataCleaner` class in `data_processing/data_cleaner.py`.

The key steps in this process are:

*   **Missing Value Imputation**:
    *   Columns with a high percentage of missing data (configurable threshold) are dropped.
    *   Remaining missing values in numerical columns are handled using a combination of forward/backward fill, interpolation, and median imputation.
    *   Missing categorical values are filled with a placeholder like "Unknown".

*   **Outlier Handling**:
    *   Outliers in numerical columns are detected using the z-score method.
    *   Identified outliers are capped at a configurable standard deviation threshold to reduce their impact.

*   **Time Series Standardization**:
    *   All time-series data is resampled to a consistent frequency (e.g., daily) to ensure alignment between different data sources.
    *   This step involves averaging values within each resampling period.

*   **Currency Conversion**:
    *   Monetary values from different markets (e.g., GBP for FTSE, USD for S&P 500) are converted to a single base currency (USD) for consistency.

*   **Feature Engineering**:
    *   New features are created to enhance the predictive power of the models. These include:
        *   **Moving Averages**: (e.g., 7-day, 30-day) of prices and other numerical features.
        *   **Lag Features**: Historical values of key metrics.
        *   **Rolling Statistics**: Standard deviation and other volatility measures.
        *   **Date-based Features**: Day of the week, month, year, etc.

*   **Data Integration**:
    *   The cleaned financial, economic, and sentiment datasets are merged into a single, unified DataFrame, which serves as the input for the model training pipeline.

The entire process is highly configurable through the `CLEANING_CONFIG` dictionary in the `config.py` file. The final processed dataset is saved to the `data/processed/` directory.

### 3.3. Model Training

The project includes a robust pipeline for training and evaluating multiple forecasting models. The core logic is in `model_training_pipeline.py`, and the model implementations are in the `models/` directory.

#### Model Architecture:

A `BaseForecastModel` abstract class (`models/base_model.py`) provides a standardized interface for all models, ensuring consistency in how they are trained, evaluated, and used.

The following models are implemented:

*   **ARIMA**: A classical statistical model for time series forecasting (`models/arima_model.py`).
*   **Prophet**: A procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. Developed by Facebook (`models/prophet_model.py`).
*   **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) well-suited to time series forecasting (`models/lstm_model.py`). Implemented using TensorFlow/Keras.
*   **Ensemble Model**: A model that combines the predictions of the ARIMA, Prophet, and LSTM models, using a weighted average to potentially improve accuracy (`models/ensemble_model.py`).

#### Training Pipeline:

The `ModelTrainingPipeline` class in `model_training_pipeline.py` orchestrates the end-to-end training process:

1.  **Data Loading and Preparation**: The cleaned dataset is loaded from `data/processed/`, and split into training and testing sets based on a chronological split.
2.  **Model Initialization**: Instances of all forecasting models are created with parameters defined in the pipeline script.
3.  **Model Training**: Each model is trained on the historical training data. The `fit` method of each model is called.
4.  **Evaluation**: After training, the models are evaluated on the held-out test set. A `ModelEvaluator` class calculates key performance metrics (e.g., MAE, RMSE, MAPE, R^2).
5.  **Model Saving**: The trained models are serialized and saved to the `models/` directory, allowing them to be loaded later for inference without retraining.

This structured approach allows for easy comparison between different models and facilitates experimentation with new model architectures.

### 3.4. Model Evaluation

A systematic evaluation process is in place to assess the performance of the trained models. This is primarily handled by the `ModelEvaluator` class (`models/model_evaluator.py`), which is integrated into the `ModelTrainingPipeline`.

#### Key Evaluation Features:

*   **Comprehensive Metrics**: The evaluator calculates a suite of standard regression and forecasting metrics, including:
    *   Mean Absolute Error (MAE)
    *   Mean Squared Error (MSE)
    *   Root Mean Squared Error (RMSE)
    *   Mean Absolute Percentage Error (MAPE)
    *   R-squared (R²)

*   **Model Comparison**: It provides functions to compare all trained models based on a selected metric, making it easy to identify the best-performing model for a given task.

*   **Visualization**: The evaluator can generate plots to visually inspect the results:
    *   **Comparison Plots**: Bar charts comparing the performance metrics of different models.
    *   **Prediction Plots**: Time series plots that overlay the predictions of each model against the actual historical data, providing a qualitative assessment of the model's accuracy.

*   **Reporting**: A detailed text-based report can be generated, summarizing the performance of all models and providing a clear overview of the evaluation results.

*   **Cross-Validation**: The `ModelEvaluator` also includes functionality for cross-validation, offering a more robust way to estimate model performance and reduce the risk of overfitting.

#### Specialized Evaluation Scripts:

In addition to the integrated evaluation pipeline, the project contains several standalone scripts for more specific evaluation scenarios (e.g., `evaluate_models_2024.py`, `comprehensive_model_evaluation_2024.py`). These scripts are used for ad-hoc analyses, such as evaluating model performance on a specific year's data or against pre-trained models.

### 3.5. Web Application

The project includes an interactive web application for visualizing data, generating forecasts, and comparing models. The application is built with Flask and is located in the `webapp/` directory.

#### Key Features:

*   **On-Demand Forecasting**: Users can input a stock ticker (e.g., AAPL, GOOGL) and receive a forecast for the next 30 days (or a configurable period).
*   **Multi-Model Comparison**: The interface displays forecasts generated by multiple models (ARIMA, Prophet, and an Ensemble model), allowing for a visual comparison of their predictions.
*   **Interactive Charts**: The historical data and forecasts are displayed on an interactive chart, likely using a library like Plotly or Chart.js.
*   **API-Driven**: The frontend communicates with the Flask backend via a RESTful API. The main endpoint is `/api/forecast-asset`, which handles the forecasting logic.
*   **Caching**: The application uses a caching mechanism to store data from external sources (like `yfinance`), which improves performance by reducing redundant API calls.

#### How to Run:

The web application can be started by running the `run_webapp.py` script from the project's root directory:

```bash
python run_webapp.py
```

This will launch a local development server, and the application will be accessible at `http://localhost:5001`.

The web application serves as the primary user interface for the project, making the complex forecasting pipeline accessible to non-technical users.

## 4. Main Pipeline

The project is designed around a clear, sequential workflow, with dedicated scripts for each major stage. The `main_pipeline.py` script is the starting point, responsible for orchestrating the entire data engineering process.

#### Data Engineering Pipeline (`main_pipeline.py`):

The `ForecastModelPipeline` class within this script automates all the steps required to get the data ready for modeling:

1.  **Parallel Data Acquisition**: It kicks off the data collection process by running the `FinancialDataCollector`, `EconomicDataCollector`, and `SentimentDataCollector` simultaneously using multi-threading. This efficiently gathers all the necessary raw data.
2.  **Data Cleaning and Preprocessing**: Once the raw data is collected, it's passed to the `DataCleaner`. This stage runs in parallel using multi-processing to handle the computationally intensive tasks of cleaning, standardization, and feature engineering.
3.  **Data Integration**: The various cleaned datasets are merged into a single, cohesive dataset.
4.  **Exploratory Data Analysis (EDA)**: The pipeline generates a series of visualizations (e.g., time series plots, correlation matrices) to provide insights into the cleaned data.
5.  **Data Saving**: The raw and processed datasets are saved to the `data/raw/` and `data/processed/` directories, respectively.

#### Overall Project Workflow:

The project is intended to be run in the following sequence:

1.  **Run the Data Pipeline**:
    ```bash
    python main_pipeline.py
    ```
    This command populates the `data/` directory with clean, model-ready data.

2.  **Run the Model Training Pipeline**:
    ```bash
    python model_training_pipeline.py
    ```
    This script loads the processed data, trains all the forecasting models, evaluates them, and saves the trained artifacts to the `models/` directory.

3.  **Launch the Web Application**:
    ```bash
    python run_webapp.py
    ```
    This starts the Flask web application, which loads the saved models and allows users to generate new forecasts through an interactive interface.

This separation of concerns makes the project modular and easy to manage.

## 5. Configuration

The project's behavior is controlled by a central configuration file, `config.py`, which allows for easy modification of parameters without altering the core codebase.

#### The `config.py` file:

This file is organized into several sections:

*   **Paths**: Defines the directory structure for data, models, and logs.
*   **API Keys**: Specifies the environment variables that hold the API keys for services like FRED, Twitter, and Reddit.
*   **Data Sources**: Sets the global parameters for data collection, such as the start and end dates, data frequency (e.g., daily), and the base currency for standardization.
*   **Company Selection**: A dictionary (`COMPANIES`) that lists the stock tickers for the FTSE 100 and S&P 500 companies to be analyzed.
*   **Economic Indicators**: A dictionary defining the specific macroeconomic indicators to be downloaded from sources like FRED and EIA.
*   **Data Cleaning**: A `CLEANING_CONFIG` dictionary that controls the behavior of the `DataCleaner`, including thresholds for missing values and outliers.
*   **Logging**: Configures the logging level and format for the entire application.

#### Environment Variables (`.env` file):

To use the data acquisition modules that rely on external APIs, you need to provide your own API keys. This is done by creating a `.env` file in the project's root directory. You can use `env_template.txt` as a starting point.

A typical `.env` file would look like this:

```
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

By centralizing configuration, the project is made more flexible and easier to adapt to different requirements.

## 6. Setup and Usage

Follow these steps to set up and run the project on your local machine.

#### 1. Clone the Repository

First, clone the project repository from your version control system.

```bash
# Replace with your repository's URL
git clone https://your-git-repository-url.com/fste_and_sandp_forcaster.git
cd fste_and_sandp_forcaster
```

#### 2. Set Up the Python Environment

It is highly recommended to use a virtual environment to manage the project's dependencies.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS and Linux:
source .venv/bin/activate
# On Windows:
# .venv\\Scripts\\activate

# Install the required packages
pip install -r requirements.txt
```

#### 3. Configure API Keys

The project requires API keys to access external data sources.

1.  Make a copy of the `env_template.txt` file and rename it to `.env`.
2.  Open the `.env` file and add your API keys for FRED, Twitter, and Reddit.

```
# .env file
FRED_API_KEY=YOUR_KEY_HERE
TWITTER_API_KEY=YOUR_KEY_HERE
...
```

If you don't provide API keys, the data acquisition scripts will fall back to using sample data, so the pipeline will still run.

#### 4. Run the Full Pipeline

The project has separate pipelines for data engineering and model training. Run them in the following order:

```bash
# 1. Run the data pipeline to collect and process data
python main_pipeline.py

# 2. Run the model training pipeline to train and evaluate models
python model_training_pipeline.py
```

After these scripts complete, you will have the processed data in `data/processed/` and the trained models in `models/`.

#### 5. Launch the Web Application

To interact with the models and generate new forecasts, start the web application:

```bash
python run_webapp.py
```

Open your web browser and navigate to `http://localhost:5001` to use the application.

## 7. Key Scripts

Here is a summary of the most important scripts in the project:

*   **`main_pipeline.py`**: The main entry point for the data engineering pipeline; it orchestrates data collection, cleaning, and preprocessing.
*   **`model_training_pipeline.py`**: The main entry point for the machine learning pipeline; it orchestrates model training, evaluation, and saving.
*   **`run_webapp.py`**: A simple script to launch the Flask web application for interactive forecasting.
*   **`config.py`**: The central configuration file where all project parameters, paths, and API key references are stored.
*   **`data_acquisition/` (directory)**: Contains the Python modules responsible for collecting data from various sources (e.g., `financial_data.py`, `economic_data.py`).
*   **`data_processing/data_cleaner.py`**: Contains the `DataCleaner` class, which handles all data cleaning and feature engineering tasks.
*   **`models/` (directory)**: Contains the implementations of the forecasting models (e.g., `lstm_model.py`, `prophet_model.py`) and the `ModelEvaluator`.
*   **`webapp/app.py`**: The core of the Flask web application, defining the API endpoints and server-side logic.
*   **`evaluate_*.py` (various scripts)**: A collection of standalone scripts for performing specific, in-depth model evaluations (e.g., on 2024 data).

This documentation should provide a comprehensive overview of the project. For more specific details, please refer to the source code of the individual scripts. 