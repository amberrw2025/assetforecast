import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exploratory Data Analysis (EDA) - Initial Findings\n",
                "\n",
                "**Objective:** Perform an initial exploration of the raw company and macroeconomic datasets. This corresponds to **Step 5** of the project plan.\n",
                "\n",
                "**Goals:**\n",
                "1. Load the raw datasets.\n",
                "2. Generate summary statistics.\n",
                "3. Visualize correlations and trends.\n",
                "4. Form initial hypotheses for feature engineering."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Load Raw Datasets"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "DATA_DIR = Path('../data/raw')\n",
                "\n",
                "try:\n",
                "    sp500_df = pd.read_csv(DATA_DIR / 'sp500_raw_data.csv', parse_dates=['date'])\n",
                "    ftse100_df = pd.read_csv(DATA_DIR / 'ftse100_raw_data.csv', parse_dates=['date'])\n",
                "    macro_df = pd.read_csv(DATA_DIR / 'macro_raw_data.csv', parse_dates=['date'])\n",
                "    \n",
                "    company_df = pd.concat([sp500_df, ftse100_df], ignore_index=True)\n",
                "    print(\"Raw datasets loaded successfully.\")\n",
                "except FileNotFoundError as e:\n",
                "    print(f\"Error: {e}.\\nPlease run the data ingestion scripts first:\")\n",
                "    print(\"python src/data_ingestion/get_company_data.py\")\n",
                "    print(\"python src/data_ingestion/get_macro_data.py\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Summary Statistics (Raw Data)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Company Data Summary:\")\n",
                "display(company_df.describe())\n",
                "\n",
                "print(\"Macroeconomic Data Summary:\")\n",
                "display(macro_df.describe())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Trend Analysis (Raw Data)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(14, 7))\n",
                "sns.lineplot(data=company_df, x='date', y='close_price', hue='ticker')\n",
                "plt.title('Stock Closing Prices (2023)')\n",
                "plt.ylabel('Close Price (in local currency)')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(14, 7))\n",
                "macro_melted = macro_df.melt('date', var_name='indicator', value_name='value')\n",
                "sns.lineplot(data=macro_melted, x='date', y='value', hue='indicator')\n",
                "plt.title('Macroeconomic Indicators (2023)')\n",
                "plt.ylabel('Value')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Analysis of Processed Data"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Now, let's load the processed and merged data to get a more holistic view."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "PROCESSED_DATA_PATH = Path('../data/processed/merged_data.csv')\n",
                "try:\n",
                "    processed_df = pd.read_csv(PROCESSED_DATA_PATH)\n",
                "    print(\"Processed data loaded successfully.\")\n",
                "    display(processed_df.head())\n",
                "except FileNotFoundError:\n",
                "    print(f\"File not found at {PROCESSED_DATA_PATH}.\")\n",
                "    print(\"Please run `python src/data_ingestion/process_data.py` first.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### 4.1. Feature Distributions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "numeric_cols = processed_df.select_dtypes(include=np.number).columns\n",
                "# Exclude some columns that are less interesting for distribution plots\n",
                "cols_to_plot = [col for col in numeric_cols if col not in ['Volume', 'total_revenue', 'full_time_employees']]\n",
                "\n",
                "processed_df[cols_to_plot].hist(figsize=(15, 12), bins=30, layout=(-1, 4))\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### 4.2. Correlation Heatmap"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(16, 12))\n",
                "# Re-calculating numeric_cols in case the cell above wasn't run\n",
                "numeric_cols_corr = processed_df.select_dtypes(include=np.number)\n",
                "correlation_matrix = numeric_cols_corr.corr()\n",
                "sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')\n",
                "plt.title('Correlation Matrix of Merged & Processed Data')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Insights from Modeling - Feature Importance"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "After running the XGBoost model, a feature importance plot is saved. Let's load and display it here to understand which factors the model found most predictive."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image\n",
                "\n",
                "IMPORTANCE_PLOT_PATH = Path('../reports/figures/xgboost_feature_importance.png')\n",
                "\n",
                "if IMPORTANCE_PLOT_PATH.exists():\n",
                "    print(\"Displaying XGBoost Feature Importance Plot:\")\n",
                "    display(Image(filename=IMPORTANCE_PLOT_PATH))\n",
                "else:\n",
                "    print(f\"Plot not found at {IMPORTANCE_PLOT_PATH}.\")\n",
                "    print(\"Please run `python src/modeling/xgboost_model.py` first.\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to a file
with open('notebooks/01_initial_eda.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 