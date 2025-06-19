"""
Main pipeline script for the Forecast Accuracy Assessment Model.
Orchestrates the entire data acquisition, cleaning, and preparation process.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import *
from data_acquisition import FinancialDataCollector, EconomicDataCollector, SentimentDataCollector
from data_processing import DataCleaner
from utils import get_logger, DataVisualizer
from dvc_integration import DVCManager

logger = get_logger("main_pipeline")

class ForecastModelPipeline:
    """
    Main pipeline class that orchestrates the entire data processing workflow.
    """
    
    def __init__(self):
        self.financial_collector = FinancialDataCollector()
        self.economic_collector = EconomicDataCollector()
        self.sentiment_collector = SentimentDataCollector()
        self.data_cleaner = DataCleaner()
        self.visualizer = DataVisualizer()
        self.dvc_manager = DVCManager()
        
        # Store collected data
        self.raw_datasets = {}
        self.cleaned_dataset = None
        
    def run_data_acquisition(self) -> bool:
        """
        Run the complete data acquisition process.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting data acquisition phase")
        
        try:
            # Step 1: Collect financial data
            logger.info("Collecting financial data...")
            financial_data = self.financial_collector.collect_all_companies_data()
            
            if not financial_data.empty:
                self.raw_datasets['financial'] = financial_data
                logger.info(f"Financial data collected: {len(financial_data)} records")
            else:
                logger.warning("No financial data collected")
            
            # Step 2: Collect economic data
            logger.info("Collecting economic data...")
            economic_data = self.economic_collector.collect_all_economic_data()
            
            if economic_data:
                self.raw_datasets.update(economic_data)
                total_economic_records = sum(len(df) for df in economic_data.values())
                logger.info(f"Economic data collected: {total_economic_records} records across {len(economic_data)} sources")
            else:
                logger.warning("No economic data collected")
            
            # Step 3: Collect sentiment data
            logger.info("Collecting sentiment data...")
            sentiment_data = self.sentiment_collector.collect_all_sentiment_data()
            
            if sentiment_data:
                self.raw_datasets.update(sentiment_data)
                total_sentiment_records = sum(len(df) for df in sentiment_data.values())
                logger.info(f"Sentiment data collected: {total_sentiment_records} records across {len(sentiment_data)} sources")
            else:
                logger.warning("No sentiment data collected")
            
            # Save raw data
            self._save_raw_data()
            
            logger.info("Data acquisition phase completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in data acquisition phase: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def run_data_cleaning(self) -> bool:
        """
        Run the complete data cleaning and preparation process.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting data cleaning phase")
        
        try:
            if not self.raw_datasets:
                logger.error("No raw datasets available. Run data acquisition first.")
                return False
            
            # Clean and prepare all datasets
            self.cleaned_dataset = self.data_cleaner.clean_and_prepare_data(self.raw_datasets)
            
            if self.cleaned_dataset is not None and not self.cleaned_dataset.empty:
                logger.info(f"Data cleaning completed. Final dataset shape: {self.cleaned_dataset.shape}")
                
                # Save cleaned data
                self._save_cleaned_data()
                
                logger.info("Data cleaning phase completed successfully")
                return True
            else:
                logger.error("Data cleaning failed - no valid dataset produced")
                return False
                
        except Exception as e:
            logger.error(f"Error in data cleaning phase: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def run_exploratory_analysis(self) -> bool:
        """
        Run exploratory data analysis and create visualizations.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting exploratory data analysis phase")
        
        try:
            if self.cleaned_dataset is None or self.cleaned_dataset.empty:
                logger.error("No cleaned dataset available. Run data cleaning first.")
                return False
            
            # Create visualizations
            self._create_analysis_plots()
            
            # Generate summary statistics
            self._generate_summary_statistics()
            
            logger.info("Exploratory data analysis phase completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in exploratory analysis phase: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def setup_dvc_tracking(self) -> bool:
        """
        Setup DVC tracking for the project.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Setting up DVC tracking")
        
        try:
            # Initialize DVC
            if not self.dvc_manager.initialize_dvc():
                logger.warning("DVC initialization failed. Continuing without DVC tracking.")
                return False
            
            # Add data files to tracking
            data_files = [
                "data/raw/company_financials.csv",
                "data/raw/economic_indicators.csv",
                "data/raw/sentiment_data.csv",
                "data/processed/cleaned_dataset.csv"
            ]
            
            results = self.dvc_manager.add_multiple_data_files(data_files)
            success_count = sum(results.values())
            
            logger.info(f"DVC tracking setup completed. {success_count}/{len(data_files)} files tracked.")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error setting up DVC tracking: {str(e)}")
            return False
    
    def _save_raw_data(self):
        """Save raw datasets to files."""
        try:
            for name, data in self.raw_datasets.items():
                if not data.empty:
                    filename = f"{name}_data.csv"
                    filepath = RAW_DATA_DIR / filename
                    data.to_csv(filepath, index=False)
                    logger.info(f"Raw {name} data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving raw data: {str(e)}")
    
    def _save_cleaned_data(self):
        """Save cleaned dataset to file."""
        try:
            filepath = self.data_cleaner.save_cleaned_data(self.cleaned_dataset)
            logger.info(f"Cleaned data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving cleaned data: {str(e)}")
    
    def _create_analysis_plots(self):
        """Create exploratory analysis plots."""
        try:
            # Time series plot
            if 'date' in self.cleaned_dataset.columns:
                numeric_cols = self.cleaned_dataset.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    fig = self.visualizer.plot_time_series(
                        self.cleaned_dataset,
                        y_columns=numeric_cols[:5],
                        title="Time Series Analysis"
                    )
                    self.visualizer.save_plot(fig, PROCESSED_DATA_DIR / "time_series_analysis.html")
            
            # Correlation matrix
            numeric_cols = self.cleaned_dataset.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                fig = self.visualizer.plot_correlation_matrix(
                    self.cleaned_dataset,
                    columns=numeric_cols[:10],
                    title="Feature Correlation Matrix"
                )
                self.visualizer.save_plot(fig, PROCESSED_DATA_DIR / "correlation_matrix.html")
            
            # Missing values analysis
            fig = self.visualizer.plot_missing_values(
                self.cleaned_dataset,
                title="Missing Values Analysis"
            )
            self.visualizer.save_plot(fig, PROCESSED_DATA_DIR / "missing_values_analysis.html")
            
            # Distribution plots
            fig = self.visualizer.plot_distribution(
                self.cleaned_dataset,
                title="Feature Distributions"
            )
            self.visualizer.save_plot(fig, PROCESSED_DATA_DIR / "feature_distributions.html")
            
            logger.info("Analysis plots created and saved")
            
        except Exception as e:
            logger.error(f"Error creating analysis plots: {str(e)}")
    
    def _generate_summary_statistics(self):
        """Generate and save summary statistics."""
        try:
            # Basic statistics
            summary_stats = self.cleaned_dataset.describe()
            
            # Save to file
            summary_file = PROCESSED_DATA_DIR / "summary_statistics.csv"
            summary_stats.to_csv(summary_file)
            
            # Print summary
            print("\n" + "="*50)
            print("DATASET SUMMARY")
            print("="*50)
            print(f"Shape: {self.cleaned_dataset.shape}")
            print(f"Columns: {list(self.cleaned_dataset.columns)}")
            print(f"Date range: {self.cleaned_dataset['date'].min()} to {self.cleaned_dataset['date'].max()}")
            print(f"Memory usage: {self.cleaned_dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Missing values summary
            missing_summary = self.cleaned_dataset.isnull().sum()
            if missing_summary.sum() > 0:
                print(f"\nMissing values:\n{missing_summary[missing_summary > 0]}")
            else:
                print("\nNo missing values in the dataset")
            
            print("="*50)
            
            logger.info("Summary statistics generated and saved")
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {str(e)}")
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete pipeline from data acquisition to analysis.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting complete forecast model pipeline")
        
        start_time = datetime.now()
        
        try:
            # Phase 1: Data Acquisition
            if not self.run_data_acquisition():
                logger.error("Data acquisition phase failed")
                return False
            
            # Phase 2: Data Cleaning
            if not self.run_data_cleaning():
                logger.error("Data cleaning phase failed")
                return False
            
            # Phase 3: Exploratory Analysis
            if not self.run_exploratory_analysis():
                logger.error("Exploratory analysis phase failed")
                return False
            
            # Phase 4: DVC Setup (optional)
            self.setup_dvc_tracking()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"Complete pipeline executed successfully in {duration}")
            
            # Print final summary
            self._print_pipeline_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _print_pipeline_summary(self):
        """Print a summary of the pipeline execution."""
        print("\n" + "="*60)
        print("FORECAST MODEL PIPELINE - EXECUTION SUMMARY")
        print("="*60)
        
        print(f"Raw datasets collected: {len(self.raw_datasets)}")
        for name, data in self.raw_datasets.items():
            print(f"  - {name}: {len(data)} records")
        
        if self.cleaned_dataset is not None:
            print(f"\nFinal cleaned dataset: {self.cleaned_dataset.shape}")
            print(f"Features available: {list(self.cleaned_dataset.columns)}")
        
        print(f"\nOutput files created:")
        print(f"  - Raw data: {RAW_DATA_DIR}")
        print(f"  - Processed data: {PROCESSED_DATA_DIR}")
        print(f"  - Analysis plots: {PROCESSED_DATA_DIR}")
        
        print("\nNext steps:")
        print("1. Review the generated visualizations")
        print("2. Proceed to model training and validation")
        print("3. Implement the web application")
        
        print("="*60)

def main():
    """
    Main function to run the complete pipeline.
    """
    logger.info("Initializing Forecast Model Pipeline")
    
    # Create pipeline instance
    pipeline = ForecastModelPipeline()
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        logger.info("Pipeline completed successfully!")
        print("\n🎉 Pipeline completed successfully!")
        print("📊 Check the 'data/processed/' directory for results and visualizations")
        print("📈 Ready for model training and web app development")
    else:
        logger.error("Pipeline failed!")
        print("\n❌ Pipeline failed! Check the logs for details")
    
    return success

if __name__ == "__main__":
    main() 