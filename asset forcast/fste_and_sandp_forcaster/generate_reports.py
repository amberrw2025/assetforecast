#!/usr/bin/env python3
"""
Comprehensive Reporting and Visualization Generator
Generates detailed reports, visualizations, and deployment documentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import joblib
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import our modules
from models import ARIMAModel, ProphetModel, LSTMModel, EnsembleModel, ModelEvaluator
from config import PROCESSED_DATA_DIR, MODELS_DIR

class ReportGenerator:
    """Generate comprehensive reports and visualizations."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.data = None
        self.models = {}
        self.results = {}
        self.reports_dir = PROCESSED_DATA_DIR / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        print("📊 Report Generator Initialized")
    
    def load_data_and_models(self):
        """Load processed data and trained models."""
        print("📂 Loading data and models...")
        
        # Load data
        data_path = PROCESSED_DATA_DIR / "cleaned_dataset.csv"
        self.data = pd.read_csv(data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date')
        
        # Load models
        model_files = {
            'ARIMA': 'arima',
            'Prophet': 'prophet', 
            'LSTM': 'lstm'
        }
        
        for name, filename in model_files.items():
            try:
                model_path = MODELS_DIR / filename
                if name == 'ARIMA':
                    model = ARIMAModel()
                elif name == 'Prophet':
                    model = ProphetModel()
                elif name == 'LSTM':
                    model = LSTMModel()
                
                model.load_model(str(model_path))
                self.models[name] = model
                print(f"✅ Loaded {name} model")
                
            except Exception as e:
                print(f"⚠️ Could not load {name} model: {e}")
        
        print(f"📈 Loaded {len(self.data)} data records")
        print(f"🤖 Loaded {len(self.models)} trained models")
    
    def generate_model_performance_report(self):
        """Generate detailed model performance report."""
        print("📊 Generating model performance report...")
        
        # Create evaluator
        evaluator = ModelEvaluator()
        for name, model in self.models.items():
            evaluator.add_model(model, name)
        
        # Evaluate models
        test_size = 0.2
        total_size = len(self.data)
        test_samples = int(total_size * test_size)
        test_data = self.data.iloc[-test_samples:].copy()
        
        results = evaluator.evaluate_all_models(test_data)
        
        # Generate comparison
        comparison = evaluator.compare_models('rmse')
        
        # Create performance visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RMSE Comparison
        axes[0, 0].bar(comparison['Model'], comparison['RMSE'])
        axes[0, 0].set_title('Model RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE Comparison
        mae_data = []
        for model_name, metrics in results.items():
            if 'mae' in metrics:
                mae_data.append({'Model': model_name, 'MAE': metrics['mae']})
        
        if mae_data:
            mae_df = pd.DataFrame(mae_data)
            axes[0, 1].bar(mae_df['Model'], mae_df['MAE'])
            axes[0, 1].set_title('Model MAE Comparison')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² Comparison
        r2_data = []
        for model_name, metrics in results.items():
            if 'r2' in metrics:
                r2_data.append({'Model': model_name, 'R²': metrics['r2']})
        
        if r2_data:
            r2_df = pd.DataFrame(r2_data)
            axes[1, 0].bar(r2_df['Model'], r2_df['R²'])
            axes[1, 0].set_title('Model R² Comparison')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE Comparison
        mape_data = []
        for model_name, metrics in results.items():
            if 'mape' in metrics:
                mape_data.append({'Model': model_name, 'MAPE': metrics['mape']})
        
        if mape_data:
            mape_df = pd.DataFrame(mape_data)
            axes[1, 1].bar(mape_df['Model'], mape_df['MAPE'])
            axes[1, 1].set_title('Model MAPE Comparison')
            axes[1, 1].set_ylabel('MAPE (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'model_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(self.reports_dir / 'model_performance_metrics.csv')
        
        print(f"✅ Performance report saved to {self.reports_dir}")
        return results
    
    def generate_forecast_visualizations(self):
        """Generate forecast visualizations."""
        print("🔮 Generating forecast visualizations...")
        
        # Get test data
        test_size = 0.2
        total_size = len(self.data)
        test_samples = int(total_size * test_size)
        test_data = self.data.iloc[-test_samples:].copy()
        
        # Create forecast plot
        plt.figure(figsize=(15, 8))
        
        # Plot actual values
        plt.plot(test_data['date'], test_data['close_price'], 
                label='Actual Values', linewidth=2, color='black')
        
        # Plot predictions for each model
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                predictions = model.predict_in_sample(test_data)
                
                # Remove NaN values
                mask = ~np.isnan(predictions)
                pred_dates = test_data['date'].iloc[mask]
                pred_values = predictions[mask]
                
                plt.plot(pred_dates, pred_values, 
                        label=f'{model_name} Predictions', 
                        color=colors[i % len(colors)], alpha=0.7, linewidth=1.5)
                
            except Exception as e:
                print(f"⚠️ Could not plot predictions for {model_name}: {e}")
        
        plt.title('Model Forecast Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.reports_dir / 'forecast_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Forecast visualization saved to {self.reports_dir}")
    
    def generate_data_insights(self):
        """Generate data insights and analysis."""
        print("📈 Generating data insights...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Time series plot
        axes[0, 0].plot(self.data['date'], self.data['close_price'])
        axes[0, 0].set_title('Close Price Time Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Close Price')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Price distribution
        axes[0, 1].hist(self.data['close_price'].dropna(), bins=50, alpha=0.7)
        axes[0, 1].set_title('Close Price Distribution')
        axes[0, 1].set_xlabel('Close Price')
        axes[0, 1].set_ylabel('Frequency')
        
        # Rolling statistics
        rolling_mean = self.data['close_price'].rolling(window=30).mean()
        rolling_std = self.data['close_price'].rolling(window=30).std()
        
        axes[1, 0].plot(self.data['date'], self.data['close_price'], 
                       label='Actual', alpha=0.7)
        axes[1, 0].plot(self.data['date'], rolling_mean, 
                       label='30-day Moving Average', linewidth=2)
        axes[1, 0].fill_between(self.data['date'], 
                               rolling_mean - rolling_std, 
                               rolling_mean + rolling_std, 
                               alpha=0.3, label='±1 Std Dev')
        axes[1, 0].set_title('Price with Rolling Statistics')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Close Price')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Volatility analysis
        returns = self.data['close_price'].pct_change().dropna()
        axes[1, 1].hist(returns, bins=50, alpha=0.7)
        axes[1, 1].set_title('Price Returns Distribution')
        axes[1, 1].set_xlabel('Returns')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'data_insights.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save data statistics
        stats = {
            'total_records': len(self.data),
            'date_range': {
                'start': self.data['date'].min().strftime('%Y-%m-%d'),
                'end': self.data['date'].max().strftime('%Y-%m-%d')
            },
            'price_stats': {
                'mean': self.data['close_price'].mean(),
                'std': self.data['close_price'].std(),
                'min': self.data['close_price'].min(),
                'max': self.data['close_price'].max(),
                'volatility': returns.std()
            }
        }
        
        with open(self.reports_dir / 'data_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Data insights saved to {self.reports_dir}")
    
    def generate_deployment_report(self):
        """Generate deployment-ready report."""
        print("🚀 Generating deployment report...")
        
        report = f"""
# FORECAST MODEL DEPLOYMENT REPORT

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## MODEL OVERVIEW

This report contains information for deploying the trained forecasting models.

### Available Models
"""
        
        for name, model in self.models.items():
            report += f"""
**{name} Model**
- Status: {'✅ Trained' if model.is_fitted else '❌ Not Trained'}
- Parameters: {model.model_params}
- File: {name.lower()}.joblib
"""
        
        report += f"""

## DEPLOYMENT INSTRUCTIONS

### 1. Model Loading
```python
from models import ARIMAModel, ProphetModel, LSTMModel

# Load models
arima = ARIMAModel()
arima.load_model('models/arima')

prophet = ProphetModel()
prophet.load_model('models/prophet')

lstm = LSTMModel()
lstm.load_model('models/lstm')
```

### 2. Making Predictions
```python
# Load your data
data = pd.read_csv('data/processed/cleaned_dataset.csv')
data['date'] = pd.to_datetime(data['date'])

# Make predictions
arima_forecast = arima.predict(data, steps=30)
prophet_forecast = prophet.predict(data, periods=30)
lstm_forecast = lstm.predict(data, steps=30)
```

### 3. Model Evaluation
```python
from models import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.add_model(arima, 'ARIMA')
evaluator.add_model(prophet, 'Prophet')
evaluator.add_model(lstm, 'LSTM')

metrics = evaluator.evaluate_all_models(test_data)
```

## PERFORMANCE SUMMARY

Based on the latest evaluation:

"""
        
        # Add performance metrics if available
        try:
            test_size = 0.2
            total_size = len(self.data)
            test_samples = int(total_size * test_size)
            test_data = self.data.iloc[-test_samples:].copy()
            
            evaluator = ModelEvaluator()
            for name, model in self.models.items():
                evaluator.add_model(model, name)
            
            results = evaluator.evaluate_all_models(test_data)
            
            report += "| Model | RMSE | MAE | R² | MAPE |\n"
            report += "|-------|------|-----|----|------|\n"
            
            for model_name, metrics in results.items():
                rmse = metrics.get('rmse', 'N/A')
                mae = metrics.get('mae', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                mape = metrics.get('mape', 'N/A')
                
                report += f"| {model_name} | {rmse:.4f} | {mae:.4f} | {r2:.4f} | {mape:.2f}% |\n"
        
        except Exception as e:
            report += f"Performance metrics could not be calculated: {e}\n"
        
        report += f"""

## RECOMMENDATIONS

1. **Best Model**: Use the model with the lowest RMSE for production
2. **Ensemble**: Consider using an ensemble of multiple models for better accuracy
3. **Monitoring**: Implement model performance monitoring in production
4. **Retraining**: Schedule regular model retraining with new data

## FILES AND LOCATIONS

- **Trained Models**: `models/` directory
- **Processed Data**: `data/processed/` directory
- **Reports**: `data/processed/reports/` directory
- **Configuration**: `config.py`

## SUPPORT

For questions or issues with model deployment, refer to the project documentation.
"""
        
        with open(self.reports_dir / 'deployment_report.md', 'w') as f:
            f.write(report)
        
        print(f"✅ Deployment report saved to {self.reports_dir}")
    
    def generate_executive_summary(self):
        """Generate executive summary report."""
        print("📋 Generating executive summary...")
        
        summary = f"""
# EXECUTIVE SUMMARY - FORECAST ACCURACY ASSESSMENT MODEL

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## PROJECT OVERVIEW

This project implements a comprehensive forecast accuracy assessment model for financial time series data. The system includes multiple forecasting algorithms and provides detailed performance analysis.

## KEY ACHIEVEMENTS

✅ **Data Pipeline**: Successfully processed and cleaned financial data
✅ **Model Training**: Trained {len(self.models)} different forecasting models
✅ **Performance Evaluation**: Comprehensive model comparison and evaluation
✅ **Deployment Ready**: Models saved and ready for production use

## MODEL PERFORMANCE

"""
        
        # Add performance summary
        try:
            test_size = 0.2
            total_size = len(self.data)
            test_samples = int(total_size * test_size)
            test_data = self.data.iloc[-test_samples:].copy()
            
            evaluator = ModelEvaluator()
            for name, model in self.models.items():
                evaluator.add_model(model, name)
            
            results = evaluator.evaluate_all_models(test_data)
            
            # Find best model
            best_model = None
            best_rmse = float('inf')
            
            for model_name, metrics in results.items():
                rmse = metrics.get('rmse', float('inf'))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_name
            
            summary += f"""
**Best Performing Model**: {best_model}
**Best RMSE**: {best_rmse:.4f}

**Model Comparison**:
"""
            
            for model_name, metrics in results.items():
                rmse = metrics.get('rmse', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                summary += f"- {model_name}: RMSE={rmse:.4f}, R²={r2:.4f}\n"
        
        except Exception as e:
            summary += f"Performance analysis could not be completed: {e}\n"
        
        summary += f"""

## DATA INSIGHTS

- **Total Records**: {len(self.data):,}
- **Date Range**: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}
- **Features**: {len(self.data.columns)} variables
- **Data Quality**: Cleaned and preprocessed for optimal model performance

## BUSINESS IMPACT

1. **Improved Forecasting**: Multiple model approaches provide robust predictions
2. **Risk Management**: Better understanding of forecast uncertainty
3. **Decision Support**: Data-driven insights for financial planning
4. **Scalability**: Modular design allows easy expansion and updates

## NEXT STEPS

1. **Production Deployment**: Deploy best-performing model to production
2. **Monitoring**: Implement real-time performance monitoring
3. **Continuous Improvement**: Regular model retraining and optimization
4. **Feature Engineering**: Explore additional features for improved accuracy

## TECHNICAL DETAILS

- **Programming Language**: Python 3.9+
- **Key Libraries**: TensorFlow, Prophet, StatsModels, Scikit-learn
- **Architecture**: Modular, scalable pipeline design
- **Documentation**: Comprehensive reports and deployment guides

---
*This report was automatically generated by the Forecast Accuracy Assessment Model Pipeline*
"""
        
        with open(self.reports_dir / 'executive_summary.md', 'w') as f:
            f.write(summary)
        
        print(f"✅ Executive summary saved to {self.reports_dir}")
    
    def run_complete_reporting(self):
        """Run complete reporting pipeline."""
        print("🚀 Starting complete reporting pipeline...")
        print("=" * 60)
        
        try:
            # Load data and models
            self.load_data_and_models()
            
            # Generate all reports
            self.generate_data_insights()
            self.generate_model_performance_report()
            self.generate_forecast_visualizations()
            self.generate_deployment_report()
            self.generate_executive_summary()
            
            print("=" * 60)
            print("✅ COMPLETE REPORTING PIPELINE FINISHED")
            print("=" * 60)
            print(f"📁 Reports saved to: {self.reports_dir}")
            print("📊 Generated reports:")
            print("   - Model performance comparison")
            print("   - Forecast visualizations")
            print("   - Data insights analysis")
            print("   - Deployment guide")
            print("   - Executive summary")
            print("\n🎯 Ready for production deployment!")
            
        except Exception as e:
            print(f"❌ Error in reporting pipeline: {e}")
            raise


def main():
    """Main function to run the reporting pipeline."""
    print("📊 FORECAST MODEL REPORTING GENERATOR")
    print("=" * 50)
    
    generator = ReportGenerator()
    generator.run_complete_reporting()


if __name__ == "__main__":
    main() 