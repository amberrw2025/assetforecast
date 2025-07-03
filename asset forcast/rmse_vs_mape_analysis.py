#!/usr/bin/env python3
"""
RMSE vs MAPE Analysis for Standard Forecast Graphs
================================================

Comprehensive comparison of Root Mean Square Error (RMSE) vs 
Mean Absolute Percentage Error (MAPE) for forecast evaluation.

Key Insights:
- RMSE: Sensitive to outliers, uses actual price units
- MAPE: Scale-independent, good for relative comparison
- Both provide complementary insights for forecast quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import seaborn as sns
warnings.filterwarnings('ignore')

class RMSEMAPEAnalyzer:
    def __init__(self):
        self.results = []
        self.output_dir = Path('rmse_mape_analysis')
        self.output_dir.mkdir(exist_ok=True)
    
    def calculate_both_metrics(self, actual, predicted, method_name="Method"):
        """Calculate both RMSE and MAPE with error handling"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return None
        
        # RMSE calculation
        rmse = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
        
        # MAPE calculation (handle division by zero)
        non_zero_mask = actual_clean != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual_clean[non_zero_mask] - predicted_clean[non_zero_mask]) 
                                / actual_clean[non_zero_mask])) * 100
        else:
            mape = np.inf
        
        # Additional metrics
        mae = np.mean(np.abs(actual_clean - predicted_clean))
        mean_price = np.mean(actual_clean)
        price_range = np.max(actual_clean) - np.min(actual_clean)
        
        # Normalized RMSE (as percentage of mean price)
        rmse_normalized = (rmse / mean_price) * 100 if mean_price > 0 else np.inf
        
        return {
            'method': method_name,
            'rmse': rmse,
            'mape': mape,
            'mae': mae,
            'rmse_normalized': rmse_normalized,
            'mean_price': mean_price,
            'price_range': price_range,
            'n_points': len(actual_clean)
        }
    
    def analyze_ticker(self, ticker):
        """Analyze RMSE vs MAPE for a single ticker"""
        try:
            print(f"📊 Analyzing {ticker}...")
            
            # Download data
            train_data = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)
            actual_2024 = yf.download(ticker, start='2024-01-01', end='2024-06-30', progress=False)
            
            if train_data.empty or actual_2024.empty:
                print(f"   ❌ Insufficient data for {ticker}")
                return None
            
            # Handle new yfinance format
            if isinstance(train_data['Close'], pd.DataFrame):
                prices = train_data['Close'].iloc[:, 0].values
                actual_prices = actual_2024['Close'].iloc[:, 0].values
            else:
                prices = train_data['Close'].values
                actual_prices = actual_2024['Close'].values
            
            # Calculate trend (same as standardize script)
            if len(prices) >= 30:
                trend = (prices[-1] - prices[-30]) / 30
            else:
                trend = (prices[-1] - prices[0]) / len(prices)
            
            # Generate forecasts
            forecast_days = min(126, len(actual_prices))
            
            # OLD Method (simple linear trend)
            old_forecast = []
            for i in range(1, forecast_days + 1):
                predicted_price = prices[-1] + (trend * i)
                old_forecast.append(max(0.01, predicted_price))
            
            # NEW Enhanced (with volatility adjustment)
            enhanced_forecast = []
            for i in range(1, forecast_days + 1):
                volatility_factor = 1 + (0.001 * i)
                predicted_price = prices[-1] + (trend * i * volatility_factor)
                enhanced_forecast.append(max(0.01, predicted_price))
            
            # Naive baseline (last price)
            naive_forecast = [prices[-1]] * forecast_days
            
            # Calculate metrics for all methods
            overlap_days = min(len(actual_prices), forecast_days)
            actual_subset = actual_prices[:overlap_days]
            
            old_metrics = self.calculate_both_metrics(
                actual_subset, old_forecast[:overlap_days], "OLD Method")
            enhanced_metrics = self.calculate_both_metrics(
                actual_subset, enhanced_forecast[:overlap_days], "NEW Enhanced")
            naive_metrics = self.calculate_both_metrics(
                actual_subset, naive_forecast[:overlap_days], "Naive")
            
            # Store results
            market = "FTSE_100" if ticker.endswith('.L') else "SP_500"
            
            for metrics in [old_metrics, enhanced_metrics, naive_metrics]:
                if metrics:
                    metrics.update({
                        'ticker': ticker,
                        'market': market,
                        'price_level': 'high' if metrics['mean_price'] > 100 else 'low'
                    })
                    self.results.append(metrics)
            
            print(f"   ✅ Analysis complete for {ticker}")
            
            # Print comparison
            if old_metrics and enhanced_metrics:
                print(f"      OLD  - RMSE: {old_metrics['rmse']:.2f}, MAPE: {old_metrics['mape']:.1f}%")
                print(f"      NEW  - RMSE: {enhanced_metrics['rmse']:.2f}, MAPE: {enhanced_metrics['mape']:.1f}%")
                
                # Which metric shows bigger improvement?
                rmse_improvement = ((old_metrics['rmse'] - enhanced_metrics['rmse']) / old_metrics['rmse']) * 100
                mape_improvement = ((old_metrics['mape'] - enhanced_metrics['mape']) / old_metrics['mape']) * 100
                
                print(f"      Improvements - RMSE: {rmse_improvement:.1f}%, MAPE: {mape_improvement:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error analyzing {ticker}: {e}")
            return False
    
    def create_comprehensive_plots(self):
        """Create comprehensive comparison plots"""
        if not self.results:
            print("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. RMSE vs MAPE scatter plot
        ax1 = plt.subplot(2, 3, 1)
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            plt.scatter(method_data['rmse'], method_data['mape'], 
                       label=method, alpha=0.7, s=60)
        plt.xlabel('RMSE (Price Units)')
        plt.ylabel('MAPE (%)')
        plt.title('RMSE vs MAPE Scatter')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Normalized RMSE comparison
        ax2 = plt.subplot(2, 3, 2)
        method_rmse = df.groupby('method')['rmse_normalized'].agg(['mean', 'std'])
        x_pos = range(len(method_rmse))
        plt.bar(x_pos, method_rmse['mean'], yerr=method_rmse['std'], 
                capsize=5, alpha=0.7)
        plt.xticks(x_pos, method_rmse.index, rotation=45)
        plt.ylabel('Normalized RMSE (%)')
        plt.title('Average Normalized RMSE by Method')
        plt.grid(True, alpha=0.3)
        
        # 3. MAPE comparison
        ax3 = plt.subplot(2, 3, 3)
        method_mape = df.groupby('method')['mape'].agg(['mean', 'std'])
        x_pos = range(len(method_mape))
        plt.bar(x_pos, method_mape['mean'], yerr=method_mape['std'], 
                capsize=5, alpha=0.7, color='orange')
        plt.xticks(x_pos, method_mape.index, rotation=45)
        plt.ylabel('MAPE (%)')
        plt.title('Average MAPE by Method')
        plt.grid(True, alpha=0.3)
        
        # 4. Market comparison (RMSE)
        ax4 = plt.subplot(2, 3, 4)
        market_rmse = df.groupby(['market', 'method'])['rmse_normalized'].mean().unstack()
        market_rmse.plot(kind='bar', ax=ax4, width=0.8)
        plt.title('Normalized RMSE by Market')
        plt.ylabel('Normalized RMSE (%)')
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 5. Market comparison (MAPE)
        ax5 = plt.subplot(2, 3, 5)
        market_mape = df.groupby(['market', 'method'])['mape'].mean().unstack()
        market_mape.plot(kind='bar', ax=ax5, width=0.8, colormap='viridis')
        plt.title('MAPE by Market')
        plt.ylabel('MAPE (%)')
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 6. Price level impact
        ax6 = plt.subplot(2, 3, 6)
        price_level_comparison = df.groupby(['price_level', 'method'])[['rmse_normalized', 'mape']].mean()
        
        # Create grouped bar chart
        price_levels = price_level_comparison.index.get_level_values(0).unique()
        methods = price_level_comparison.index.get_level_values(1).unique()
        
        x = np.arange(len(price_levels))
        width = 0.25
        
        for i, method in enumerate(methods):
            rmse_vals = [price_level_comparison.loc[(pl, method), 'rmse_normalized'] 
                        for pl in price_levels if (pl, method) in price_level_comparison.index]
            plt.bar(x + i*width, rmse_vals, width, label=f'{method} (RMSE%)', alpha=0.7)
        
        plt.xlabel('Price Level')
        plt.ylabel('Normalized RMSE (%)')
        plt.title('Price Level Impact on RMSE')
        plt.xticks(x + width, price_levels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rmse_vs_mape_comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self):
        """Create a detailed summary report"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("📊 RMSE vs MAPE ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        # Overall statistics
        print("\n🔍 OVERALL STATISTICS:")
        print("-" * 40)
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            rmse_avg = method_data['rmse'].mean()
            mape_avg = method_data['mape'].mean()
            rmse_norm_avg = method_data['rmse_normalized'].mean()
            
            print(f"{method:15} - RMSE: {rmse_avg:6.2f}, MAPE: {mape_avg:5.1f}%, Norm RMSE: {rmse_norm_avg:5.1f}%")
        
        # Market breakdown
        print("\n🌍 MARKET BREAKDOWN:")
        print("-" * 40)
        for market in df['market'].unique():
            market_data = df[df['market'] == market]
            print(f"\n{market}:")
            for method in market_data['method'].unique():
                method_market_data = market_data[market_data['method'] == method]
                rmse_avg = method_market_data['rmse_normalized'].mean()
                mape_avg = method_market_data['mape'].mean()
                print(f"  {method:12} - Norm RMSE: {rmse_avg:5.1f}%, MAPE: {mape_avg:5.1f}%")
        
        # Best performers
        print("\n🏆 BEST PERFORMERS:")
        print("-" * 40)
        
        # Best by RMSE
        best_rmse = df.loc[df['rmse_normalized'].idxmin()]
        print(f"Lowest Normalized RMSE: {best_rmse['ticker']} ({best_rmse['method']}) - {best_rmse['rmse_normalized']:.1f}%")
        
        # Best by MAPE
        best_mape = df.loc[df['mape'].idxmin()]
        print(f"Lowest MAPE: {best_mape['ticker']} ({best_mape['method']}) - {best_mape['mape']:.1f}%")
        
        # Metric correlation
        correlation = df['rmse_normalized'].corr(df['mape'])
        print(f"\n📈 RMSE-MAPE Correlation: {correlation:.3f}")
        
        print("\n💡 KEY INSIGHTS:")
        print("-" * 40)
        print("• RMSE is sensitive to price scale - use normalized RMSE for comparison")
        print("• MAPE is scale-independent - better for cross-stock comparison")
        print("• High correlation suggests both metrics agree on forecast quality")
        print("• Large RMSE with low MAPE indicates price scale effects")
        print("• Both metrics provide complementary evaluation perspectives")
        
        # Save detailed results
        df.to_csv(self.output_dir / 'detailed_rmse_mape_results.csv', index=False)
        print(f"\n📁 Detailed results saved to: {self.output_dir / 'detailed_rmse_mape_results.csv'}")

def main():
    """Main execution function"""
    print("🔬 RMSE vs MAPE ANALYSIS FOR STANDARD FORECAST GRAPHS")
    print("=" * 80)
    print("🎯 Comparing Root Mean Square Error vs Mean Absolute Percentage Error")
    print("📊 Understanding when to use each metric for forecast evaluation")
    print()
    
    analyzer = RMSEMAPEAnalyzer()
    
    # Get sample tickers (mix of high and low price stocks)
    sample_tickers = [
        # US Stocks (S&P 500)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'JPM', 'V', 'JNJ',
        # UK Stocks (FTSE 100)  
        'BARC.L', 'BP.L', 'LLOY.L', 'VOD.L', 'TSCO.L',
        'AZN.L', 'RIO.L', 'SHEL.L', 'LSEG.L', 'ULVR.L'
    ]
    
    print(f"📈 Analyzing {len(sample_tickers)} tickers for RMSE vs MAPE comparison...")
    print("-" * 50)
    
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(sample_tickers, 1):
        print(f"[{i:2d}/{len(sample_tickers)}]", end=" ")
        if analyzer.analyze_ticker(ticker):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 80)
    print("📊 ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"✅ Successfully analyzed: {successful} tickers")
    print(f"❌ Failed: {failed} tickers")
    
    if analyzer.results:
        # Create comprehensive plots
        analyzer.create_comprehensive_plots()
        
        # Generate summary report
        analyzer.create_summary_report()
        
        print("\n🎉 RMSE vs MAPE analysis complete!")
        print(f"📁 Results saved to: {analyzer.output_dir}")
    else:
        print("❌ No results generated for analysis")

if __name__ == "__main__":
    main() 