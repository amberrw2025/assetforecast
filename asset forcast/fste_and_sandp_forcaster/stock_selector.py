#!/usr/bin/env python3
"""
Interactive Stock Selection and Forecasting Tool
Choose a stock and view all data plus forecasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockAnalyzer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.available_stocks = self.get_available_stocks()
        self.stock_names = {
            'AZN_L': 'AstraZeneca (FTSE)', 'LSEG_L': 'London Stock Exchange (FTSE)',
            'RKT_L': 'Reckitt Benckiser (FTSE)', 'OCDO_L': 'Ocado Group (FTSE)',
            'CRDA_L': 'Croda International (FTSE)', 'BT-A_L': 'BT Group (FTSE)',
            'VOD_L': 'Vodafone Group (FTSE)', 'SSE_L': 'SSE (FTSE)',
            'GLEN_L': 'Glencore (FTSE)', 'TSCO_L': 'Tesco (FTSE)',
            'NVDA': 'NVIDIA Corporation (S&P 500)', 'TSLA': 'Tesla Inc (S&P 500)',
            'MRNA': 'Moderna Inc (S&P 500)', 'ZM': 'Zoom Communications (S&P 500)',
            'NFLX': 'Netflix Inc (S&P 500)', 'WBA': 'Walgreens Alliance (S&P 500)',
            'INTC': 'Intel Corporation (S&P 500)', 'PARA': 'Paramount Global (S&P 500)',
            'PAYC': 'Paycom Software (S&P 500)', 'F': 'Ford Motor Company (S&P 500)'
        }
        
    def get_available_stocks(self):
        """Get list of available stocks from data files"""
        data_dir = self.project_root / "data/internal"
        stock_files = list(data_dir.glob("fundamentals_*.csv"))
        
        stocks = []
        for file in stock_files:
            if file.name != "financial_fundamentals_combined.csv":
                # Extract stock symbol from filename
                symbol = file.name.replace("fundamentals_", "").replace(".csv", "")
                stocks.append(symbol)
        
        return sorted(stocks)
    
    def display_stock_menu(self):
        """Display interactive stock selection menu"""
        print("\n" + "="*80)
        print("🎯 STOCK ANALYSIS & FORECASTING TOOL")
        print("="*80)
        print("Choose a stock to analyze:")
        print()
        
        # Group by market
        ftse_stocks = [s for s in self.available_stocks if s.endswith('_L')]
        sp500_stocks = [s for s in self.available_stocks if not s.endswith('_L')]
        
        print("📈 FTSE 100 STOCKS:")
        print("-" * 40)
        for i, stock in enumerate(ftse_stocks, 1):
            name = self.stock_names.get(stock, stock)
            print(f"{i:2d}. {stock:10s} - {name}")
        
        print("\n📊 S&P 500 STOCKS:")
        print("-" * 40)
        start_num = len(ftse_stocks) + 1
        for i, stock in enumerate(sp500_stocks, start_num):
            name = self.stock_names.get(stock, stock)
            print(f"{i:2d}. {stock:10s} - {name}")
        
        print(f"\n{len(self.available_stocks)+1:2d}. Exit")
        print("="*80)
        
        while True:
            try:
                choice = input("\nEnter your choice (number): ").strip()
                choice_num = int(choice)
                
                if choice_num == len(self.available_stocks) + 1:
                    print("👋 Goodbye!")
                    return None
                elif 1 <= choice_num <= len(self.available_stocks):
                    selected_stock = self.available_stocks[choice_num - 1]
                    return selected_stock
                else:
                    print(f"❌ Please enter a number between 1 and {len(self.available_stocks)+1}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def load_stock_data(self, symbol):
        """Load all available data for a stock"""
        print(f"\n📊 Loading data for {symbol}...")
        
        data = {}
        
        # Load fundamentals data
        fundamentals_file = self.project_root / f"data/internal/fundamentals_{symbol}.csv"
        if fundamentals_file.exists():
            data['fundamentals'] = pd.read_csv(fundamentals_file)
            data['fundamentals']['date'] = pd.to_datetime(data['fundamentals']['date'])
            print(f"✅ Loaded fundamentals: {len(data['fundamentals'])} records")
        
        # Try to load stock price data from various locations
        possible_locations = [
            f"data/ftse100/{symbol}.csv",
            f"data/sp500/{symbol}.csv", 
            f"data/raw/{symbol}.csv",
            f"data/{symbol}.csv"
        ]
        
        for location in possible_locations:
            price_file = self.project_root / location
            if price_file.exists():
                data['prices'] = pd.read_csv(price_file)
                data['prices']['Date'] = pd.to_datetime(data['prices']['Date'])
                print(f"✅ Loaded price data: {len(data['prices'])} records")
                break
        
        # Load economic data for context
        econ_file = self.project_root / "data/external/economic_combined.csv"
        if econ_file.exists():
            data['economic'] = pd.read_csv(econ_file)
            data['economic']['date'] = pd.to_datetime(data['economic']['date'])
            print(f"✅ Loaded economic data: {len(data['economic'])} records")
        
        return data
    
    def analyze_stock(self, symbol, data):
        """Perform comprehensive stock analysis"""
        print(f"\n🔍 ANALYZING {symbol.upper()}")
        print("="*60)
        
        name = self.stock_names.get(symbol, symbol)
        print(f"Company: {name}")
        
        if 'fundamentals' in data:
            fund_data = data['fundamentals']
            print(f"Data Period: {fund_data['date'].min().strftime('%Y-%m-%d')} to {fund_data['date'].max().strftime('%Y-%m-%d')}")
            
            # Latest metrics
            latest = fund_data.iloc[-1]
            print(f"\n📊 LATEST METRICS (as of {latest['date'].strftime('%Y-%m-%d')}):")
            print("-" * 50)
            
            key_metrics = ['stock_price', 'revenue', 'net_income', 'total_assets', 'market_cap']
            for metric in key_metrics:
                if metric in fund_data.columns:
                    value = latest[metric]
                    if pd.notna(value):
                        if metric == 'stock_price':
                            print(f"Stock Price: ${value:,.2f}")
                        elif metric in ['revenue', 'net_income', 'total_assets', 'market_cap']:
                            print(f"{metric.replace('_', ' ').title()}: ${value:,.0f}")
            
            # Performance metrics
            if 'stock_price' in fund_data.columns:
                prices = fund_data['stock_price'].dropna()
                if len(prices) > 1:
                    price_change = prices.iloc[-1] - prices.iloc[0]
                    price_change_pct = (price_change / prices.iloc[0]) * 100
                    print(f"\n📈 PERFORMANCE:")
                    print(f"Total Price Change: ${price_change:,.2f} ({price_change_pct:+.1f}%)")
                    
                    # Recent volatility
                    recent_std = prices.tail(30).std()
                    print(f"Recent Volatility (30d): ${recent_std:.2f}")
    
    def create_stock_visualization(self, symbol, data):
        """Create comprehensive visualization for the stock"""
        print(f"\n📈 Creating visualizations for {symbol}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'📊 Analysis: {self.stock_names.get(symbol, symbol)} ({symbol})', fontsize=16, fontweight='bold')
        
        if 'fundamentals' in data:
            fund_data = data['fundamentals']
            
            # 1. Stock Price Over Time
            ax1 = axes[0, 0]
            if 'stock_price' in fund_data.columns:
                price_data = fund_data[['date', 'stock_price']].dropna()
                ax1.plot(price_data['date'], price_data['stock_price'], linewidth=2, color='blue')
                ax1.set_title(f'Stock Price History')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price ($)')
                ax1.grid(True, alpha=0.3)
                
                # Simple forecast
                if len(price_data) > 10:
                    forecast = self.generate_simple_forecast(price_data)
                    if forecast:
                        ax1.plot(forecast['dates'], forecast['prices'], 
                                '--', color='red', linewidth=2, label='Forecast')
                        ax1.legend()
            else:
                ax1.text(0.5, 0.5, 'Stock Price Data\nNot Available', ha='center', va='center')
                ax1.set_title('Stock Price History')
            
            # 2. Revenue Trend
            ax2 = axes[0, 1]
            if 'revenue' in fund_data.columns:
                revenue_data = fund_data[['date', 'revenue']].dropna()
                ax2.plot(revenue_data['date'], revenue_data['revenue'] / 1e9, marker='o')
                ax2.set_title('Revenue Trend')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Revenue (Billions $)')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Revenue Data\nNot Available', ha='center', va='center')
                ax2.set_title('Revenue Trend')
            
            # 3. Profitability
            ax3 = axes[1, 0]
            if 'net_income' in fund_data.columns:
                profit_data = fund_data[['date', 'net_income']].dropna()
                colors = ['red' if x < 0 else 'green' for x in profit_data['net_income']]
                ax3.scatter(profit_data['date'], profit_data['net_income'] / 1e9, c=colors)
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.set_title('Net Income')
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Net Income (Billions $)')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Profitability Data\nNot Available', ha='center', va='center')
                ax3.set_title('Net Income')
            
            # 4. Market Cap
            ax4 = axes[1, 1]
            if 'market_cap' in fund_data.columns:
                market_data = fund_data[['date', 'market_cap']].dropna()
                ax4.plot(market_data['date'], market_data['market_cap'] / 1e9, color='purple')
                ax4.set_title('Market Capitalization')
                ax4.set_xlabel('Date')
                ax4.set_ylabel('Market Cap (Billions $)')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Market Cap\nData Not Available', ha='center', va='center')
                ax4.set_title('Market Capitalization')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = self.project_root / "data/processed/reports/individual_stocks"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"{symbol}_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"✅ Visualization saved to: {output_path}")
        plt.show()
        
        return output_path
    
    def generate_simple_forecast(self, price_data, days=30):
        """Generate simple forecast"""
        try:
            prices = price_data['stock_price'].values
            dates = price_data['date'].values
            
            if len(prices) < 5:
                return None
            
            # Simple linear trend
            x = np.arange(len(prices))
            coeffs = np.polyfit(x, prices, 1)
            
            # Generate future dates and prices
            last_date = pd.to_datetime(dates[-1])
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
            
            future_x = np.arange(len(prices), len(prices) + days)
            future_prices = np.polyval(coeffs, future_x)
            
            return {
                'dates': future_dates,
                'prices': future_prices
            }
        except:
            return None
    
    def display_data_summary(self, symbol, data):
        """Display comprehensive data summary"""
        print(f"\n📋 DATA SUMMARY FOR {symbol.upper()}")
        print("="*60)
        
        name = self.stock_names.get(symbol, symbol)
        print(f"Company: {name}")
        
        for data_type, df in data.items():
            if isinstance(df, pd.DataFrame):
                print(f"\n{data_type.upper()} DATA:")
                print(f"  Records: {len(df):,}")
                print(f"  Columns: {len(df.columns)}")
                
                if 'date' in df.columns:
                    date_col = 'date'
                elif 'Date' in df.columns:
                    date_col = 'Date'
                else:
                    date_col = None
                
                if date_col:
                    print(f"  Date Range: {df[date_col].min()} to {df[date_col].max()}")
                
                # Show key columns
                if data_type == 'fundamentals':
                    key_cols = ['stock_price', 'revenue', 'net_income', 'market_cap', 'pe_ratio']
                    available_cols = [col for col in key_cols if col in df.columns]
                    if available_cols:
                        print(f"  Key Metrics: {', '.join(available_cols)}")

def main():
    """Main execution function"""
    analyzer = StockAnalyzer()
    
    while True:
        # Display menu and get user selection
        selected_stock = analyzer.display_stock_menu()
        
        if selected_stock is None:
            break
        
        try:
            # Load all data for the selected stock
            data = analyzer.load_stock_data(selected_stock)
            
            if not data:
                print(f"❌ No data found for {selected_stock}")
                continue
            
            # Display data summary
            analyzer.display_data_summary(selected_stock, data)
            
            # Perform analysis
            analyzer.analyze_stock(selected_stock, data)
            
            # Create comprehensive visualization
            analyzer.create_stock_visualization(selected_stock, data)
            
            # Ask if user wants to continue
            print("\n" + "="*60)
            continue_choice = input("Would you like to analyze another stock? (y/n): ").strip().lower()
            
            if continue_choice not in ['y', 'yes']:
                print("👋 Thank you for using the Stock Analysis Tool!")
                break
                
        except Exception as e:
            print(f"❌ Error analyzing {selected_stock}: {e}")
            print("Please try another stock.")

if __name__ == "__main__":
    main() 