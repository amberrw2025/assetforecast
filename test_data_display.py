#!/usr/bin/env python3
"""
Test script to verify that stock data is being returned correctly
"""
import requests
import json

def test_stock_data(symbol):
    """Test data for a specific stock"""
    try:
        response = requests.get(f'http://localhost:5001/api/stock/{symbol}')
        data = response.json()
        
        if not data.get('success'):
            print(f"❌ {symbol}: API Error - {data.get('error')}")
            return False
            
        stock_data = data['data']
        chart_data = stock_data.get('chart_data', {})
        metrics = stock_data.get('latest_metrics', {})
        forecast = stock_data.get('forecast', {})
        
        print(f"✅ {symbol} ({stock_data['name']}):")
        print(f"   📊 Chart data: {list(chart_data.keys())}")
        print(f"   💰 Price data points: {len(chart_data.get('prices', {}).get('dates', []))}")
        print(f"   📈 Current price: ${metrics.get('close_price', 0):.2f}")
        print(f"   🔮 Forecast available: {'Yes' if forecast else 'No'}")
        if forecast:
            print(f"   📊 30-day forecast: ${forecast.get('current_price', 0):.2f} → ${forecast.get('forecast_price', 0):.2f}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ {symbol}: Connection Error - {e}")
        return False

def test_stocks_list():
    """Test the stocks list API"""
    try:
        response = requests.get('http://localhost:5001/api/stocks')
        data = response.json()
        
        if data.get('success'):
            stocks = data['stocks']
            print(f"✅ Stocks API: {len(stocks)} stocks available")
            
            ftse_stocks = [s for s in stocks if s['market'] == 'FTSE 100']
            sp500_stocks = [s for s in stocks if s['market'] == 'S&P 500']
            
            print(f"   🇬🇧 FTSE 100: {len(ftse_stocks)} stocks")
            print(f"   🇺🇸 S&P 500: {len(sp500_stocks)} stocks")
            print()
            return stocks
        else:
            print(f"❌ Stocks API Error: {data.get('error')}")
            return []
            
    except Exception as e:
        print(f"❌ Stocks API Connection Error: {e}")
        return []

if __name__ == "__main__":
    print("🧪 Testing Stock Data API...")
    print("=" * 50)
    
    # Test stocks list
    stocks = test_stocks_list()
    
    if stocks:
        # Test a few representative stocks
        test_stocks = ['NVDA', 'TSLA', 'AZN_L', 'TSCO_L']
        
        print("🔍 Testing individual stock data:")
        print("-" * 30)
        
        success_count = 0
        for symbol in test_stocks:
            if test_stock_data(symbol):
                success_count += 1
        
        print("=" * 50)
        print(f"📊 Results: {success_count}/{len(test_stocks)} stocks working correctly")
        
        if success_count == len(test_stocks):
            print("🎉 All tests passed! The web application should display data correctly.")
            print("🌐 Visit http://localhost:5001/stocks to see the interactive charts!")
        else:
            print("⚠️  Some issues detected. Check the errors above.")
    else:
        print("❌ Cannot test individual stocks - stocks list API failed") 