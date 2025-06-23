#!/usr/bin/env python3
"""
Quick script to open the stock analysis web application
"""
import webbrowser
import time
import subprocess
import sys

print("🚀 Opening Stock Analysis Web Application...")
print("📊 You can now analyze any of your 20 stocks:")
print("   • FTSE 100: AZN_L, LSEG_L, RKT_L, OCDO_L, CRDA_L, BT-A_L, VOD_L, SSE_L, GLEN_L, TSCO_L")
print("   • S&P 500: NVDA, TSLA, MRNA, ZM, NFLX, WBA, INTC, PARA, PAYC, F")
print()
print("🌐 Opening browser to: http://localhost:5001/stocks")
print("💡 Navigate to 'Stock Analysis' from the top menu if the direct link doesn't work")
print()

# Try to open the browser
try:
    webbrowser.open('http://localhost:5001/stocks')
    print("✅ Browser opened successfully!")
except Exception as e:
    print(f"❌ Could not open browser automatically: {e}")
    print("📋 Please manually navigate to: http://localhost:5001/stocks")

print()
print("📖 How to use the Stock Analysis feature:")
print("   1. Use the search box to find a specific stock")
print("   2. Filter by market (FTSE 100 or S&P 500)")
print("   3. Click on any stock to see detailed analysis")
print("   4. View price charts, revenue trends, and 30-day forecasts")
print("   5. Check key financial metrics and performance data")
print()
print("🔄 The application is running on http://localhost:5001")
print("🛑 To stop the application, press Ctrl+C in the terminal where it's running") 