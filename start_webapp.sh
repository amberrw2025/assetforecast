#!/bin/bash

# Forecast Assessment Web Application Startup Script
# This script starts the Flask web application with proper error handling

echo "🚀 Starting Forecast Assessment Web Application..."
echo "============================================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python3 -c "import flask, plotly, pandas, numpy, joblib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing required packages. Installing..."
    pip3 install -r requirements_webapp.txt
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "❌ Models directory not found. Please run model training first:"
    echo "   python3 model_training_pipeline.py"
    exit 1
fi

# Start the web application
echo "✅ All checks passed. Starting web application..."
echo "📱 The application will be available at: http://localhost:5001"
echo "🛑 Press Ctrl+C to stop the server"
echo "============================================================"

python3 run_webapp.py 