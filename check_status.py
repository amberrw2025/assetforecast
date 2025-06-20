#!/usr/bin/env python3
"""
Status check script for the Forecast Assessment Web Application.
Verifies that all components are working correctly.
"""

import os
import sys
import requests
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed."""
    print("🔍 Checking Python packages...")
    
    required_packages = [
        'flask', 'plotly', 'pandas', 'numpy', 'joblib',
        'sklearn', 'prophet', 'statsmodels', 'tensorflow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip3 install -r requirements_webapp.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_models():
    """Check if trained models exist."""
    print("\n🔍 Checking trained models...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    required_models = ['arima.joblib', 'prophet.joblib', 'lstm.joblib']
    missing_models = []
    
    for model in required_models:
        model_path = models_dir / model
        if model_path.exists():
            print(f"  ✅ {model}")
        else:
            print(f"  ❌ {model}")
            missing_models.append(model)
    
    if missing_models:
        print(f"\n❌ Missing models: {', '.join(missing_models)}")
        print("Train models with: python3 model_training_pipeline.py")
        return False
    
    print("✅ All trained models are available")
    return True

def check_webapp():
    """Check if web application is running."""
    print("\n🔍 Checking web application...")
    
    try:
        response = requests.get("http://localhost:5001", timeout=5)
        if response.status_code == 200:
            print("✅ Web application is running on http://localhost:5001")
            return True
        else:
            print(f"❌ Web application returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Web application is not running")
        print("Start with: python3 run_webapp.py")
        return False
    except Exception as e:
        print(f"❌ Error checking web application: {e}")
        return False

def main():
    """Main status check function."""
    print("🚀 Forecast Assessment Web Application - Status Check")
    print("=" * 60)
    
    # Check Python packages
    packages_ok = check_python_packages()
    
    # Check models
    models_ok = check_models()
    
    # Check web application
    webapp_ok = check_webapp()
    
    print("\n" + "=" * 60)
    if packages_ok and models_ok and webapp_ok:
        print("🎉 All systems are operational!")
        print("📱 Access your web application at: http://localhost:5001")
    else:
        print("⚠️  Some issues detected. Please fix them before using the application.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 