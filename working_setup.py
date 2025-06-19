#!/usr/bin/env python3
"""
Working setup script for the Forecast Accuracy Assessment Model Pipeline.
This script is specifically designed for macOS compatibility.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required Python packages using python3 -m pip."""
    print("📦 Installing Python dependencies...")
    
    # Check if requirements_minimal.txt exists
    requirements_file = Path("requirements_minimal.txt")
    if not requirements_file.exists():
        print("❌ requirements_minimal.txt not found")
        return False
    
    # Install dependencies using python3 -m pip
    return run_command("python3 -m pip install -r requirements_minimal.txt", "Installing minimal dependencies")

def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data",
        "data/raw",
        "data/processed", 
        "models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def setup_environment_file():
    """Create .env file from template."""
    print("🔧 Setting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys (optional)")
        return True
    else:
        print("⚠️  env_example.txt not found. Please create .env file manually.")
        return False

def test_imports():
    """Test that key modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        print("✅ Core dependencies imported successfully")
        
        # Test project modules
        sys.path.append(str(Path.cwd()))
        from config import COMPANIES, FINANCIAL_METRICS
        print("✅ Project modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def run_quick_test():
    """Run a quick test to verify setup."""
    print("🧪 Running quick test...")
    
    try:
        # Test configuration
        from config import COMPANIES
        print(f"✅ Configuration loaded: {len(COMPANIES)} markets configured")
        
        # Test data collector (with sample data)
        from data_acquisition.financial_data import FinancialDataCollector
        collector = FinancialDataCollector()
        print("✅ Financial data collector initialized")
        
        # Test data cleaner
        from data_processing.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        print("✅ Data cleaner initialized")
        
        # Test visualizer
        from utils.visualization import DataVisualizer
        visualizer = DataVisualizer()
        print("✅ Data visualizer initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("="*60)
    print("FORECAST MODEL PIPELINE - WORKING SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        print("❌ Setup failed: Incompatible Python version")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed: Could not install dependencies")
        return False
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed: Could not create directories")
        return False
    
    # Setup environment file
    setup_environment_file()
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed: Import test failed")
        return False
    
    # Run quick test
    if not run_quick_test():
        print("❌ Setup failed: Quick test failed")
        return False
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Edit .env file with your API keys (optional)")
    print("2. Run 'python3 test_pipeline.py' to verify everything works")
    print("3. Run 'python3 run_pipeline.py' to start the full pipeline")
    print("\nFor more information, see README.md")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 