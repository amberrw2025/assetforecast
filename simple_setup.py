#!/usr/bin/env python3
"""
Simple setup script for the Forecast Accuracy Assessment Model Pipeline.
This script provides detailed error information and troubleshooting steps.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command_with_details(command, description):
    """Run a command and provide detailed output."""
    print(f"🔄 {description}...")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout[:200]}...")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Return code: {e.returncode}")
        print(f"   Error output: {e.stderr}")
        if e.stdout:
            print(f"   Standard output: {e.stdout}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python version is compatible")
    return True

def check_pip():
    """Check if pip is available and working."""
    print("📦 Checking pip installation...")
    
    try:
        result = subprocess.run(
            ["pip", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ Pip is available: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Pip not found or not working: {e}")
        return False

def install_dependencies_step_by_step():
    """Install dependencies one by one to identify issues."""
    print("📦 Installing dependencies step by step...")
    
    # Core dependencies first
    core_deps = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0"
    ]
    
    print("   Installing core dependencies...")
    for dep in core_deps:
        if not run_command_with_details(f"pip install {dep}", f"Installing {dep}"):
            print(f"❌ Failed to install {dep}")
            return False
    
    # Financial data dependencies
    financial_deps = [
        "yfinance>=0.2.18"
    ]
    
    print("   Installing financial data dependencies...")
    for dep in financial_deps:
        if not run_command_with_details(f"pip install {dep}", f"Installing {dep}"):
            print(f"❌ Failed to install {dep}")
            return False
    
    # Visualization dependencies
    viz_deps = [
        "plotly>=5.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    print("   Installing visualization dependencies...")
    for dep in viz_deps:
        if not run_command_with_details(f"pip install {dep}", f"Installing {dep}"):
            print(f"❌ Failed to install {dep}")
            return False
    
    # Optional dependencies (don't fail if these don't install)
    optional_deps = [
        "fredapi>=0.5.0",
        "pytrends>=4.9.0",
        "tweepy>=4.14.0",
        "praw>=7.7.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "prophet>=1.1.4",
        "statsmodels>=0.14.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "streamlit>=1.28.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0"
    ]
    
    print("   Installing optional dependencies...")
    failed_optional = []
    for dep in optional_deps:
        if not run_command_with_details(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install optional dependency: {dep}")
            failed_optional.append(dep)
    
    if failed_optional:
        print(f"⚠️  {len(failed_optional)} optional dependencies failed to install:")
        for dep in failed_optional:
            print(f"   - {dep}")
        print("   The pipeline will still work with sample data.")
    
    return True

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
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
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
        try:
            # Copy example to .env
            with open(env_example, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your API keys (optional)")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("⚠️  env_example.txt not found. Please create .env file manually.")
        return False

def test_basic_imports():
    """Test that basic modules can be imported."""
    print("🧪 Testing basic module imports...")
    
    basic_modules = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("requests", "requests")
    ]
    
    for module_name, import_name in basic_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            return False
    
    return True

def test_project_imports():
    """Test that project modules can be imported."""
    print("🧪 Testing project module imports...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path.cwd()))
        
        # Test config import
        from config import COMPANIES, FINANCIAL_METRICS
        print(f"✅ Configuration loaded: {len(COMPANIES)} markets configured")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import project modules: {e}")
        print("   This might be due to missing dependencies or file issues.")
        return False

def main():
    """Main setup function with detailed error reporting."""
    print("="*60)
    print("FORECAST MODEL PIPELINE - SIMPLE SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        print("   Please upgrade to Python 3.8 or higher")
        return False
    
    # Check pip
    if not check_pip():
        print("\n❌ Setup failed: Pip not available")
        print("   Please install pip or ensure it's in your PATH")
        return False
    
    # Install dependencies step by step
    if not install_dependencies_step_by_step():
        print("\n❌ Setup failed: Could not install core dependencies")
        print("   Please check your internet connection and pip configuration")
        return False
    
    # Create directories
    if not create_directories():
        print("\n❌ Setup failed: Could not create directories")
        print("   Please check file permissions")
        return False
    
    # Setup environment file
    setup_environment_file()
    
    # Test basic imports
    if not test_basic_imports():
        print("\n❌ Setup failed: Basic imports failed")
        print("   Please check your Python environment")
        return False
    
    # Test project imports
    if not test_project_imports():
        print("\n⚠️  Project imports failed, but basic setup is complete")
        print("   You can still run basic functionality")
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 'python test_pipeline.py' to verify everything works")
    print("2. Run 'python run_pipeline.py' to start the full pipeline")
    print("3. Edit .env file with your API keys (optional)")
    print("\nFor troubleshooting, see the detailed output above.")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 