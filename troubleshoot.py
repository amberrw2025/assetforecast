#!/usr/bin/env python3
"""
Troubleshooting script for the Forecast Accuracy Assessment Model Pipeline.
This script helps identify and fix common setup issues.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_system_info():
    """Check system information."""
    print("🖥️  System Information:")
    print(f"   Python version: {sys.version}")
    print(f"   Platform: {sys.platform}")
    print(f"   Architecture: {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   Virtual environment: Yes")
    else:
        print("   Virtual environment: No")

def check_pip_info():
    """Check pip information."""
    print("\n📦 Pip Information:")
    
    try:
        result = subprocess.run(
            ["pip", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"   Pip version: {result.stdout.strip()}")
    except Exception as e:
        print(f"   Pip error: {e}")
        return False
    
    try:
        result = subprocess.run(
            ["pip", "list"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        installed_packages = result.stdout
        print(f"   Installed packages: {len(installed_packages.splitlines()) - 2} packages")
    except Exception as e:
        print(f"   Could not list packages: {e}")
    
    return True

def test_individual_packages():
    """Test installation of individual packages."""
    print("\n🧪 Testing Individual Packages:")
    
    packages = [
        ("pandas", "Core data manipulation"),
        ("numpy", "Numerical computing"),
        ("yfinance", "Financial data"),
        ("requests", "HTTP requests"),
        ("plotly", "Data visualization"),
        ("python-dotenv", "Environment variables"),
        ("tqdm", "Progress bars")
    ]
    
    results = {}
    
    for package, description in packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
            results[package] = True
        except ImportError:
            print(f"   ❌ {package} - {description} (not installed)")
            results[package] = False
    
    return results

def try_install_package(package, description):
    """Try to install a specific package."""
    print(f"\n🔄 Installing {package} ({description})...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"   ✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install {package}")
        print(f"   Error: {e.stderr}")
        return False

def suggest_solutions():
    """Suggest solutions based on common issues."""
    print("\n💡 Common Solutions:")
    print("1. If you're on macOS and getting permission errors:")
    print("   Try: pip install --user package_name")
    print("   Or: sudo pip install package_name")
    
    print("\n2. If you're getting SSL errors:")
    print("   Try: pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org package_name")
    
    print("\n3. If you're behind a proxy:")
    print("   Try: pip install --proxy http://proxy:port package_name")
    
    print("\n4. If you want to use a virtual environment:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # On macOS/Linux")
    print("   venv\\Scripts\\activate     # On Windows")
    print("   pip install -r requirements.txt")
    
    print("\n5. If you want to upgrade pip first:")
    print("   python -m pip install --upgrade pip")

def main():
    """Main troubleshooting function."""
    print("="*60)
    print("FORECAST MODEL PIPELINE - TROUBLESHOOTING")
    print("="*60)
    
    # Check system info
    check_system_info()
    
    # Check pip info
    if not check_pip_info():
        print("\n❌ Pip is not working properly")
        suggest_solutions()
        return False
    
    # Test individual packages
    package_results = test_individual_packages()
    
    # Count missing packages
    missing_packages = [pkg for pkg, installed in package_results.items() if not installed]
    
    if missing_packages:
        print(f"\n⚠️  {len(missing_packages)} packages are missing:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        
        print("\n🔄 Attempting to install missing packages...")
        
        for pkg in missing_packages:
            if not try_install_package(pkg, "Essential package"):
                print(f"   ⚠️  Could not install {pkg}")
        
        # Test again after installation attempts
        print("\n🧪 Re-testing packages...")
        package_results = test_individual_packages()
        missing_packages = [pkg for pkg, installed in package_results.items() if not installed]
    
    if not missing_packages:
        print("\n🎉 All essential packages are installed!")
        print("   You can now run the pipeline.")
    else:
        print(f"\n⚠️  {len(missing_packages)} packages are still missing:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        
        print("\n💡 The pipeline will still work with sample data for missing packages.")
        suggest_solutions()
    
    print("\n" + "="*60)
    return True

if __name__ == "__main__":
    main() 