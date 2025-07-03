#!/usr/bin/env python3
"""
Test Webapp Updates
==================

Test script to verify the webapp updates work correctly.
"""

import os
import sys
from pathlib import Path

def test_webapp_updates():
    """Test the webapp updates"""
    print("🚀 TESTING UPDATED WEBAPP")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('webapp').exists():
        print("❌ webapp directory not found")
        return False
    
    # Test webapp import
    try:
        sys.path.append('webapp')
        from app import create_app
        app = create_app()
        print("✅ Webapp loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load webapp: {e}")
        return False
    
    # Check plot directories
    plot_dirs = [
        'forecast_plots_2024',
        'improved_forecast_plots_2024', 
        'unified_mape_plots_2024',
        'comprehensive_plots_2024'
    ]
    
    print("\n📁 Plot Directory Status:")
    for dir_name in plot_dirs:
        if Path(dir_name).exists():
            file_count = len(list(Path(dir_name).glob('*.png')))
            print(f"   ✅ {dir_name}/: {file_count} PNG files")
        else:
            print(f"   ❌ {dir_name}/: Directory not found")
    
    # Check comprehensive plots specifically
    comprehensive_dir = Path('comprehensive_plots_2024')
    if comprehensive_dir.exists():
        option1_dir = comprehensive_dir / 'option1_standard'
        option2_dir = comprehensive_dir / 'option2_mape_enhanced'
        
        option1_count = len(list(option1_dir.glob('*.png'))) if option1_dir.exists() else 0
        option2_count = len(list(option2_dir.glob('*.png'))) if option2_dir.exists() else 0
        
        print(f"\n📊 Comprehensive Plots:")
        print(f"   📁 Option 1 (Standard): {option1_count} plots")
        print(f"   🌟 Option 2 (MAPE Enhanced): {option2_count} plots")
    
    # Check templates
    templates_dir = Path('webapp/templates')
    required_templates = ['forecast_plots.html', 'comprehensive_plots.html']
    
    print(f"\n📄 Template Status:")
    for template in required_templates:
        template_path = templates_dir / template
        if template_path.exists():
            print(f"   ✅ {template}: Found")
        else:
            print(f"   ❌ {template}: Missing")
    
    print(f"\n✅ Webapp update test completed!")
    return True

if __name__ == "__main__":
    test_webapp_updates() 