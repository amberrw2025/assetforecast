#!/usr/bin/env python3
"""
Web Application Runner for Forecast Accuracy Assessment Model.
Starts the Flask web server with the interactive dashboard.
"""

import os
import sys
from pathlib import Path
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

def main():
    """Main function to run the web application."""
    print("🚀 Starting Forecast Assessment Web Application...")
    print("=" * 60)
    
    try:
        # Import and create app
        from webapp.app import create_app
        
        app = create_app()
        
        print("✅ Web application created successfully")
        print("🌐 Starting server...")
        print("📱 Open your browser and go to: http://localhost:5003")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run the app
        logger.info("Starting server...")
        # Bind to 0.0.0.0 to make it accessible on the network
        app.run(debug=True, host='0.0.0.0', port=5003)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("  python3 -m pip install flask plotly")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting web application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 