#!/bin/bash
# This script navigates to the correct directory and starts the web app.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to the webapp's directory
cd "$SCRIPT_DIR/fste_and_sandp_forcaster"

# Start the web app
echo "✅ Navigated to the correct directory."
echo "🚀 Starting the web application..."
echo "----------------------------------------"
python3 run_webapp.py

# Run the pipeline
echo "----------------------------------------"
echo "🚀 Running the pipeline..."
python3 fste_and_sandp_forcaster/run_pipeline.py 