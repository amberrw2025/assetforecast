#!/usr/bin/env python3
"""
Simple runner script for the Forecast Accuracy Assessment Model Pipeline.
This script provides an easy way to execute the pipeline with different options.
"""

import sys
import argparse
from pathlib import Path
import config  # Load environment variables

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from main_pipeline import ForecastModelPipeline
from utils import get_logger

logger = get_logger("pipeline_runner")

def main():
    """
    Main function to run the pipeline with command line options.
    """
    parser = argparse.ArgumentParser(
        description="Forecast Accuracy Assessment Model Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run complete pipeline
  python run_pipeline.py --data-only        # Run only data acquisition
  python run_pipeline.py --clean-only       # Run only data cleaning
  python run_pipeline.py --analysis-only    # Run only analysis
  python run_pipeline.py --no-dvc           # Run without DVC setup
        """
    )
    
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Run only the data acquisition phase'
    )
    
    parser.add_argument(
        '--clean-only',
        action='store_true',
        help='Run only the data cleaning phase'
    )
    
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Run only the exploratory analysis phase'
    )
    
    parser.add_argument(
        '--no-dvc',
        action='store_true',
        help='Skip DVC setup and tracking'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print("="*60)
    print("FORECAST ACCURACY ASSESSMENT MODEL PIPELINE")
    print("="*60)
    print("This pipeline will collect, clean, and prepare data for forecasting models.")
    print("Starting execution...\n")
    
    # Create pipeline instance
    pipeline = ForecastModelPipeline()
    
    try:
        if args.data_only:
            logger.info("Running data acquisition phase only")
            success = pipeline.run_data_acquisition()
            
        elif args.clean_only:
            logger.info("Running data cleaning phase only")
            success = pipeline.run_data_cleaning()
            
        elif args.analysis_only:
            logger.info("Running exploratory analysis phase only")
            success = pipeline.run_exploratory_analysis()
            
        else:
            logger.info("Running complete pipeline")
            success = pipeline.run_complete_pipeline()
            
            # Setup DVC unless disabled
            if success and not args.no_dvc:
                logger.info("Setting up DVC tracking")
                pipeline.setup_dvc_tracking()
        
        if success:
            print("\n" + "="*60)
            print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("="*60)
            print("📊 Check the following directories for results:")
            print("   - data/raw/     : Raw collected data")
            print("   - data/processed/: Cleaned data and visualizations")
            print("   - logs/         : Execution logs")
            print("\n🚀 Ready for the next phase: Model Training!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ PIPELINE EXECUTION FAILED")
            print("="*60)
            print("📋 Check the logs for detailed error information:")
            print("   - logs/pipeline.log")
            print("\n🔧 Common troubleshooting steps:")
            print("   1. Verify API keys are configured in .env file")
            print("   2. Check internet connection")
            print("   3. Ensure all dependencies are installed")
            print("   4. Review the README.md for setup instructions")
            print("="*60)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline execution interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Check the logs for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 