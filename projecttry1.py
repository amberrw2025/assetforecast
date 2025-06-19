# 1. Setup environment
python setup.py

# 2. Test components
python test_pipeline.py

# 3. Run full pipeline
python run_pipeline.py

# Run specific phases
python run_pipeline.py --data-only        # Data acquisition only
python run_pipeline.py --clean-only       # Data cleaning only
python run_pipeline.py --analysis-only    # Analysis only
python run_pipeline.py --no-dvc           # Skip DVC setup
