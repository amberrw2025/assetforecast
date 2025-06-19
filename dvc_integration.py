"""
DVC (Data Version Control) integration for the Forecast Accuracy Assessment Model Pipeline.
This module provides guidance and structure for implementing DVC to track changes in datasets.
"""

import subprocess
import os
from pathlib import Path
from typing import List, Dict, Optional
import json

from config import DVC_CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger("dvc_integration")

class DVCManager:
    """
    Manages DVC operations for data versioning in the forecast model pipeline.
    """
    
    def __init__(self):
        self.dvc_config = DVC_CONFIG
        self.project_root = Path(__file__).parent
        
    def initialize_dvc(self) -> bool:
        """
        Initialize DVC in the project directory.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if DVC is already initialized
            if (self.project_root / ".dvc").exists():
                logger.info("DVC is already initialized")
                return True
            
            # Initialize DVC
            result = subprocess.run(
                ["dvc", "init"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("DVC initialized successfully")
                return True
            else:
                logger.error(f"Failed to initialize DVC: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("DVC is not installed. Please install with: pip install dvc")
            return False
        except Exception as e:
            logger.error(f"Error initializing DVC: {str(e)}")
            return False
    
    def add_remote_storage(self, remote_name: str = "default", 
                          remote_url: str = None) -> bool:
        """
        Add remote storage for DVC.
        
        Args:
            remote_name (str): Name of the remote
            remote_url (str): URL of the remote storage
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if remote_url is None:
                remote_url = self.dvc_config["remote_storage"]
            
            # Add remote
            result = subprocess.run(
                ["dvc", "remote", "add", remote_name, remote_url],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Remote storage '{remote_name}' added successfully")
                return True
            else:
                logger.error(f"Failed to add remote storage: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding remote storage: {str(e)}")
            return False
    
    def add_data_file(self, file_path: str, description: str = None) -> bool:
        """
        Add a data file to DVC tracking.
        
        Args:
            file_path (str): Path to the data file
            description (str): Description of the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                logger.error(f"File not found: {full_path}")
                return False
            
            # Add file to DVC
            result = subprocess.run(
                ["dvc", "add", str(full_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Added {file_path} to DVC tracking")
                
                # Add .dvc file to git
                dvc_file = full_path.with_suffix('.dvc')
                if dvc_file.exists():
                    subprocess.run(
                        ["git", "add", str(dvc_file)],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True
                    )
                    logger.info(f"Added {dvc_file.name} to git")
                
                return True
            else:
                logger.error(f"Failed to add {file_path} to DVC: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding data file to DVC: {str(e)}")
            return False
    
    def add_multiple_data_files(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        Add multiple data files to DVC tracking.
        
        Args:
            file_paths (List[str]): List of file paths to add
            
        Returns:
            Dict[str, bool]: Dictionary mapping file paths to success status
        """
        results = {}
        
        for file_path in file_paths:
            results[file_path] = self.add_data_file(file_path)
        
        return results
    
    def push_data(self, remote_name: str = "default") -> bool:
        """
        Push data to remote storage.
        
        Args:
            remote_name (str): Name of the remote storage
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ["dvc", "push", "-r", remote_name],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Data pushed to remote '{remote_name}' successfully")
                return True
            else:
                logger.error(f"Failed to push data: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error pushing data: {str(e)}")
            return False
    
    def pull_data(self, remote_name: str = "default") -> bool:
        """
        Pull data from remote storage.
        
        Args:
            remote_name (str): Name of the remote storage
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ["dvc", "pull", "-r", remote_name],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Data pulled from remote '{remote_name}' successfully")
                return True
            else:
                logger.error(f"Failed to pull data: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling data: {str(e)}")
            return False
    
    def list_tracked_files(self) -> List[str]:
        """
        List all files tracked by DVC.
        
        Returns:
            List[str]: List of tracked file paths
        """
        try:
            result = subprocess.run(
                ["dvc", "list", "."],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                return [f for f in files if f]
            else:
                logger.error(f"Failed to list tracked files: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing tracked files: {str(e)}")
            return []
    
    def get_file_status(self, file_path: str) -> Dict[str, str]:
        """
        Get the status of a tracked file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Dict[str, str]: Status information
        """
        try:
            result = subprocess.run(
                ["dvc", "status", file_path],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            status_info = {
                "file": file_path,
                "status": "unknown",
                "details": result.stdout
            }
            
            if result.returncode == 0:
                if "up to date" in result.stdout.lower():
                    status_info["status"] = "up_to_date"
                elif "modified" in result.stdout.lower():
                    status_info["status"] = "modified"
                elif "new" in result.stdout.lower():
                    status_info["status"] = "new"
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting file status: {str(e)}")
            return {"file": file_path, "status": "error", "details": str(e)}
    
    def create_data_pipeline(self) -> bool:
        """
        Create a DVC pipeline for the data processing workflow.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create dvc.yaml file
            pipeline_config = {
                "stages": {
                    "data_collection": {
                        "cmd": "python -m data_acquisition.financial_data && python -m data_acquisition.economic_data && python -m data_acquisition.sentiment_data",
                        "deps": ["data_acquisition/"],
                        "outs": ["data/raw/"]
                    },
                    "data_cleaning": {
                        "cmd": "python -m data_processing.data_cleaner",
                        "deps": ["data/raw/", "data_processing/"],
                        "outs": ["data/processed/"]
                    }
                }
            }
            
            pipeline_file = self.project_root / "dvc.yaml"
            with open(pipeline_file, 'w') as f:
                json.dump(pipeline_config, f, indent=2)
            
            logger.info("DVC pipeline configuration created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating DVC pipeline: {str(e)}")
            return False
    
    def run_pipeline(self, stage: str = None) -> bool:
        """
        Run the DVC pipeline.
        
        Args:
            stage (str): Specific stage to run (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cmd = ["dvc", "repro"]
            if stage:
                cmd.append(stage)
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"DVC pipeline executed successfully")
                return True
            else:
                logger.error(f"Failed to run DVC pipeline: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running DVC pipeline: {str(e)}")
            return False

def setup_dvc_for_project():
    """
    Setup DVC for the entire project with all necessary configurations.
    """
    logger.info("Setting up DVC for the forecast model project")
    
    dvc_manager = DVCManager()
    
    # Initialize DVC
    if not dvc_manager.initialize_dvc():
        logger.error("Failed to initialize DVC")
        return False
    
    # Add remote storage (configure this with your actual storage)
    if not dvc_manager.add_remote_storage():
        logger.warning("Failed to add remote storage. You can add it manually later.")
    
    # Add data files to tracking
    data_files = [
        "data/raw/company_financials.csv",
        "data/raw/economic_indicators.csv",
        "data/raw/sentiment_data.csv",
        "data/processed/cleaned_dataset.csv"
    ]
    
    results = dvc_manager.add_multiple_data_files(data_files)
    
    success_count = sum(results.values())
    logger.info(f"Successfully added {success_count}/{len(data_files)} files to DVC tracking")
    
    # Create pipeline
    dvc_manager.create_data_pipeline()
    
    logger.info("DVC setup completed. Remember to:")
    logger.info("1. Configure your remote storage in dvc.yaml")
    logger.info("2. Run 'dvc push' to upload data to remote storage")
    logger.info("3. Commit .dvc files to git for version control")
    
    return True

def main():
    """
    Main function to demonstrate DVC integration.
    """
    logger.info("DVC integration module loaded successfully")
    logger.info("Use setup_dvc_for_project() to initialize DVC for the entire project")

if __name__ == "__main__":
    main() 