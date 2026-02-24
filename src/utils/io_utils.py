import os
import re
import csv
import yaml
from pathlib import Path
from typing import Dict, List


class IOUtils:
    """Utility functions for file I/O"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load YAML configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def save_config(config: Dict, output_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @staticmethod
    def create_debug_folder(base_folder: str) -> str:
        """
        Create a new debug folder with incremental numbering
        
        Args:
            base_folder: Base debug folder path
            
        Returns:
            Path to new debug folder
        """
        os.makedirs(base_folder, exist_ok=True)
        
        subfolders = [f for f in os.listdir(base_folder) 
                     if os.path.isdir(os.path.join(base_folder, f)) 
                     and re.match(r"debug\d+", f)]
        
        if subfolders:
            numbers = [int(re.findall(r'\d+', f)[0]) for f in subfolders]
            next_number = max(numbers) + 1
        else:
            next_number = 1
        
        new_folder = os.path.join(base_folder, f"debug{next_number}")
        os.makedirs(new_folder, exist_ok=True)
        
        return new_folder
    
    @staticmethod
    def save_submission(results: List[Dict], output_file: str):
        """
        Save results to CSV submission file
        
        Args:
            results: List of result dictionaries
            output_file: Output CSV file path
        """
        fieldnames = ['image_filename', 'x', 'y', 'z', 'Rx', 'Ry', 'Rz']
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        print(f"Saved submission to {output_file}")