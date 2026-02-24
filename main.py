import argparse
from pathlib import Path
from src.pipeline import Pipeline
from src.utils import IOUtils


def main():
    parser = argparse.ArgumentParser(description='3D Parcel Detection Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = IOUtils.load_config(args.config)
    
    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()