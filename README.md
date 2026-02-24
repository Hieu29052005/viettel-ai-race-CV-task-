# 3D Parcel Detection Pipeline

This is the code that helps me reach in top 15 in the private test for the Computer Vision task in Viettel AI Race 2025

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py --config config/config.yaml
```

### Configuration
Edit `config/config.yaml` to customize:
- Camera intrinsics
- ROI (Region of Interest)
- Detection parameters
- Depth processing strategy
- Point cloud parameters

## Project Structure

```
project/
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── models/
│   │   └── detector.py      # YOLO detector
│   ├── processors/
│   │   ├── depth_processor.py      # Depth processing
│   │   ├── pointcloud_processor.py # Point cloud processing
│   │   └── postprocessor.py        # Post-processing
│   ├── utils/
│   │   ├── geometry.py      # Geometry utilities
│   │   ├── io_utils.py      # I/O utilities
│   │   └── visualization.py # Visualization tools
│   └── pipeline.py          # Main pipeline
├── main.py                  # Entry point
└── requirements.txt         # Dependencies
```

## Output

- `Submission_3D.csv`: Final results

## Key Features

- **Modular Design**: Easy to maintain and extend
- **Robust Depth Processing**: Multi-strategy depth estimation
- **Flexible Configuration**: YAML-based config
- **Debug Visualization**: Automatic debug output
- **Error Handling**: Graceful fallbacks


For accessing the dataset, you can contact me via email: nguyenvuongtrunghieuu9@gmail.com

## License
MIT
