# 3D Parcel Detection Pipeline

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
- `debug/debugN/`: Debug visualizations and config

## Key Features

- **Modular Design**: Easy to maintain and extend
- **Robust Depth Processing**: Multi-strategy depth estimation
- **Flexible Configuration**: YAML-based config
- **Debug Visualization**: Automatic debug output
- **Error Handling**: Graceful fallbacks

## License

MIT