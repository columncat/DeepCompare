# DeepCompare

![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)

## About the Project

DeepCompare is a computer vision project designed to detect missing or inappropriately assembled parts in car trim processes using CCTV footage. The system leverages deep learning embeddings to compare video frames against reference images for accurate scene identification and classification.

This project is a collaboration between Pusan National University and Renault Korea, developed by team ICU.

## Project Overview

- **Duration**: September 1, 2024 - June 30, 2025
- **Organization**: Pusan National University
- **Cooperation**: Renault Korea
- **Team**: ICU (Geonju Kim, Dongin Park, Taeju Park, Jinyeong Cho, Hyunsik Jeong)

## Features

- Detection of anomalies in car assembly lines via CCTV footage
- Deep learning-based image comparison using MobileNetV2 architecture
- Preprocessing techniques including ROI extraction and CLAHE enhancement
- Video processing with configurable frame-skipping for efficiency
- Detailed logging and CSV reporting for detection events
- GPU acceleration when available

## Technical Architecture

DeepCompare consists of three main components:

1. **DeepCompare.py**: Core module for image comparison
   - Uses MobileNetV2 pre-trained on ImageNet
   - Implements cosine distance metrics for image comparison
   - Applies preprocessing techniques
   - Generates probability-based classifications

2. **tnc_video.py**: Video processing module
   - Handles video I/O operations
   - Implements frame-skipping for performance
   - Detects vehicles with configurable thresholds
   - Logs results to CSV files

3. **data.py**: Data management module
   - Loads reference images for each class
   - Lists video files for processing
   - Provides utility functions

## Requirements

```
numpy==1.24.4
opencv-python==4.11.0.86
scipy==1.10.1
tqdm==4.67.1
torch==2.4.1+cu118
torchvision==0.19.1+cu118
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/columncat/DeepCompare.git
   cd DeepCompare
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure paths in `data.py`:
   - Set `IMAGE_DIR` to the path of your reference images
   - Set `VIDEO_DIR` to the path of your video files
   - Set `REFERENCE_CLASSES` to your desired classes
   - Set `VIDEO_EXTENSIONS` to your video file extensions

## Usage

1. Prepare your reference images in folders according to class names
2. Update the configuration in `data.py`
3. Run the main processing script:
   ```bash
   python tnc_video.py
   ```
4. Check the results in `count_trim.csv` and log files

## Configuration Options

The system can be configured through several parameters:

- **skip_frame**: Number of frames to skip between processing (default: 60)
- **least_interval**: Minimum interval between detections in seconds (default: 45.0)
- **probability_threshold**: Minimum probability for detection (default: 0.90)

These parameters can be adjusted in the `count_trim_number` function in `tnc_video.py`.

## Processing Pipeline

1. Reference images are loaded for each class
2. Videos are processed frame by frame with configurable frame skipping
3. Each processed frame is:
   - Preprocessed (ROI extraction, CLAHE)
   - Converted to tensor format and normalized
   - Passed through the MobileNetV2 model
   - Compared against reference embeddings
   - Classified based on similarity probabilities
4. Detection results are logged and saved to CSV files

## Results

The system outputs:
- A CSV file (`count_trim.csv`) with detection details:
  - Video filename
  - Timestamp of detection
  - Count of detections
  - Detected class name
- Log files with processing details

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Team ICU** - Pusan National University

## Acknowledgments

- Renault Korea for cooperation and support
- Pusan National University for academic guidance
