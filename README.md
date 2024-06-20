# YOLOV1 from Scratch with TensorFlow

This project involves building the YOLOV1 (You Only Look Once) computer vision model from scratch using TensorFlow. The model is trained on a dataset of 400 images and has achieved a loss value of around 7.9 and a mean Average Precision (mAP) of 0.92.

## Features

- **Custom YOLOV1 Model**: Implementation of the YOLOV1 model architecture from scratch.
- **Training**: Trained on a dataset of 400 images.
- **Performance**: Achieved a loss value of 7.9 and mAP of 0.92.

## Technology Stack

- **Framework**: TensorFlow
- **Language**: Python
- **Dataset**: Custom dataset of 400 images
- **Metrics**: Loss value and mean Average Precision (mAP)

## Model Overview

Unlike traditional detection methods that involve multiple passes through an image pyramid or sliding window approaches, 
YOLO V1 frames object detection as a single regression problem. It divides the input image into a grid and predicts bounding 
boxes and class probabilities directly from the grid cells, offering remarkable speed and efficiency. Despite its simplicity, 
YOLO V1 achieved competitive accuracy while significantly outperforming existing methods in terms of speed, making it a landmark 
contribution to the field of deep learning and computer vision.

## Setup and Installation

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy
- OpenCV
- Matplotlib

### Note that this is just a personal project, and not inteded for Industry Usage
