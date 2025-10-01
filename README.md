# Invisible-red
Implements invisibility effect with Python and OpenCV using color detection and segmentation. The background is captured, and the red region is replaced by this background.

## Overview
Invisible-red is inspired by the “Harry Potter” invisibility cloak and leverages computer vision to make objects covered with a red cloth appear invisible in real-time video streams. This is achieved using background capture, color detection with HSV masks, and smart masking techniques using OpenCV in Python.

## Features
Real-time cloak effect using webcam or video file
Color-based segmentation (default: red)
Easy code customization for other colors
No green-screen required

## Requirements
Python 3.x
OpenCV (opencv-python)
NumPy (numpy)
A red cloth (or any solid color cloth; adjust code for other colors)

## Installation
bash
pip install opencv-python numpy

## Usage
Clone the repository and run the script:
bash
python invisible_cloak.py
Use a red-colored cloth as your "invisibility cloak".
Hold up the cloth in front of the camera; the region covered by the cloth will be replaced with the previously captured static background.

## Algorithm Steps
Background Capture: Capture and store the background frame for a few seconds at startup. This is the “scene” that will replace red regions.
Color Detection: Convert each frame to HSV color space. Detect red regions using tuned thresholds for maximum accuracy.
Mask Generation: Create binary masks to identify red regions in the image. Combine masks for different shades of red and apply morphological operations for noise reduction.
Cloak Effect: Replace pixels in the detected mask with corresponding background pixels, making the cloth-covered areas "invisible".

## License
MIT License

