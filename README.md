# Nematode Particle Tracking
This project provides a solution for tracking small particles within a specific region of interest in a video, specifically designed to analyze nematode movements. The code preprocesses video frames to filter out irrelevant objects, detects features using ORB (Oriented FAST and Rotated BRIEF), and tracks particle movements across frames using a brute-force matcher.

## Features
Preprocessing: Converts frames to grayscale, applies Gaussian blur, uses adaptive thresholding for better segmentation, and filters objects based on size.
Feature Detection: Utilizes ORB with customized parameters to detect keypoints and compute descriptors.
Tracking: Matches features across frames using a brute-force matcher to track particle movements.
Region of Interest: Focuses on tracking particles within a specified region, useful for videos with a defined area of interest.
## Requirements
OpenCV
NumPy

## Installation
1. Install Python: 3.9.11

2. Clone the repository:
    git clone https://github.com/yourusername/nematode-particle-tracking.git

2. Navigate to the project directory:
    cd nematode-particle-tracking
3. Install the required libraries:
    pip install -r requirements.txt
4. Usage
    Place your video file in the appropriate directory.
    Modify the path to your video file in process_video function if necessary.
5. Run the script:
    python your_script.py
