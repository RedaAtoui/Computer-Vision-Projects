Overview

This project implements a real-time sign language detection system that recognizes specific hand gestures and converts them into speech using Text-to-Speech (TTS). The system is designed to assist communication by bridging the gap between sign language users and those unfamiliar with it.

Features

Real-time hand gesture detection using YOLOv8 and MediaPipe

Classification of detected gestures (e.g., "Hello", "Thanks", "Sorry")

Integration with a Text-to-Speech (TTS) system for real-time speech output

Multi-language support (configurable for different languages)

Optimized tracking using Kalman Filters for smoother recognition

Technologies Used

Python (for sign language detection and classification)

C++ (for real-time Text-to-Speech system)

OpenCV (for image processing)

YOLOv8 (for gesture detection)

MediaPipe (for hand tracking)

TensorFlow (for gesture classification)

Intel RealSense D435i (for depth-based gesture tracking)

Docker (for containerization and deployment)

Installation

Prerequisites

Ensure you have the following dependencies installed:

Python 3.x

OpenCV

YOLOv8

TensorFlow

MediaPipe

PyTorch

Intel RealSense SDK (if using RealSense camera)

CMake & Visual Studio (for compiling the C++ TTS system)

Setup Instructions

1. Clone the Repository

git clone https://github.com/yourusername/SignLanguage_TTS.git
cd SignLanguage_TTS

2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Build the C++ TTS System

cd TTS_Client
mkdir build && cd build
cmake ..
make  # On Windows use: cmake --build .

5. Run the Sign Language Detection System

python sign_language_detection.py

Usage

Start the detection system to recognize sign language gestures.

Detected gestures will be classified into predefined actions.

The C++ TTS module will convert detected gestures into speech in real-time.

Project Structure

SignLanguage_TTS/
│── sign_language_detection.py   # Main script for detection
│── TTS_Client/                  # C++ TTS module
│── models/                       # Trained gesture classification models
│── data/                         # Sample dataset for testing
│── utils/                        # Helper functions
│── requirements.txt              # Python dependencies
│── README.md                     # Project documentation

Future Enhancements

Extend the dataset for more diverse sign language recognition.

Improve accuracy with custom-trained models.

Add support for more complex phrases and sentences.

Optimize real-time performance on edge devices.

Contributing

Feel free to contribute by submitting pull requests or reporting issues!
