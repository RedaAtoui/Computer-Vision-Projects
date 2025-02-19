# Sign Language Detection with Real-Time TTS

This project implements a real-time **Sign Language Detection** system that recognizes specific gestures (`hello`, `thanks`, `sorry`) and converts them into speech using **Text-to-Speech (TTS)**. The system is built using **Python for detection** and **C++ for TTS**, ensuring efficient real-time performance.

## 🚀 Features
- **Real-time gesture detection** using a trained model relying on LSTM layers.
- **Text-to-Speech (TTS) conversion** for recognized gestures.
- **Seamless Python & C++ integration** for performance optimization.

## 🛠️ Tech Stack
- **Python**: Gesture recognition (OpenCV, Mediapipe, TensorFlow)
- **C++**: TTS system implementation

## 🎯 How It Works
1. **Sign language detection**  
   - The system captures video frames and detects hand gestures.
   - Recognized gestures (`hello`, `thanks`, `sorry`) are classified using a **deep learning LSTM model**.
  
2. **Text-to-Speech (TTS) conversion**  
   - Once a gesture is recognized, the C++ module converts it into **spoken words**.
   - The system runs **both components simultaneously**.



