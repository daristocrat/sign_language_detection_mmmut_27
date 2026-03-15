# 🤟 Sign Language Detection — MMMUT 27

> A real-time Sign Language to Text & Speech conversion system using **MediaPipe**, **OpenCV**, and **Machine Learning**.  
> Built as a project at **Madan Mohan Malaviya University of Technology (MMMUT)**.

---

## 📌 Project Overview

This project recognizes **American Sign Language (ASL)** hand gestures in real-time using a webcam and converts them into:
- 📝 **Text** — displayed on screen
- 🔊 **Speech** — spoken aloud using Text-to-Speech

The system has two modules:
| Module | Description |
|--------|-------------|
| **ASL Alphabet Recognition** | Detects all 26 letters (A–Z) |
| **Gesture Recognition** | Detects 12 common gestures (Hi, Thank You, OK, etc.) |

---

## 🎯 Gestures Supported

| Gesture | Sign | Says |
|---------|------|------|
| 👋 Hi / Wave | Open hand, fingers spread | *"Hi there!"* |
| 🙏 Thank You | Flat open palm | *"Thank you"* |
| 👌 OK | Thumb + index circle | *"OK"* |
| 👍 Yes / Thumbs Up | Thumb pointing up | *"Yes"* |
| 👎 No / Thumbs Down | Thumb pointing down | *"No"* |
| ✋ Stop | Open palm forward | *"Stop"* |
| ✌️ Peace | Index + middle in V | *"Peace"* |
| 👈 Point Left | Index pointing left | *"Go left"* |
| 👉 Point Right | Index pointing right | *"Go right"* |
| 🤙 Call Me | Thumb + pinky out | *"Call me"* |
| 🤟 I Love You | Thumb + index + pinky | *"I love you"* |
| ✊ Fist / Power | All fingers closed | *"Power!"* |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **MediaPipe 0.10.30+** | Hand landmark detection (21 points) |
| **OpenCV** | Webcam capture & UI rendering |
| **Scikit-learn** | Random Forest classifier (98% accuracy) |
| **pyttsx3** | Offline Text-to-Speech |
| **NumPy** | Feature engineering |

---

## 📁 Project Structure

```
sign_language_detection_mmmut_27/
│
├── sign_language_model.py     # ASL A–Z dataset, training, inference
├── realtime_demo.py           # Live webcam ASL recognition
├── demo_no_camera.py          # Test model without webcam
│
├── gesture_model.py           # 12 gesture dataset, training, inference
├── realtime_gesture.py        # Live webcam gesture recognition + TTS
├── gesture_demo.py            # Test gesture model without webcam
│
├── models/
│   ├── asl_rf_model.pkl       # Trained ASL model
│   ├── gesture_rf_model.pkl   # Trained gesture model
│   └── hand_landmarker.task   # MediaPipe hand detection model
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/daristocrat/sign_language_detection_mmmut_27.git
cd sign_language_detection_mmmut_27
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 3. Install dependencies
```bash
pip install mediapipe opencv-python scikit-learn numpy pyttsx3 matplotlib seaborn
```

### 4. Download the hand detection model
```bash
python realtime_gesture.py --download
```

---

## 🚀 Usage

### ASL Alphabet Recognition (A–Z)
```bash
# Train the model
python sign_language_model.py

# Run live webcam
python realtime_demo.py

# Test without webcam
python demo_no_camera.py
```

### Gesture Recognition + Text-to-Speech
```bash
# Train the model
python gesture_model.py

# Run live webcam with voice output
python realtime_gesture.py

# Test without webcam
python gesture_demo.py
```

---

## 🎮 Controls (Webcam Mode)

| Key | Action |
|-----|--------|
| `SPACE` | Add detected letter to sentence |
| `BACKSPACE` | Delete last character |
| `ENTER` | Clear sentence |
| `M` | Mute / Unmute voice |
| `+` / `-` | Increase / Decrease speech speed |
| `S` | Save screenshot |
| `C` | Clear gesture log |
| `Q` / `ESC` | Quit |

---

## 🧠 How It Works

```
Webcam Frame
     ↓
MediaPipe Hand Landmarker
     ↓
21 Hand Landmarks (x, y, z)
     ↓
Feature Engineering
  • Normalized coordinates (63 features)
  • Joint bend angles (10 features)
  • Finger extension scores (5 features)
  • Finger spread distances (4 features)
  • Thumb-index distance (1 feature)
     ↓
Random Forest Classifier (300 trees)
     ↓
Predicted Gesture / Letter + Confidence
     ↓
Display on Screen + Speak via TTS
```

---

## 📊 Model Performance

| Model | Accuracy | Classes |
|-------|----------|---------|
| ASL Alphabet (A–Z) | **97.9%** | 26 letters |
| Gesture Recognition | **98.1%** | 12 gestures |

---

## 👨‍💻 Author

**daristocrat**  
Madan Mohan Malaviya University of Technology (MMMUT)  
GitHub: [@daristocrat](https://github.com/daristocrat)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) by Google — Hand landmark detection
- [OpenCV](https://opencv.org/) — Computer vision library
- [Scikit-learn](https://scikit-learn.org/) — Machine learning
- [pyttsx3](https://pyttsx3.readthedocs.io/) — Offline text-to-speech
