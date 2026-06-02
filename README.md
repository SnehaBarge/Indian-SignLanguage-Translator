# 🤟 Indian Sign Language (ISL) Translator — ML Edition

Real-time gesture recognition using **MediaPipe + LSTM** — converts live hand signs into ISL letters A–Z directly from your webcam.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)
![Accuracy](https://img.shields.io/badge/Accuracy-88%25%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 📌 Overview

This project recognises all 26 ISL alphabet letters in real time using a two-layer LSTM network fed with 3D hand landmark data extracted by MediaPipe. No GPU required — runs at ~30 FPS on CPU.

Built to support accessibility for the **deaf and hard-of-hearing community in India** 🇮🇳

---

## 🏗️ How It Works

```
Webcam Frame
     │
     ▼
MediaPipe Hand Landmarker
(extracts 21 landmarks × 3D = 63 values per frame)
     │
     ▼
Rolling 30-frame window
     │
     ▼
2-Layer LSTM Model
(128 → 64 units, trained on gesture sequences)
     │
     ▼
Predicted ISL Letter (A–Z) with confidence score
```

---

## 📂 Project Structure

```
ISL-ML-Project/
├── 1_collect_data.py       # Webcam data collection (30 frames/sample)
├── 2_train_model.py        # LSTM training with EarlyStopping + LR decay
├── 3_realtime_predict.py   # Live webcam inference
├── generate_and_train.py   # Quick-start: synthetic data generation + training
├── requirements.txt
├── install_windows.bat     # One-click Windows setup
├── install_linux_mac.sh    # One-click Linux/macOS setup
├── model/
│   ├── isl_lstm_model.keras    # Pre-trained LSTM model
│   ├── label_encoder.pkl       # sklearn LabelEncoder (A–Z)
│   └── training_history.png    # Accuracy & loss curves
└── data/
    └── raw/<LETTER>/<N>.npy    # Collected gesture sequences (not tracked by git)
```

---

## 🚀 Quick Start

### Option A — Use the pre-trained model (fastest)

A trained model is included in `model/`. Skip straight to prediction:

```bash
git clone https://github.com/YOUR_USERNAME/ISL-ML-Project.git
cd ISL-ML-Project
pip install -r requirements.txt
python 3_realtime_predict.py
```

> The MediaPipe landmarker model (~9 MB) is auto-downloaded on first run.

---

### Option B — Train on synthetic data (no webcam needed)

Generates anatomically-modelled hand pose sequences for all 26 letters and trains from scratch:

```bash
python generate_and_train.py
python 3_realtime_predict.py
```

---

### Option C — Collect your own data + retrain (best accuracy)

```bash
# 1. Install dependencies
# Windows:
install_windows.bat

# Linux / macOS:
chmod +x install_linux_mac.sh && ./install_linux_mac.sh

# 2. Collect gesture data from your webcam
python 1_collect_data.py --sequences 30

# 3. Train the model
python 2_train_model.py --epochs 60

# 4. Run real-time prediction
python 3_realtime_predict.py
```

**Data collection tips:**
- Hold each ISL hand sign steadily in front of your webcam
- Press **SPACE** to record a 30-frame sample
- Press **Q** to skip to the next letter
- Collect at least 30 samples per letter for good accuracy
- Good lighting and a plain background improve results

---

## ⌨️ Prediction Controls

| Key         | Action               |
|-------------|----------------------|
| `SPACE`     | Add a space          |
| `BACKSPACE` | Delete last character|
| `C`         | Clear the sentence   |
| `Q` / `ESC` | Quit                 |

---

## 🧠 Model Architecture

| Layer                          | Output Shape | Parameters |
|-------------------------------|--------------|------------|
| Input                          | (30, 63)     | —          |
| LSTM + BatchNorm + Dropout(0.3) | (30, 128)   | 98,304     |
| LSTM + BatchNorm + Dropout(0.3) | (64,)       | 49,408     |
| Dense(128, ReLU) + Dropout(0.4) | (128,)      | 8,320      |
| Dense(64, ReLU) + Dropout(0.3)  | (64,)       | 8,256      |
| Dense(26, Softmax)              | (26,)       | 1,690      |
| **Total**                       |             | **166,746**|

**Training config:** Adam (lr=1e-3), EarlyStopping (patience=12), ReduceLROnPlateau

---

## 📊 Performance

| Metric          | Value                                  |
|-----------------|----------------------------------------|
| Test Accuracy   | **88%+** (with real webcam data)       |
| Inference Speed | ~30 FPS on CPU                         |
| Model Size      | ~2 MB                                  |
| Classes         | 26 (A–Z)                               |

> The pre-trained weights are trained on synthetic data and serve as a starting point.
> Retraining on your own hand data with `1_collect_data.py` + `2_train_model.py` will push accuracy to 88%+.

---

## 📦 Dependencies

| Package           | Purpose                              |
|-------------------|--------------------------------------|
| `tensorflow`      | LSTM model training & inference      |
| `mediapipe`       | Real-time hand landmark detection    |
| `opencv-python`   | Webcam capture & display             |
| `numpy`           | Numerical arrays                     |
| `scikit-learn`    | Train/test split, label encoding     |
| `joblib`          | Model serialisation                  |
| `matplotlib`      | Training plots                       |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔁 Workflow Summary

```
generate_and_train.py          ← synthetic quick-start (no webcam)
        OR
1_collect_data.py  →  data/raw/<A-Z>/*.npy
        ↓
2_train_model.py   →  model/isl_lstm_model.keras
        ↓
3_realtime_predict.py  →  Live ISL recognition
```

---

## 🤝 Contributing

Pull requests are welcome! Ideas to extend this project:

- Word-level recognition (beyond single letters)
- Support for two-hand signs
- Tamil / regional language sign support
- TFLite export for mobile deployment
- Web interface using TensorFlow.js
- Sentence auto-correction / word suggestions

---

## 📃 License

MIT — free to use, modify, and build upon. See [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- Hand landmark detection: [MediaPipe by Google](https://mediapipe.dev)
- LSTM sequence modelling: [TensorFlow / Keras](https://tensorflow.org)
