# ✋ Hand Gesture Recognition AI (Rock–Paper–Scissors)

A real-time **hand gesture recognition system** that detects **Rock, Paper, and Scissors** using **TensorFlow CNN** and **MediaPipe hand tracking**.

The project trains a deep learning model and performs **live gesture prediction using a webcam**.

---

## 🚀 Features

- Real-time hand gesture detection
- CNN model trained using TensorFlow
- MediaPipe hand landmark detection
- Webcam-based prediction
- Clean Python implementation

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Project Structure

```
hand_gesture_ai
│
├── predict.py            # Real-time gesture prediction
├── train_fast.py         # Model training using dataset
├── train.py              # Alternative training script
├── capture_images.py     # Script to collect gesture images
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Aqib053/hand_gesture_ai.git
cd hand_gesture_ai
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run:

```bash
python train_fast.py
```

This will train the model and create:

```
rps_model.keras
```

---

## 🎥 Run Real-Time Detection

Run:

```bash
python predict.py
```

The webcam will open and detect your hand gestures.

---

## 📸 Supported Gestures

| Gesture | Meaning |
|-------|-------|
| ✊ Rock | Closed fist |
| ✋ Paper | Open palm |
| ✌️ Scissors | Two fingers |

---

## 🔮 Future Improvements

- Improve dataset diversity
- Add prediction smoothing
- Build a web interface
- Convert model to TensorFlow Lite for mobile apps

---

## 👤 Author

**Mohammed Aqib**

GitHub:  
https://github.com/Aqib053

---

## ⭐ Support

If you find this project useful, consider giving it a **star ⭐ on GitHub**.