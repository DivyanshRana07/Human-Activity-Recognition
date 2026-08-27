# 🧠 Human Activity Recognition (HAR) from Videos

An end-to-end deep learning project for **video-based human activity recognition** using the **UCF50** dataset.

This notebook-driven project builds, trains, and compares two sequence models:
- **ConvLSTM** (spatiotemporal modeling in a single recurrent-convolution block)
- **LRCN** (CNN feature extraction + LSTM temporal modeling)

It also includes an inference pipeline to run prediction on a real YouTube video and generate an annotated output video.

---

## ✨ Highlights

- 📦 Automatic dataset download and extraction (UCF50)
- 🎞️ Video frame extraction, resizing, and normalization
- 🏷️ One-hot label encoding and train/test split
- 🧱 Two model architectures (ConvLSTM and LRCN)
- 🛑 Early stopping to reduce overfitting
- 📈 Training/validation metric visualization
- 🎬 Real-world video inference with predicted activity overlay

---

## 🎯 Objective

The main goal is to classify human activities from short video clips by learning both:
- **Spatial patterns** (what appears in each frame), and
- **Temporal patterns** (how motion evolves over time).

---

## 🗂️ Dataset

The notebook uses **UCF50**, then selects the following 4 classes for training:

- `WalkingWithDog`
- `TaiChi`
- `Swing`
- `HorseRace`

### Input representation per sample

Each video sample is transformed into a fixed-length sequence:
- **Sequence length:** `20` frames
- **Frame size:** `64 × 64`
- **Channels:** `3 (RGB)`
- **Per-sample tensor shape:** `(20, 64, 64, 3)`

---

## 🧹 Preprocessing Pipeline

1. Read each video with OpenCV
2. Uniformly sample frames across full video duration
3. Resize each frame to `64×64`
4. Normalize pixel values to `[0, 1]`
5. Discard videos with fewer than 20 valid frames
6. Build feature and label arrays
7. Convert labels to one-hot vectors
8. Split into train/test (`75% / 25%`)

---

## 🏗️ Model Architectures

## 1) ConvLSTM Model

Stacked architecture:
- `ConvLSTM2D(filters=4)` → `MaxPooling3D` → `TimeDistributed(Dropout)`
- `ConvLSTM2D(filters=8)` → `MaxPooling3D` → `TimeDistributed(Dropout)`
- `ConvLSTM2D(filters=14)` → `MaxPooling3D` → `TimeDistributed(Dropout)`
- `ConvLSTM2D(filters=16)` → `MaxPooling3D` → `TimeDistributed(Dropout)`
- `Flatten` → `Dense(4, softmax)`

**Total params:** `44,524`

## 2) LRCN Model

Stacked architecture:
- `TimeDistributed(Conv2D(16))` → `MaxPooling2D` → `Dropout`
- `TimeDistributed(Conv2D(32))` → `MaxPooling2D` → `Dropout`
- `TimeDistributed(Conv2D(64))` → `MaxPooling2D` → `Dropout`
- `TimeDistributed(Conv2D(128))` → `MaxPooling2D` → `Dropout`
- `TimeDistributed(Flatten)` → `LSTM(32)` → `Dense(4, softmax)`

**Total params:** `118,180`

---

## ⚙️ Training Setup

Common configuration:
- **Loss:** `categorical_crossentropy`
- **Optimizer:** `Adam`
- **Metric:** `accuracy`
- **Validation split:** `0.2`
- **Batch size:** `4`

Model-specific training:
- **ConvLSTM:** up to `50` epochs, EarlyStopping `patience=10`
- **LRCN:** up to `70` epochs, EarlyStopping `patience=15`

Best weights are restored using `restore_best_weights=True`.

---

## 📊 Results (from notebook run)

### ConvLSTM
- **Test Accuracy:** `0.7689`
- **Test Loss:** `0.7506`
- **Best Validation Accuracy (logged):** `0.8767`

### LRCN
- **Test Accuracy:** `0.8273`
- **Test Loss:** `0.5960`
- **Best Validation Accuracy (logged):** `0.9589`

✅ In this run, **LRCN outperformed ConvLSTM** on the held-out test set.

---

## 🎬 Inference on Custom/YouTube Video

The notebook includes:
- Video download via `yt-dlp`
- Sliding frame queue of length 20
- Frame-wise prediction with trained **LRCN** model
- Writing activity label onto each frame
- Exporting an annotated output `.mp4`

---

## 🧰 Dependencies

Main libraries used:
- `tensorflow`
- `opencv-python (cv2)`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `moviepy`
- `pafy`
- `youtube-dl`
- `yt-dlp`

---

## 🚀 How to Run

1. Open `HAR.ipynb` in Jupyter/Colab.
2. Run cells in order:
   - Install dependencies
   - Download/extract UCF50
   - Create dataset
   - Train ConvLSTM and LRCN
   - Evaluate and visualize metrics
   - Run inference on a test video

---

## 📁 Repository Structure

```text
Human-Activity-Recognition/
└── HAR.ipynb
```

---

## 📝 Notes

- This project currently trains on a **4-class subset** of UCF50 for faster experimentation.
- You can extend to more classes by updating `CLASSES_LIST` in the notebook.

