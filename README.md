# Real-Time Drowsiness Detector 

A real-time driver drowsiness detection system built using a **custom CNN model trained from scratch** on the **Closed Eyes in the Wild (CEW)** dataset. This version is implemented with a clean **Streamlit-based UI** and serves as the baseline prototype for future fine-tuned and production-ready releases.

---

## Features 

-  Real-time webcam capture via Streamlit
-  CNN model trained from scratch (3 Conv layers)
-  Live prediction of `Drowsy` vs `Alert` with confidence
-  Modular code with preprocessing and inference utilities
-  Includes pre-trained model: [`drowsiness_cnn.h5`](models/drowsiness_cnn.h5)

### Model Architecture:

| Layer        | Parameters                  |
|--------------|-----------------------------|
| Conv2D       | 32 filters, (3x3), ReLU      |
| MaxPooling2D | (2x2)                        |
| Conv2D       | 64 filters, (3x3), ReLU      |
| MaxPooling2D | (2x2)                        |
| Conv2D       | 128 filters, (3x3), ReLU     |
| MaxPooling2D | (2x2)                        |
| Flatten      | —                           |
| Dropout      | 0.5                         |
| Dense        | 128, ReLU                    |
| Dense        | 1, Sigmoid (binary output)   |

- Input Shape: `224 × 224 × 1` (grayscale)
- Loss Function: `binary_crossentropy`
- Optimizer: `Adam`
- Output: Binary prediction → `Drowsy` (1) or `Alert` (0)

## Dataset Used
- **Closed Eyes in the Wild (CEW)**
> Dataset link: [Kaggle - CEW Dataset](https://www.kaggle.com/datasets/harskish/closed-eyes-in-the-wild)

## Training Performance

| Metric    | Value (approx.)  |
|-----------|------------------|
| Accuracy  | ~48–55%          |
| Input     | Single eye image |
| Limitation| Poor generalization to new users or lighting conditions |

---

##  How to Run

### 1. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Train the Model
```bash
python inspect_data.py
python train_model.py
```
Output: models/drowsiness_cnn.h5

### 4. Run the Real-Time Drowsiness Detection App
```bash
python main.py
```
---

##  v1 — Future Scope & Improvements

| Feature                                | Description                                                                 |
|----------------------------------------|-----------------------------------------------------------------------------|
|  **ResNet50 + Transfer Learning**     | Uses pretrained ResNet50 on ImageNet for stronger feature extraction        |
|  **Fine-Tuning Strategy**             | Unfreezes deeper layers for domain-specific learning (eyes, face patterns)  |
|  **Grad-CAM Heatmap Support**         | Visualize which parts of the face influence predictions                     |
|  **Temporal Drowsiness Logic**        | Ignores short blinks; only alerts on sustained drowsiness over N frames     |
|  **Improved Accuracy & Robustness**   | Handles glasses, low-light, diverse faces more effectively                  |
|  **Toggleable Grad-CAM in UI**        | Users can show/hide heatmaps in real-time                                   |
|  **Pretrained Model Included**        | Model saved as `resnet_drowsiness.h5` for plug-and-play usage               |
|  **Refactored Modular Codebase**      | Clean separation: inference, preprocessing, alert, Grad-CAM modules         |
|  **Streamlit UI**                     | Better layout, color-coded status display, confidence meter                 |

---
