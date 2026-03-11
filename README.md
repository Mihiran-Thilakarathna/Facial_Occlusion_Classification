# Facial Occlusion Classification for Restricted Environments

**Team IntelliSense** — ICT3212 Mini Project 

---

## 📌 About the Project

Ensuring security in restricted environments such as ATMs, bank vaults, and examination centres is a critical challenge. This project develops an intelligent image classification system using deep learning to categorize facial images into three predefined occlusion classes:

- **Clear Face** — No occlusion
- **Face with Helmet** — Helmet covering the head
- **Face with Mask** — Mask covering the lower face

The project evolved through multiple phases, culminating in a highly accurate Transfer Learning model using MobileNetV2, achieving **98% test accuracy** on a properly cleaned dataset.

---

## 👥 Team Members

| Name                  | Reg No       | Index No | GitHub                                                             |
|-----------------------|--------------|----------|--------------------------------------------------------------------|
| T.H.M. Thilakarathna  | ICT/2022/104 | 5707     | [@Mihiran-Thilakarathna](https://github.com/Mihiran-Thilakarathna) |
| S.H.M.P.K. Senadheera | ICT/2022/123 | 5725     | [@Piyumanjalee](https://github.com/Piyumanjalee)                   |
| H.M.S.S.W. Bandara    | ICT/2022/145 | 5946     | [@sandew119](https://github.com/sandew119)                         |
| A.K.A. Sanjula        | ICT/2022/093 | 5696     | [@2002Ashan](https://github.com/2002Ashan)                         |
| S.M.S.C. Seneviratne  | ICT/2022/086 | 5690     | [@Sachith1227](https://github.com/Sachith1227)                     |

---

## 📥 Downloads (Dataset & Pre-trained Models)

> ⚠️ **Note:** Due to GitHub's 100MB file size limit, the dataset and trained models are hosted externally on Google Drive.

### Implementation 1 (Original — Uncleaned Dataset & Baseline Model)

| Resource                                      |                                         Link                                                              |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| 📂 Dataset (6000 images, uncleaned)           | [Download Dataset](https://drive.google.com/drive/folders/1egZItqzFd3jRCjIlJdO9zD2tNezKPi3m?usp=sharing) |
| 🧠 Baseline Model (`helmet_mask_model.keras`) | [Download Model](https://drive.google.com/drive/folders/1AlbmFThrn5My_gYhLlBJWiEQCIz9J8AS?usp=sharing)   |

### Implementation 2 (Final — Cleaned Dataset & Optimized Models)

| Resource | Link |
|----------|------|
| 📂 Cleaned Dataset (4500 images, deduplicated) | [Download Dataset](https://drive.google.com/drive/folders/1RDd0s57LvfggV16qOWMR9aaJsgd7aniG?usp=sharing) |
| 🏆 Final Model V2 (`model_v2.keras` — MobileNetV2) | [Download Model](https://drive.google.com/drive/folders/1RWKMPt4kknm-cbWY18VWxOnJN-mlaazK?usp=sharing) |

---

## 📊 Dataset Details

### Implementation 2 (Cleaned Dataset)
The dataset was rigorously cleaned to remove all duplicate images identified during Implementation 1.

| Class | Images |
|-------|--------|
| Clear Face | 1500 |
| Face with Helmet | 1500 |
| Face with Mask | 1500 |
| **Total** | **4500** |

**Dataset Split:**
- 80% Training → 1200 images per class
- 20% Testing → 300 images per class
- Internal 20% validation split applied during training

### Image Preprocessing
- Resized to **224 × 224 pixels**
- **Custom CNNs (V0, V1):** Pixel values normalized to `[0, 1]` (rescale factor `1./255`)
- **MobileNetV2 (V2):** `preprocess_input` function used to scale pixels to `[-1, 1]`
- Batch size: **32**

---

## 🛠️ Technologies Used

| Category | Libraries / Tools |
|----------|-------------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Pre-trained Model | MobileNetV2 (ImageNet weights) |
| Image Processing | NumPy, PIL |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn (Confusion Matrix, Classification Report) |
| GUI | Tkinter |

---

## 🧠 Model Architecture & Evolution

### Implementation 1 — Baseline CNN (Uncleaned Dataset)
> Achieved **97% accuracy** — later found to be misleading due to duplicate images in the dataset.

- 3 Conv2D + MaxPooling2D blocks (32, 64, 128 filters)
- BatchNormalization + Dropout (0.25, 0.5)
- Dense (128) → Softmax (3 classes)
- Optimizer: Adam | Epochs: 15

---

### Model V0 — Simplified Baseline CNN (Cleaned Dataset)
> **85% test accuracy** | Severe overfitting observed

| Layer | Details |
|-------|---------|
| Input | 224 × 224 × 3 |
| Conv2D (8 filters) | 3×3, ReLU, same padding |
| MaxPooling2D | — |
| Conv2D (16 filters) | 3×3, ReLU, same padding |
| MaxPooling2D | — |
| Flatten | — |
| Dense (16) | ReLU |
| Dense (3) | Softmax |

- Optimizer: SGD (lr=0.01) | Epochs: 15 | No augmentation or regularization

---

### Model V1 — Improved Custom CNN (Cleaned Dataset)
> **88% test accuracy** | Overfitting mitigated

| Layer | Details |
|-------|---------|
| Input | 224 × 224 × 3 |
| Conv2D (32) + BN + MaxPool + Dropout(0.25) | 3×3, ReLU |
| Conv2D (64) + BN + MaxPool + Dropout(0.25) | 3×3, ReLU |
| Conv2D (128) + BN + MaxPool + Dropout(0.25) | 3×3, ReLU |
| Flatten | — |
| Dense (128) + BN + Dropout(0.5) | ReLU |
| Dense (3) | Softmax |

- Optimizer: Adam | Max Epochs: 25 | Early Stopping (patience=5) | LR Reduction (factor=0.2)

---

### Model V2 — MobileNetV2 Transfer Learning ✅ FINAL MODEL (Cleaned Dataset)
> **98% test accuracy** | No overfitting | Selected as final model

| Component | Details |
|-----------|---------|
| Base Model | MobileNetV2 (pre-trained on ImageNet, frozen) |
| Custom Head | GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(3, Softmax) |
| Preprocessing | MobileNetV2 `preprocess_input` (scales to [-1, 1]) |
| Augmentation | Rotation (±20°), Zoom (15%), Shift, Horizontal Flip |
| Optimizer | Adam |
| Max Epochs | 20 (Early Stopping triggered at epoch 6) |
| Early Stopping | Patience = 5 |
| LR Reduction | Factor=0.2, Patience=3, Min LR=1e-5 |

---

## 📈 Model Comparison Summary

| Model | Dataset | Test Accuracy | Key Features |
|-------|---------|---------------|--------------|
| Implementation 1 (Baseline CNN) | Uncleaned (6000 imgs) | 97%* | 3 Conv layers, Adam, Dropout |
| V0 (Simplified CNN) | Cleaned (4500 imgs) | 85% | 2 Conv layers, SGD, no regularization |
| V1 (Improved CNN) | Cleaned (4500 imgs) | 88% | 3 Conv layers, BatchNorm, Dropout, EarlyStopping |
| **V2 (MobileNetV2)** | **Cleaned (4500 imgs)** | **98%** | **Transfer learning, Augmentation, EarlyStopping** |

> *97% accuracy in Implementation 1 was misleading due to duplicate images in the dataset.

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Mihiran-Thilakarathna/Facial_Occlusion_Classification.git
cd Facial_Occlusion_Classification
```

### 2️⃣ Create & Activate a Virtual Environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Dataset & Models

Download the **cleaned dataset** and **final model (V2)** from the Google Drive links above.

- Extract the dataset folder and place it in the project root directory
- Place `model_v2.keras` and `classes_v2.json` in the project root directory

> For Implementation 1 files, download the original dataset and `helmet_mask_model.keras` from the Implementation 1 links.

### 5️⃣ Train a Model (Optional)

To retrain from scratch:

```bash
python train.py
```

This trains for up to 25 epochs (with early stopping) and saves:
- `model_v2.keras`
- `history_v2.pkl`
- `classes_v2.json`

### 6️⃣ View Training History & Evaluation

To visualize accuracy/loss curves and confusion matrix:

```bash
python view_history.py
```

### 7️⃣ Run the Prediction GUI

```bash
python predict.py
```

**Features:**
- Opens a file picker dialog
- Select any image
- Displays the uploaded image
- Shows a confidence bar chart of predictions across all 3 classes

---

