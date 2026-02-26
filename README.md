# Facial Occlusion Classification for Restricted Environments

**Team IntelliSense** - ICT3212 Mini Project  (Implementation 1)

---

## 📌 About the Project
Ensuring security in restricted environments such as Automated Teller Machines (ATMs), bank vaults, and examination centres is a critical challenge. This project proposes an intelligent image classification system using a Convolutional Neural Network (CNN) to categorize facial images into predefined occlusion classes.  

The primary objective of this phase (Implementation 1) is to build a correctly working baseline CNN model and evaluate its performance.

---

## 👥 Team Members
- **T.H.M. Thilakarathna (ICT/2022/104)**  
  https://github.com/Mihiran-Thilakarathna  

- **S.H.M.P.K. Senadheera (ICT/2022/123)**  
  https://github.com/Piyumanjalee  

- **H.M.S.S.W. Bandara (ICT/2022/145)**  
  https://github.com/sandew119  

- **A.K.A. Sanjula (ICT/2022/093)**  
  https://github.com/2002Ashan  

- **S.M.S.C. Seneviratne (ICT/2022/086)**  
  https://github.com/Sachith1227  

---

## 📥 Downloads (Dataset & Pre-trained Model)

⚠️ **Note:** Due to GitHub's 100MB file size limit, the dataset (6000 images) and trained model (`helmet_mask_model.keras` - 148MB) are hosted externally on Google Drive.

- 📂 **Download Dataset Here:**  
  https://drive.google.com/drive/folders/1egZItqzFd3jRCjIlJdO9zD2tNezKPi3m?usp=sharing  

- 🧠 **Download Trained Model Here:**  
  https://drive.google.com/drive/folders/1AlbmFThrn5My_gYhLlBJWiEQCIz9J8AS?usp=sharing  

---

## 📊 Dataset Details
The dataset consists of **6000 images**, divided into three balanced classes:

1. **Clear Face** – 2000 images  
2. **Face with Helmet** – 2000 images  
3. **Face with Mask** – 2000 images  

### Dataset Split:
- 80% Training (4800 images)  
- 20% Validation/Testing (1200 images)  

### Image Preprocessing:
- Resized to **224 × 224 pixels**
- Normalized pixel values

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow, Keras  
- **Image Processing & Visualization:** NumPy, Matplotlib, Seaborn, PIL  
- **Evaluation:** Scikit-learn (Confusion Matrix, Classification Report)  
- **GUI (Prediction Interface):** Tkinter  

---

## 🧠 Model Architecture (Baseline CNN)

The implemented baseline model is a custom Convolutional Neural Network consisting of:

- **Conv2D + MaxPooling2D Layers** – Feature extraction and spatial reduction  
- **BatchNormalization** – Stabilizes and accelerates training  
- **Dropout (0.25 & 0.5)** – Prevents overfitting  
- **Dense Layers** – Fully connected classification layers  
- **Output Layer** – Softmax activation (3-class probability distribution)  

---

## ⚙️ How to Run the Project

### 1️⃣ Setup Environment

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Mihiran-Thilakarathna/Facial_Occlusion_Classification.git
cd Facial_Occlusion_Classification
python -m venv venv
```

Activate the virtual environment:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 3️⃣ Add Dataset & Model

- Download dataset from the Google Drive link above  
- Extract dataset  
- Place dataset folder inside project root directory  

Then:

- Download `helmet_mask_model.keras`  
- Place it in the project root directory  

---

### 4️⃣ Train the Model (Optional)

To train from scratch:

```bash
python train.py
```

Trains for 15 epochs and saves:
- `history.pkl`
- `confusion_matrix.npy`
- `classification_report.txt`

---

### 5️⃣ View Training History

To visualize accuracy, loss curves, and confusion matrix:

```bash
python view_history.py
```

---

### 6️⃣ Make Predictions (GUI)

Run prediction interface:

```bash
python predict.py
```

Features:
- Opens file picker dialog  
- Select an image  
- Displays uploaded image  
- Shows confidence bar chart of predictions  

---
