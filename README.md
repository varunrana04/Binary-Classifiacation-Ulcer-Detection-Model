# 🩺 Binary Classification — Ulcer Detection Model

## 📌 Overview
This project focuses on **binary image classification** for **stomach ulcer detection** using deep learning (PyTorch).  
The model was trained on the **Stomach Ulcer Dataset (StomachUlcerDS)**, curated and preprocessed for clean binary labels.  
Evaluation shows **perfect classification performance** on the test set with **100% accuracy, precision, recall, and F1 score**.

---

## 🛠️ Tech Stack
- **PyTorch** for model training and evaluation  
- **Torchvision** (Transforms, Datasets, DataLoader)  
- **NumPy, OpenCV, PIL** for image preprocessing  
- **Scikit-learn** for classification metrics  
- **Kaggle** for training environment  

---

## 📊 Dataset
- Source: [StomachUlcerDS (Kaggle)](https://www.kaggle.com/datasets/varunrana3104/stomachulcerds)  
- Total Classes: **2 (Binary)**  
  - `yara1` → Ulcer type/class 1  
  - `yara2` → Ulcer type/class 2  
- Sample Size (Test): **24 images**  

---

## 🔍 Methodology
### 1. **EDA & Preprocessing**
- Removed duplicates using **image hashing (PIL + imagehash)**  
- Ensured consistent image resolution & color normalization  
- Applied **data augmentation** (flips, rotations, brightness adjustments)  
- Created **balanced class splits**  

### 2. **Model**
- Implemented a **PyTorch CNN** for binary classification  
- Used **CrossEntropyLoss + Adam optimizer**  
- Training with early stopping and validation monitoring  

### 3. **Evaluation**
- Metrics: **Accuracy, Precision, Recall, F1-score, AP Score**  
- Evaluation performed on **held-out test set**  

---

## 📈 Results

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| yara1   | 1.00      | 1.00   | 1.00     | 8       |
| yara2   | 1.00      | 1.00   | 1.00     | 16      |
| **Overall** | **1.00** | **1.00** | **1.00** | **24** |

✅ Accuracy: **1.0000**  
✅ Precision: **1.0000**  
✅ Recall: **1.0000**  
✅ F1-Score: **1.0000**  

---

## 🚀 Features
- Full **binary classification pipeline** (EDA → preprocessing → training → evaluation)  
- **Duplicate image removal** via perceptual hashing  
- Robust evaluation with **multiple classification metrics**  
- Kaggle-ready training notebooks  

---
## 📂 Project Structure
Binary-Ulcer-Detection/
│── data/
│ ├── yara1/
│ ├── yara2/
│── eda_preprocessing.ipynb
│── train.py
│── best_model.pth
│── results/
│── README.md


---

## ▶️ Usage
```bash
# Run inference on a sample image
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = torch.load("best_model.pth")
model.eval()

# Transform input
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
img = Image.open("sample.jpg")
input_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
print("Predicted Class:", prediction)

## 📂 Project Structure
