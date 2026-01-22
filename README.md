# 🌿 Soybean Leaf Disease Detection from UAV Imagery  
### Using Custom CNN, Transfer Learning & Explainable AI (XAI)

This repository presents an **Explainable Deep Learning Framework** for **soybean leaf disease detection using UAV (drone) imagery**.  
We developed and evaluated a complete pipeline using both a **custom CNN model** and multiple **pre-trained transfer learning models**, along with **explainability methods** to interpret predictions.

📌 **Notebook:** `uav_based_soyabean_leaf_disease.ipynb`  
📄 **Report:** `Explainable-Deep-Learning-Framework-for-Soybean-Leaf.pdf`

---

## 🚀 Project Goal

Early disease detection in soybean crops is important because diseases and pest attacks can reduce crop yield. Manual inspection is slow and difficult for large farms.  
This project solves the problem by using **deep learning on UAV images** to classify soybean leaf conditions.

We classify **4 classes**:

- ✅ Healthy Soybean  
- 🐛 Soybean Semilooper Pest Attack  
- 🟡 Soybean Mosaic  
- 🔴 Rust  

---

## ✨ Key Contributions

✔️ Built a complete UAV-based soybean disease classification pipeline  
✔️ Implemented a **Custom CNN** model from scratch  
✔️ Applied **7 Transfer Learning models** (ImageNet pretrained)  
✔️ Used **data augmentation + early stopping + mixed precision (AMP)**  
✔️ Added **Explainable AI (XAI)** to make predictions interpretable using:

- Grad-CAM  
- Grad-CAM++  
- Eigen-CAM  
- LIME  

---

## 📊 Dataset Information

We used the **Soybean UAV-Based Image Dataset**, part of **MH-SoyaHealthVision** dataset.

📌 Total images: **2842**  
📌 Split:
- **70% Training**
- **20% Validation**
- **10% Testing**

### Class Distribution

| Class | Images | Percentage |
|------|--------|------------|
| Healthy Soybean | 280 | 9.85% |
| Rust | 1000 | 35.18% |
| Semilooper Pest Attack | 790 | 27.80% |
| Soybean Mosaic | 772 | 27.16% |
| **Total** | **2842** | **100%** |

---

## 🧼 Preprocessing & Augmentation

All images were processed using **PyTorch transforms**:

✅ Resize to **224 × 224**  
✅ Normalize (mean = 0.5, std = 0.5)  
✅ Data Augmentation (training only):
- RandomResizedCrop
- RandomHorizontalFlip

---

## 🧠 Models Used

### 🔹 Custom CNN (From Scratch)
A custom CNN architecture with:

- **6 convolution layers**
- **5 fully connected layers**
- MaxPooling after each conv layer  
- Softmax output for 4 classes

---

### 🔹 Transfer Learning Models (Pre-trained on ImageNet)

We fine-tuned the following models:

- ResNet50  
- VGG16  
- MobileNetV2  
- EfficientNet-B3  
- DenseNet121  
- ResNet152  
- DenseNet201  

---

## 🏆 Results Summary

Transfer learning models performed best overall.

### ✅ Performance Comparison

| Model | Accuracy (%) |
|------|--------------|
| ResNet50 | 96.84 |
| VGG16 | 34.39 |
| MobileNetV2 | 97.19 |
| **EfficientNet-B3** | **98.60** ⭐ |
| DenseNet121 | 97.54 |
| ResNet152 | 95.44 |
| DenseNet201 | 97.19 |

📌 **Best Model:** **EfficientNet-B3 (98.60% accuracy)**

---

## 🔍 Explainable AI (XAI)

To make the system trustworthy, we used explainability methods.

### ✅ Grad-CAM / Grad-CAM++ / Eigen-CAM
These methods highlight **where the model is looking** in the leaf image.

Observations:
- Healthy → uniform green areas
- Rust → reddish/brown pustules
- Mosaic → yellow-green irregular patterns
- Pest attack → damaged / eaten regions

### ✅ LIME
LIME explains predictions using **important superpixels** and shows which regions influence the decision.

This confirms that the model focuses on **real disease features**, not background noise.

---

## ⚙️ Experimental Setup

### Hardware (Kaggle Environment)
- GPU: NVIDIA Tesla T4 (or similar)
- RAM: ~16 GB

### Training Setup
- Optimizer: Adam  
- Learning rate: 0.001  
- Batch size: 16  
- Early stopping patience: 5  
- Mixed precision training (AMP): Enabled  

---

## 📌 How to Run

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Lamia-Is/Soybean-Leaf-Disease-Detection-from-UAV-Imagery-Using-Custom-CNN-and-Pre-Trained-Models.git
cd Soybean-Leaf-Disease-Detection-from-UAV-Imagery-Using-Custom-CNN-and-Pre-Trained-Models
