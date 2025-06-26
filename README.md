


#  Plant Disease Classification using CNN (MobileNetV2)

This project uses **Convolutional Neural Networks (CNN)** with transfer learning to classify plant diseases from leaf images. It utilizes the **MobileNetV2** architecture for efficient and accurate predictions across 38 different classes of plant diseases.

---

##  GitHub Repository

The full source code for this project is available on GitHub at:  
👉 **[https://github.com/SaiVamshi45/Plant_Disease_Classification](https://github.com/SaiVamshi45/Plant_Disease_Classification)**

This repository includes:
- ✅ Source code files (`.ipynb`, `.py`)
- ✅ Installation instructions
- ✅ Example dataset structure (see below)
- ✅ Documentation and usage guidelines


---

## 🧠 Model Architecture

- Base Model: `MobileNetV2` (pretrained on ImageNet)
- Input Shape: `(256, 256, 3)`
- Layers:
  - Global Average Pooling
  - Dense Layer with ReLU
  - Dropout
  - Dense Layer with Softmax (38 classes)

---

## 📊 Dataset Overview

- **Training images**: 34,756  
- **Validation images**: 8,673  
- **Total classes**: 38  

Dataset should follow this folder structure:

```

dataset/
├── train/
│   ├── ClassA/
│   ├── ClassB/
│   └── ...
├── validation/
│   ├── ClassA/
│   ├── ClassB/
│   └── ...
└── test/
├── ClassA/
├── ClassB/
└── ...

````

---

## 📈 Training Results

| Epoch | Accuracy | Loss   | Validation Accuracy | Validation Loss | Training Time |
|-------|----------|--------|----------------------|------------------|----------------|
| 1     | 75.82%   | 0.9333 | 92.57%               | 0.2434           | ~7507s         |
| 2     | 97.19%   | 0.0927 | 97.14%               | 0.0941           | ~3339s         |
| 3     | 98.20%   | 0.0570 | 97.19%               | 0.0917           | ~2532s         |
| 4     | 98.49%   | 0.0448 | 98.44%               | 0.0481           | ~2451s         |
| 5     | 98.86%   | 0.0351 | 97.04%               | 0.0980           | ~2954s         |

---

## ⚙️ Installation Instructions

1. Clone the repository:

```bash
git clone https://github.com/SaiVamshi45/Plant_Disease_Classification.git
cd Plant_Disease_Classification
````

2. Install the required packages:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

---

## 🚀 How to Use

1. Prepare your dataset as per the structure shown above.
2. Open the notebook:

```bash
jupyter notebook Lab08_Plant_Disease_Classification.ipynb
```

3. Run each cell to:

   * Load and preprocess the dataset
   * Build and compile the model
   * Train and evaluate it
   * (Optional) Save and test your trained model

---

## 🔬 Future Improvements

* Use data augmentation and early stopping
* Fine-tune MobileNetV2 layers after initial training
* Evaluate on real-world image samples
* Add Grad-CAM for model explainability

---

## 📄 License

This project is for educational use. Modify and adapt freely with proper citation if reused.


> 🧪 Built with deep learning to support smarter agriculture and plant health monitoring.



