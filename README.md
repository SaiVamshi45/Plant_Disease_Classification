


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

| Epoch | Training Acc | Training Loss | Val Acc | Val Loss | LR        | Duration |
|-------|-------------|---------------|---------|----------|-----------|----------|
| 1/5   | 76.46%      | 0.8995        | 87.05%  | 0.4616   | 1.0000e-04| 699s     |
| 2/5   | 96.90%      | 0.0979        | 96.26%  | 0.1129   | 1.0000e-04| 675s     |
| 3/5   | 98.17%      | 0.0584        | 97.69%  | 0.0728   | 1.0000e-04| 653s     |
| 4/5   | 98.66%      | 0.0432        | 97.73%  | 0.0651   | 1.0000e-04| 668s     |
| 5/5   | 98.96%      | 0.0327        | 97.97%  | 0.0608   | 1.0000e-04| 653s     |

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



