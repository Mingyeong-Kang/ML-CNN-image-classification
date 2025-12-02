# 📘 Image Classification with CNNs  

**Machine Learning Term Project – Ewha Womans University**  
**Models: Baseline CNN · ResNet18 · EfficientNet-B0**



</br>
   
## 📍 1. Overview
This repository contains the implementation and experiments for the **Image Classification with CNNs** project, developed as part of the Machine Learning Term Project at Ewha Womans University.

The goals of this project are to:

- Build CNN-based image classifiers  
- Compare **three architectures**  
- Apply transfer learning  
- Evaluate performance using metrics and visualizations  
- Save training logs for reproducibility  
- Prepare an **A1 poster** and an **8-minute presentation video**

Dataset used: **CIFAR-10**


</br></br>
   
## 🎯 2. Project Objectives

- Implement a **custom baseline CNN** from scratch  
- Use **ResNet18** with ImageNet pretrained weights  
- Apply **EfficientNet-B0** fine-tuning  
- Apply data augmentation  
- Compare three models (project requirement)  
- Save training logs in JSON  
- Generate plots & confusion matrix  



</br></br>
   
## 📂 3. Repository Structure
```
project-root/
│ 
├── train.py # Training script 
├── data.py # CIFAR-10 dataloader   
├── requirements.txt   
├── .gitignore   
│  
├── models/  
│ ├── baseline_cnn.py   
│ ├── resnet18_finetune.py   
│ ├── efficientnet_b0.py   
│ └── init.py   
│   
├── results/   
│ ├── results_baseline.json  
│ ├── results_resnet18.json   
│ └── results_efficientnet_b0.json   
│   
└── plots/ # Accuracy/Loss curves & confusion matrices   
```

> **Note:** Dataset (`data/`) is *not* included in GitHub.  
> CIFAR-10 is automatically downloaded using `torchvision`.



</br></br>
   
## 🗂️ 4. Dataset: CIFAR-10

| Attribute | Value |
|----------|--------|
| Classes | 10 |
| Image size | 32×32 |
| Difficulty | Easy |
| Train samples | 50,000 |
| Test samples | 10,000 |

Dataset download example:

```python
from torchvision.datasets import CIFAR10
CIFAR10(root="./data", download=True)
```


</br></br></br>
   
## 🧠 5. Models Implemented

### **1) Baseline CNN (custom)**
- 4 convolutional blocks  
- ReLU activations  
- MaxPooling  
- Dropout  
- Fully-connected classifier  

### **2) ResNet18 (pretrained)**
- ImageNet pretrained weights  
- Last fully connected layer replaced for CIFAR-10  
- Fine-tuning supported  

### **3) EfficientNet-B0**
- Lightweight and high accuracy  
- Transfer learning suitable for small datasets  
- Implemented using `timm`  


</br></br></br>
   
## ⚙️ 6. Training Pipeline

### **Run Baseline CNN**
```bash
python train.py --model baseline --epochs 20
```

### **Run EfficientNet-B0**
```bash
python train.py --model efficientnet --epochs 20
```

### **Common Arguments**
| Argument | Description |
|----------|--------|
| --model | baseline | resnet18 | efficientnet |
| --epochs | number of training epochs |
| --batch_size | batch size (default: 128) |
| --lr | learning rate |


</br></br></br>
   
## 📊 7. Evaluation Metrics

We evaluate each model using the following metrics:

- **Test Accuracy**  
- **Training & Validation Loss**
- **Accuracy Curves**
- **Confusion Matrix**
- **Class-wise Accuracy**
- **Model Comparison Table**

These metrics help understand performance differences among:
- Baseline CNN  
- ResNet18  
- EfficientNet-B0  


</br></br></br>

## 📈 8. Experimental Results *(to be updated)*

After full training, results will include:

| Model           | Test Accuracy | Train Time | Parameters |
|-----------------|---------------|------------|------------|
| Baseline CNN    | 81.13%        | Fast       | ~1M        |
| ResNet18        | 84.30%        | Medium     | ~11M       |
| EfficientNet-B0 | 85.99 %       | Medium     | ~5M        |

Plots (accuracy curves, loss curves, confusion matrices)  
will be added inside the `/plots` directory.


</br></br></br>

## 🛠️ 9. Installation

Clone the repository:

```bash
git clone https://github.com/Mingyeong-Kang/cnn-image-classification-clean.git
cd cnn-image-classification-clean
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```nginx
torch
torchvision
timm
numpy
matplotlib
scikit-learn
tqdm
```

</br></br></br>

## ▶️ 10. How to Run

### **Simple Run**
```bash
python train.py --model baseline
```

### **Full Training Example**
```bash
python train.py --model resnet18 --epochs 20 --lr 0.001
```

### **Run EfficientNet-B0**
```bash
python train.py --model efficientnet --epochs 20
```

</br></br></br>

## 🧩 11. Key Features

- Clean, modularized project structure  
- CIFAR-10 dataset automatically downloaded through `torchvision`  
- Three models implemented:  
  - **Custom Baseline CNN**  
  - **ResNet18** (ImageNet pretrained)  
  - **EfficientNet-B0** (via `timm`)  
- Training logs saved as JSON (in `/results`)  
- Ready for visualization (accuracy/loss curves, confusion matrices)  
- Reproducible experiment setup  
- Poster-ready methodology & experiment results  


</br></br></br>

## 👥 12. Team Members

| Name            | Role                                                                      |
|-----------------|---------------------------------------------------------------------------|
| **강민경**      | EfficientNet-B0 implementation, training pipeline, integration            |
| **Sanna Ascard-Soederstroem** | Baseline CNN implementation, ResNet18 fine-tuning, experimental runs      |
| **이은서** | Visualization (curves + confusion matrix), documentation, video editing, poster design |

</br></br></br>

## 📄 13. License

This project is released under the **MIT License**.  
See the `LICENSE` file for more information.

</br></br></br>

## 📚 14. References

- PyTorch Documentation  
- torchvision Documentation  
- CIFAR-10 Dataset  
- He et al., *Deep Residual Learning for Image Recognition (ResNet)*  
- Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*  
