# 🧠 Exploring ResNet-50 for Image Classification with Real-Time Detection

## 📌 Project Description

This project implements an image classification system using the **ResNet-50** deep neural network. Based on the **CIFAR-10** dataset, we extended it with a custom class called **"container"**, relevant in logistics automation. The model is integrated with a real-time camera stream via Flask and OpenCV.

---

## 🧰 Technologies Used

- Python 3.11  
- [PyTorch](https://pytorch.org/)  
- Torchvision  
- OpenCV  
- Flask  
- YOLOv5 (for optional object detection)
- Google Colab (GPU Tesla T4 for training)  
- scikit-learn (metrics and evaluation)

---

## 🗂️ Project Structure

project_4/
├── argum.py # Data augmentation for 'container' class
├── data.py # CIFAR-10 loading and dataset preparation
├── dataset/ # Processed images (CIFAR-10 + container)
├── model.py # Main ResNet-50 training script
├── opencv.py # Real-time classification via OpenCV (local)
├── confusion_matrix.png # Final confusion matrix output
├── host/
│ ├── app.py # Flask-based camera stream with detection
│ ├── best_resnet50_model.pth # Trained ResNet-50 model
│ ├── templates/
│ │ └── index.html # Front-end HTML for video stream
│ └── yolov5nu.pt # Lightweight YOLOv5 model (optional)
└── README.md # This file

---

## 📊 Dataset

- **Base**: CIFAR-10 (10 categories)
- **Added**: Custom category: **container**
  - 18 original images, augmented to 180 using `argum.py`
- **Split ratio**:
  - 70% training  
  - 15% validation  
  - 15% test  

**Augmentation techniques** used:
- Random crop and horizontal flip
- Rotation and brightness/contrast adjustment
- Perspective transformation

---

## 🧠 Model Architecture: ResNet-50

- Architecture: [ResNet-50](https://arxiv.org/abs/1512.03385) from `torchvision.models`
- All layers frozen except the last fully connected layer
- Final layer adapted to output **11 classes**
- **Loss function**: `CrossEntropyLoss`
- **Optimizer**: `Adam (lr=0.001)`
- **Epochs**: 10
- **Best model saved** based on minimum validation loss

### 🖥 Training Configuration

- Trained in **Google Colab** using GPU (**Tesla T4**)
- Batch size and performance optimized for CIFAR-10 image resolution (32x32)
- `best_resnet50_model.pth` is saved for deployment

---

## 📈 Evaluation

Evaluation metrics computed on the test set include:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

> All metrics are visualized and saved as `.png` for reporting purposes.

---

## 🔴 Real-Time Detection

- Flask app (`app.py`) serves a live webcam stream via HTML5
- Captured frames are processed with **YOLOv5** to generate bounding boxes
- Each bounding box is classified with our **trained ResNet-50 model**
- If YOLO does not detect anything, the entire frame is classified (fallback mode)

### ▶ How to Run Locally

```bash
cd host/
python app.py



