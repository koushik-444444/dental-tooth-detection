---

##  Project Overview

This project implements **automated dental tooth detection and classification** in panoramic radiographs using **YOLOv8 object detection** combined with the **FDI (Federation Dentaire Internationale) numbering system**. The system can detect and classify all **32 permanent teeth** with anatomically correct FDI numbering, providing a comprehensive solution for dental AI applications.

### 🔬 Key Features
- **Complete FDI System**: All 32 permanent teeth (11-18, 21-28, 31-38, 41-48)
- **YOLO Architecture**: State-of-the-art object detection with YOLOv8
- **High Accuracy**: Optimized for dental panoramic radiographs
- **Anatomical Post-Processing**: Upper/lower arch separation and quadrant division
- **Professional Implementation**: Production-ready code with comprehensive documentation

---

##  Results Summary

###  Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **Model Architecture** | YOLOv8s |
| **Input Resolution** | 640×640 |
| **Precision** | 0.8290 |
| **Recall** | 0.8226 |
| **mAP@0.5** | 0.8512 |
| **mAP@0.5:0.95** | 0.5710 |
| **F1-Score** | 0.8124 |
| **Classes** | 32 | Complete FDI permanent teeth |


###  Training Configuration
- **Dataset Split**: 80% train, 10% validation, 10% test
- **Training Epochs**: 150
- **Batch Size**: 8
- **Optimizer**: AdamW with learning rate scheduling
- **Augmentation**: HSV, translation, scaling, horizontal flip

---

### 🔧 Installation

#### 1. Clone Repository

```
https://github.com/koushik-444444/dental-tooth-detection.git
cd dental-tooth-detection
```

#### 2. Create Virtual Environment

```
python -m venv dental_env
source dental_env/bin/activate
```


#### 3. Install Dependencies

```
pip install -r requirements.txt
```


---

##  Dataset Setup

###  Required Dataset Structure

Organize your dataset in YOLO format as follows:

```
dataset/
├── train/
│ ├── images/ # Training images (.jpg, .png)
│ │ ├── panoramic_001.jpg
│ │ ├── panoramic_002.jpg
│ │ └── ...
│ └── labels/ # YOLO format labels (.txt)
│ ├── panoramic_001.txt
│ ├── panoramic_002.txt
│ └── ...
├── val/
│ ├── images/ # Validation images
│ └── labels/ # Validation labels
└── test/
├── images/ # Test images
└── labels/ # Test labels
```


###  Data Format Requirements

#### **Images**
- **Format**: .jpg, .jpeg, .png
- **Resolution**: Minimum 416×416, recommended 640×640 or higher
- **Type**: Panoramic dental radiographs
- **Quality**: Clear visibility of all tooth structures
- **Orientation**: Standard panoramic view (left to right)

#### **Labels (YOLO Format)**
Each `.txt` file contains one line per detected tooth:

```
class_id x_center y_center width height
```

##  Training

```
python train.py --data data.yaml --epochs 150 --imgsz 832 --batch 8 --model yolov8m.pt
```

## Evaluation

```
python evaluate.py --model runs/detect/dental_detection/weights/best.pt --data data.yaml
```




