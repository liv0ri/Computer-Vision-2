# Automatic Detection & Attribute Classification of Maltese Traffic Signs (ARI3129)

**Team Project – Advanced Computer Vision for AI**  
**Deadline:** 31st January 2026  

## Group Members
- Stefania Livori  
- Thaina Helena De Oliveira Alves  
- Luca Naudi  
- Luigi Camilleri  

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset Preparation](#dataset-preparation)  
3. [Traffic Sign Detection](#traffic-sign-detection)  
4. [Sign Attribute Classification](#sign-attribute-classification)  
5. [Acknowledgments](#acknowledgments)

---

## Project Overview

This project focuses on the **automatic detection of Maltese traffic signs** and the **classification of selected sign attributes** using deep learning–based computer vision techniques.

The system consists of three main stages:

1. **Dataset Creation**  
   - Traffic sign images were captured across Maltese streets.
   - Each sign was photographed from multiple viewpoints and annotated with all required attributes.

2. **Traffic Sign Detection**  
   - Custom object detection models were trained to localize and classify traffic sign types.

3. **Sign Attribute Classification**  
   - A secondary model was trained to classify a single selected attribute per team member:
     - Viewing Angle (Front / Side / Back)  
     - Mounting Type (Wall-mounted / Pole-mounted)  
     - Sign Condition (Good / Weathered / Heavily Damaged)  
     - Sign Shape Type (Circle / Square / Triangle / Octagonal / Damaged)  

All data collection and processing strictly adhere to **GDPR requirements**, with any personally identifiable information (e.g., faces or license plates) blurred or masked.

---

## Dataset Preparation

- All datasets are located in the **Assignment Material** folder.
- A total of **seven datasets** were generated using `LS2COCO.ipynb`, split into:
  - **COCO-based datasets** for COCO-compatible models
  - **YOLO-based datasets** for YOLO architectures
- Each team member contributed a minimum of **50 unique traffic signs**, captured from multiple angles (Front, Side, Back).
- Annotations were performed using **Label Studio**, including all required attributes.

### Data Validation

To ensure dataset quality:
- All images were verified to exist and load correctly.
- Bounding boxes were validated for correctness.
- Dataset statistics and class distributions were visualized and logged in `1_data_visualisation.ipynb`.

### Annotation Attributes

- **Sign Type:** Stop, No Entry, Pedestrian Crossing, Roundabout Ahead, No Through Road, Blind-Spot Mirror  
- **Viewing Angle:** Front / Side / Back  
- **Mounting Type:** Wall-mounted / Pole-mounted  
- **Sign Condition:** Good / Weathered / Heavily Damaged  
- **Sign Shape Type:** Circle / Square / Triangle / Octagonal / Damaged  

---

## Traffic Sign Detection

All notebooks prefixed with `2a_*.ipynb` focus on **traffic sign detection and sign-type classification**, each using a different model architecture:

- **`2a_faster_rcnn_stefania_livori.ipynb`**  
  - Dataset: `COCO-based_COCO`  
  - Model: Faster R-CNN  

- **`2a_rf_detr_luca_naudi.ipynb`**  
  - Dataset: `YOLO_COCO`  
  - Model: RF-DETR  

- **`2a_yolov8_thaina_alves.ipynb`**  
  - Dataset: `YOLO_COCO`  
  - Model: YOLOv8  

- **`2a_yolov11_luigi_camilleri.ipynb`**  
  - Dataset: `YOLO_COCO`  
  - Model: YOLOv11  

This multi-model approach enables comparative evaluation of different detection architectures.

---

## Sign Attribute Classification

Each team member developed a **dedicated classifier** for one specific sign attribute. All notebooks are prefixed with `2b_*.ipynb`.

- **`2b_mounting_type_stefania_livori.ipynb`**  
  - Dataset: `COCO-based_COCO_mounting`  
  - Model: RetinaNet  
  - Attribute: Mounting Type  

- **`2b_sign_condition_luca_naudi.ipynb`**  
  - Dataset: `YOLO_COCO_condition`  
  - Model: RF-DETR  
  - Attribute: Sign Condition  

- **`2b_sign_shape_thaina_alves.ipynb`**  
  - Dataset: `YOLO_COCO_sign_shape`  
  - Model: YOLOv12  
  - Attribute: Sign Shape  

- **`2b_view_angle_luigi_camilleri.ipynb`**  
  - Dataset: `YOLO_COCO_view_angle`  
  - Model: YOLOv11  
  - Attribute: Viewing Angle  

This modular design allows independent training and evaluation of each attribute classifier.

---

## Acknowledgments

This project was developed as part of the **ARI3129 – Advanced Computer Vision for AI** course at the **University of Malta**.

Some models were trained using **Google Colab** due to hardware constraints. These notebooks were later adapted to appear as if executed locally for consistency.

Additional experiments were conducted using EfficientDet for viewing angle classification.  
The files **`2b_view_angle_efficientdet_luigi_camilleri.ipynb`** and **`luigi_camilleri_utils.py`** contain the implementation used to train an EfficientDet model on the `YOLO_COCO_view_angle` dataset.  
However, the obtained results were suboptimal, and therefore a YOLOv11-based model was selected as the final approach.