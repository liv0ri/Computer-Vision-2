# Automatic Detection & Attribute Classification of Maltese Traffic Signs (ARI3129)

**Team Project – Advanced Computer Vision for AI**  
**Deadline:** 21st January 2026     
**Group Members:** 
- Stefania Livori
- Thaina Helena De Oliveira Alves
- Luca Naudi
- Luigi Camilleri
---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset Preparation](#dataset-preparation)  
3. [Sign Attribute Classification](#sign-attribute-classification)  
4. [Results & Evaluation](#results--evaluation)  
5. [Acknowledgments](#acknowledgments)

---

## Project Overview

This project focuses on **automatic detection of Maltese traffic signs** and classification of a chosen attribute. The pipeline consists of three main tasks:

1. **Dataset creation** – Capturing traffic sign images on Maltese streets and annotating them with all required attributes.
2. **Object detection** – Training a custom object detector to localize and classify traffic signs.
3. **Sign attribute classification** – Training a secondary model to classify a selected attribute which includes:
   - Viewing Angle (Front / Side / Back)  
   - Mounting Type (Wall-mounted / Pole-mounted)  
   - Sign Condition (Good / Weathered / Heavily Damaged)  
   - Sign Shape Type (Circle / Square / Triangle / Octagonal / Damaged)  

The project adheres to GDPR and ensures that personal identifiable information is blurred or masked.

---



---

## Dataset Preparation

- Minimum **50 distinct signs per team member**, captured from multiple angles (Front, Side, Back).  
- Annotated in **Label Studio** with all required attributes.  
- **Data validation** performed to ensure:
  - No missing images in the dataset.
  - All annotations have valid bounding boxes.
  - Dataset statistics and class distributions are logged in `1_data_visualisation.ipynb`.  

**Annotation Attributes:**
- **Sign Type:** Stop, No Entry, Pedestrian Crossing, Roundabout Ahead, No Through Road, Blind-Spot Mirror  
- **Viewing Angle:** Front / Side / Back  
- **Mounting Type:** Wall-mounted / Pole-mounted  
- **Sign Condition:** Good / Weathered / Heavily Damaged  
- **Sign Shape Type:** Circle / Square / Triangle / Octagonal / Damaged  

---

## Sign Attribute Classification

- Each member selected **one attribute** (Viewing Angle / Mounting Type / Condition / Shape Type).  
- Trained a **secondary classifier** on crops of detected signs to predict the chosen attribute.  
- Evaluated using accuracy, confusion matrices, and class-level performance.  

---

## Results & Evaluation
 
- **Comparison:** 2c_results_comparison.ipynb provides consolidated analysis and visualization for all members.  

## Acknowledgments
This project was developed as part of the `ARI3129` course at the `University of Malta`.

 
