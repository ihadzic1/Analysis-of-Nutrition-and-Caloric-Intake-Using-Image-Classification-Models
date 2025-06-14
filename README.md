# Analysis-of-Nutrition-and-Caloric-Intake-Using-Image-Classification-Models
---
## 📌 Problem Definition
Problem: 
>The rise in diet-related health issues such as obesity and diabetes highlights the need for better tools to support healthy eating. Traditional calorie-tracking methods are tedious and often inaccurate. This project addresses the problem of automating nutrition and caloric estimation from food images using machine learning.

Goal:
>Train a model that detects food items from images and outputs the caloric information.

Context:
>The growing prevalence of diet-related health issues such as obesity and diabetes requires effective tools for monitoring and improving dietary habits. Traditional methods for tracking calorie intake are manual, time-consuming, and prone to inaccuracies. Advances in computer vision and machine learning provide an opportunity to automate nutrition tracking by analyzing images of food, enabling users to easily estimate caloric intake without manual logging. This project aims to leverage image classification models to detect food items and calculate their caloric values, contributing to better health management through technology.


Relevant datasets and sources include:
>Custom Roboflow dataset and a datset from Roboflow Universe (See more in DATASET.md),
>USDA FoodData Central (Caloric information)

---

## 🧠 Model Architecture & Optimization
The solution uses YoloV8 for object detection fine tuned specifically for food detection:

Caloric estimates are computed using the classified label and corresponding entries from the OpenFoodFacts database.

Evaluation Metrics:
>mAP@0.5,
>mAP@0.5:0.95,
>Precision,
>Recall

---

## 📊 Data Preprocessing and Augmentation
The data is labeled accurately using bounding boxes

Preprocessed using: 
>resizing to 640x640
>and auto-orient

Augmented using:
>horizontal and vertical flips,
>90° rotations (clockwise, counter-clockwise, upside down),
>crop (0% minimum zoom, 20% maximum zoom),
>and rotation (between -15° and +15°)

---

## 📊 Plots & Evaluation metrics
The repository includes:
>Evaluation metrics:
>>mAP@0.5,
>>mAP@0.5:0.95,
>>Precision,
>>Recall (See more in evaluationmetrics/evaluation.md)

>Confusion matrix (evaluationmetrics/confusion_matrix.png)

---

## 🤝 Teamwork and Contributions
Each team member contributed to
>model development,
>data preparation,
>presentation preparing
>and reporting. 

---

## ✅ Conclusions & Future Steps (10%)
Key Takeaways:
>The system effectively classifies food images and estimates caloric content with reasonable accuracy on select food classes which acts as a working prototype. YoloV8 is an effective way for real-time object detection.

Future Improvements:
>Expand dataset size for better accuracy 
>Expand food classes coverage to enable the model to be used in realistic scenarios
>Build a mobile/web app for user-friendly tracking

---
## 🔁 Reproduce the Results
To reproduce our results, follow these steps using Google Colab:
1. 📥 Install Dependencies and Download Dataset
```python
# Install required libraries
!pip install ultralytics roboflow -q

# Import libraries
import os
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from IPython.display import Image, display
from google.colab import drive
from ultralytics import YOLO

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("foodobjectdetection-gbrbd").project("foodobjectdetectiondataset")
version = project.version(7)
dataset = version.download("yolov8")

```
2. 🧠 Train the Model
```python
model = YOLO("yolov8x.pt")  # Using YOLOv8x as base (transfer learning)
dataset_location = dataset.location # Set dataset location

model.train(
    data=f"{dataset_location}/data.yaml",
    epochs=20,         # Set number of epochs
    imgsz=640,        # Set image size
    batch=16,        # Determine batch size
    device=0        # GPU = 0; use 'cpu' if no GPU
)
```
3. 📈 Evaluate the Model
```python
print("\n--- Evaluation Metrics ---")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision (mean): {metrics.box.mp:.4f}")
print(f"Recall (mean): {metrics.box.mr:.4f}")

class_names = model.names
num_metrics = len(metrics.box.p)

print("\nPer-Class Metrics:")
print(f"{'Class':15s} | {'Precision':>9} | {'Recall':>7} | {'AP@0.5':>7} | {'AP@0.5:0.95':>12}")
print("-" * 60)

for i in range(num_metrics):
    name = class_names[i]
    p = metrics.box.p[i]
    r = metrics.box.r[i]
    ap50 = metrics.box.ap50[i]
    ap = metrics.box.ap[i]
    print(f"{name:15s} | {p:.2f}       | {r:.2f}   | {ap50:.2f}   | {ap:.2f}")

# Prepare lists for true and predicted class indices
true_classes = []
pred_classes = []

# Confusion matrix
cm = metrics.confusion_matrix.matrix

# Normalize confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=model.names.values(),
            yticklabels=model.names.values())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()
```
4. 📦 Export and Download Trained Model
```python
#Zip the model
!zip -r best_model_x.zip runs/detect/train/weights/

#Download the model
from google.colab import files
files.download("best_model_x.zip")

```
5. 🔍 Run Inference and Caloric Estimation
```python
#Zip the model
model_path = '/content/drive/MyDrive/model.pt' # Path to model
image_path = '/content/drive/MyDrive/Vockeirl15.jpg'  # Path to image we want to detect

#Load model
model = YOLO(model_path)

#Start prediction
results = model(image_path, save=True)

predict_folders = sorted(glob('runs/detect/predict*'), key=os.path.getmtime)
latest_predict_folder = predict_folders[-1]

base_name = os.path.splitext(os.path.basename(image_path))[0]

matched_files = [f for f in os.listdir(latest_predict_folder) if base_name in f]

if len(matched_files) == 0:
    print("Nije pronađena anotirana slika u folderu:", latest_predict_folder)
else:
    pred_image_path = os.path.join(latest_predict_folder, matched_files[0])
    display(Image(filename=pred_image_path))

#Calories per piece
calorie_dict = {
    "apple": 95, "banana": 105, "blackberry": 1, "blueberry": 1, "bread": 80, "cevap": 75,
    "egg": 70, "grape": 2, "grapefruit": 52, "half pita bread": 85, "lemon": 17,
    "mandarin orange": 47, "orange": 62, "pear": 101, "pita bread": 170, "potato": 130,
    "raspberry": 1, "slice of bread": 80, "strawberry": 4, "watermelon": 86
}


class_names = results[0].names
pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)

#Counting number of occurances of each class
counts = Counter([class_names[c] for c in pred_classes])

#Counting calories
total_calories = 0
print("Detected objects and calories per piece\n")
for cls, count in counts.items():
    calories = calorie_dict.get(cls, 0)
    total = calories * count
    total_calories += total
    print(f"{cls} (x{count}) -> {calories} cal/1 pc = {total} cal")

print(f"\nTotal number of calories: {total_calories} cal")
```
## Our Results:

![download (1)](https://github.com/user-attachments/assets/4f48eb30-1dc6-4393-aef9-663c564815ff)
![download (2)](https://github.com/user-attachments/assets/1ce63698-769c-4132-95b7-1d7af004b3b0)


## Contact

For questions or collaborations, feel free to reach out:

- Email:
- fmuratovic@etf.unsa.ba,
- ihadzic1@etf.unsa.ba,
- hsahinovic@etf.unsa.ba,
- tdukic1@etf.unsa.ba











