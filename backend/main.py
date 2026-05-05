import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

from google.colab import drive
drive.mount('/content/drive')

cv2.getBuildInformation()

train = pd.read_csv("/content/drive/MyDrive/ATIS Dataset/train/_classes.csv")
valid = pd.read_csv("/content/drive/MyDrive/ATIS Dataset/valid/_classes.csv")
test = pd.read_csv("/content/drive/MyDrive/ATIS Dataset/test/_classes.csv")

print(f"Shape of train: {train.shape}, valid: {valid.shape}, test: {test.shape}")

# Clean column names by stripping whitespace
train.columns = train.columns.str.strip()
valid.columns = valid.columns.str.strip()
test.columns = test.columns.str.strip()

print(train.columns)

classes = [
    "TwoWheelers",
    "ambulance",
    "auto-rikshaw",
    "bus",
    "car",
    "firetruck",
    "police vehicle"
]
def get_label(row):
    for c in classes:
        if row[c] == 1:
            return c
    return "unknown"

train["label"] = train.apply(get_label, axis=1)
valid["label"] = valid.apply(get_label, axis=1)
test["label"] = test.apply(get_label, axis=1)

train["split"] = "train"
valid["split"] = "valid"
test["split"] = "test"

df = pd.concat([train, valid, test], ignore_index=True)

df

print("Total Images:", len(df))
print(df["label"].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(data=df, x="label")
plt.xticks(rotation=45)
plt.title("Vehicle Class Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="split")
plt.title("Dataset Split Distribution")
plt.show()

import shutil

base_dir = "/content/drive/MyDrive/ATIS_dataset_cls"
for split in ["train", "valid", "test"]:
    for cls in classes + ["unknown"]:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# Copy images to respective folders
def organize_images(split_df, split_name):
    src_folder = f"/content/drive/MyDrive/ATIS Dataset/{split_name}/"
    for _, row in split_df.iterrows():
        src_path = os.path.join(src_folder, row["filename"])
        dest_path = os.path.join(base_dir, split_name, row["label"], row["filename"])

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)

organize_images(train, "train")
organize_images(valid, "valid")
organize_images(test, "test")

print("Dataset organized for classification!")

for split in ["train", "valid", "test"]:
    print(f"\n[{split.upper()}] Image Counts:")
    for cls in classes + ["unknown"]:
        path = os.path.join(base_dir, split, cls)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"  {cls}: {count}")


from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="/content/drive/MyDrive/ATIS_dataset_cls",
    epochs=50,
    imgsz=320,
    batch=16
)

import torch
print(torch.cuda.is_available())

model = YOLO("/content/runs/classify/train/weights/best.pt")

metrics = model.val(data="/content/drive/MyDrive/ATIS_dataset_cls")
print(metrics)

print("Top-1 Accuracy:", metrics.top1)
print("Top-5 Accuracy:", metrics.top5)
print("Fitness Score:", metrics.fitness)

metrics.confusion_matrix.plot()

from IPython.display import Image
Image(filename='/content/runs/classify/val/confusion_matrix.png', width=600)
