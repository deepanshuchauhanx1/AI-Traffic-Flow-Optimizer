# Dynamic AI Traffic Flow Optimizer & Emergency Grid

## Overview

This project aims to build an intelligent traffic management system that dynamically optimizes traffic signals using computer vision and artificial intelligence. The system detects vehicles in real-time using YOLOv8 and analyzes traffic density to adapt signal timings. It also introduces an AI-powered green corridor feature that prioritizes emergency vehicles such as ambulances, fire trucks, and police vehicles by clearing traffic signals along their route.

The goal is to reduce congestion, minimize waiting time at intersections, and improve emergency response times in urban environments.

---

## Problem Statement

Traditional traffic lights operate on fixed timers and cannot adapt to real-time traffic conditions. This often results in inefficient signal usage and delays for emergency vehicles.

This project proposes a smart traffic management system that:

* Uses computer vision to detect vehicles and traffic density.
* Dynamically adjusts traffic signals using AI.
* Creates an automatic green corridor for emergency vehicles.

---

## System Architecture

Traffic Camera Feed
↓
Frame Extraction (OpenCV)
↓
Vehicle Detection using YOLOv8
↓
Vehicle Classification (Normal vs Emergency)
↓
Traffic Density Estimation
↓
Reinforcement Learning Signal Optimization
↓
Emergency Vehicle Route Prediction
↓
Green Corridor Signal Control

---

## Features

* Real-time vehicle detection using YOLOv8
* Emergency vehicle recognition (ambulance, firetruck, police)
* Dynamic traffic signal optimization
* AI-based green corridor generation
* Traffic density estimation
* Scalable architecture for smart city deployment

---

## Dataset

The model is trained using traffic vehicle datasets containing the following classes:

* TwoWheelers
* Ambulance
* Auto-Rikshaw
* Bus
* Car
* Firetruck
* Police Vehicle

Dataset structure:

dataset/
train/
images/
train.csv

valid/
images/
valid.csv

test/
images/
test.csv

Each CSV file contains the mapping of images to their respective class labels using one-hot encoding.

---

## Exploratory Data Analysis

EDA was performed to understand the dataset distribution and class balance.

Key analyses performed:

* Class distribution analysis
* Emergency vs normal vehicle ratio
* Dataset split distribution
* Image inspection

EDA notebook can be found in the project repository.

---

## Model

### Object Detection Model

YOLOv8 is used for vehicle detection and classification.

Classes detected:

* TwoWheelers
* Ambulance
* Auto-Rikshaw
* Bus
* Car
* Firetruck
* Police Vehicle

### Training Setup

Framework: PyTorch
Model: YOLOv8
Image size: 640
Optimizer: SGD / Adam
Epochs: 50+

Training command:

yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640

---

## Traffic Density Estimation

Vehicle counts from YOLO detections are used to estimate traffic density per lane.

Density formula:

density = number_of_vehicles / lane_area

This information is passed to the traffic signal optimization system.

---

## Emergency Vehicle Detection

Emergency vehicles are detected using the trained YOLO model.

Emergency classes:

* Ambulance
* Firetruck
* Police Vehicle

Once detected, the system activates green corridor mode.

---

## AI Green Corridor Workflow

Camera detects emergency vehicle
↓
YOLO confirms emergency class
↓
Routing engine predicts vehicle path
↓
Upcoming intersections receive priority signal instructions
↓
Traffic signals turn green sequentially
↓
Cross traffic pauses temporarily
↓
Signals return to adaptive mode after passage

---

## Reinforcement Learning (Future Work)

The system can be extended using reinforcement learning to optimize signal timings.

Possible algorithms:

* Deep Q Network (DQN)
* Proximal Policy Optimization (PPO)

Simulation tools:

* SUMO Traffic Simulator
* CARLA Autonomous Driving Simulator

---

## Project Structure

project/

datasets/
models/
notebooks/
eda.ipynb

training/
train_model.ipynb

inference/
detect_vehicles.py

deployment/
traffic_controller.py

README.md

---

## Installation

Clone the repository:

git clone https://github.com/deepanshuchauhanx1/AI-Traffic-Flow-Optimizer

Install dependencies:

pip install ultralytics
pip install opencv-python
pip install torch torchvision

---

## Running the Model

Train the model:

yolo detect train model=yolov8n.pt data=data.yaml epochs=50

Run inference:

yolo detect predict model=best.pt source=traffic_video.mp4

---

## Deployment

The system can be deployed in three stages:

1. Edge Device
   Traffic cameras capture video streams and run YOLO inference.

2. Central Processing Server
   Processes vehicle detection results and computes traffic optimization decisions.

3. Traffic Signal Controller
   Applies optimized signal timings and activates green corridor when emergency vehicles are detected.

Deployment stack:

* Python
* OpenCV
* YOLOv8
* REST API for traffic controller communication

---

## Evaluation Metrics

Model performance is evaluated using:

* Precision
* Recall
* Mean Average Precision (mAP)
* Traffic waiting time reduction
* Emergency response time improvement

---

## Future Improvements

* Multi-intersection reinforcement learning traffic control
* Audio-based siren detection
* Smart city IoT integration
* Predictive traffic congestion analysis
* Edge AI deployment for real-time inference

---

## References

YOLOv8 – Ultralytics
SUMO Traffic Simulator
Emergency Vehicle Detection Research Papers

---

## Author

[Deepanshu Chauhan](https://github.com/deepanshux1)

[Ujjwal Sharma](https://github.com/24f2006252)

[Tejashvi Yadav](https://github.com/tejaxhvi)

IIT Madras BS in Data Science
