# 🚨 Emergency Vehicle Detector — End-to-End Project

A beginner-friendly project that:
- Takes an **image or video** as input
- Runs **model.pkl** emergency vehicle detector
- Returns **vehicle count**, **density**, and **density per second** (video)

# Project Status
Still Working on some things to make it better 

---

## 📁 Project Structure

```
emergency-vehicle-detector/
│
├── notebooks/                          ← Research & experimentation
│   ├── 01_dataset_eda.ipynb            ← Exploratory data analysis
│   ├── 02_model_training_yolov8.ipynb  ← YOLOv8 model training
│   ├── 03_traffic_pipeline.ipynb       ← Traffic detection pipeline
│   └── 04_tracking_and_signal.ipynb    ← Vehicle tracking & signals
│
├── backend/                            ← FastAPI inference server
│   ├── main.py                         ← API routes & logic
│   ├── requirements.txt                ← Python dependencies
│   ├── Dockerfile                      ← Backend container
│   └── model.pkl                       ← Trained YOLOv8 weights
│
├── frontend/                           ← Web dashboard
│   ├── index.html                      ← Main UI
│   ├── style.css                       ← Styling
│   └── script.js                       ← Frontend logic
│
├── main.py                             ← Standalone entry point (local run)
├── requirements.txt                    ← Top-level dependencies
├── docker-compose.yml                  ← Run full stack with one command
└── README.md
```

## 🔌Backend API Endpoints

| Method | URL | Input | Output |
|--------|-----|-------|--------|
| POST | `/detect/image` | image file (JPG/PNG) | count, density, confidence |
| POST | `/detect/video` | video file (MP4/AVI) | count, duration, density/sec |
| GET  | `/health`       | — | model status |
| GET  | `/docs`         | — | Swagger UI |


### Example Response (Video):
```json
{
  "total_detections": 5,
  "duration_seconds": 12.4,
  "density_per_second": 0.403,
  "density_label": "Medium",
  "frames_analyzed": 12,
  "message": "Analyzed 12 frames over 12.4s."
}
```

---

## 🧠 How It Works (for learning)

```
User uploads file
       ↓
FastAPI reads the file bytes
       ↓
OpenCV decodes → numpy array (frame)
       ↓
Resize to 224×224 → normalize → flatten
       ↓
model.pkl.predict() → label (0 or 1)
       ↓
Count detections → calculate density
       ↓
Return JSON response
       ↓
Frontend shows results
```


## 📊 Model Stats

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | 85.05% |
| Fitness Score | 92.5% |

---
# Future work:
- Connect Front End 
- deployment
