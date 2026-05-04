# 🚨 Emergency Vehicle Detector — End-to-End Project

A beginner-friendly project that:
- Takes an **image or video** as input
- Runs **model.pkl** emergency vehicle detector
- Returns **vehicle count**, **density**, and **density per second** (video)

---

## 📁 Project Structure

```
emergency-vehicle-detector/
├── backend/
│   ├── main.py           ← FastAPI app (your API)
│   ├── requirements.txt  ← Python packages
│   ├── Dockerfile        ← Containerise the backend
│   └── model.pkl         ← ⬅ PUT YOUR MODEL HERE
│
├── frontend/
│   └── index.html
│   └── style.css
│   └── script.js
│
├── docker-compose.yml    ← Run everything with one command
└── README.md             ← This file
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
