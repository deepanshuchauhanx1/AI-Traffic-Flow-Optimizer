# AI Traffic Flow Optimizer

An end-to-end AI system that detects and tracks vehicles in traffic footage, identifies emergency vehicles, and optimizes traffic signal timing using a YOLO-based model.

---

## Project Structure

```
AI-Traffic-Flow-Optimizer/
├── elements/
│   └── traffic.mp4            # Sample traffic footage
├── notebooks/
│   ├── 01_dataset_eda.ipynb       # Dataset exploration & class distribution
│   ├── 02_model_training_yolov8.ipynb  # YOLOv8 training pipeline
│   ├── 03_traffic_pipeline.ipynb  # End-to-end inference pipeline
│   └── 04_tracking_and_signal.ipynb   # Vehicle tracking + signal optimization
├── best (1).pt                # Trained YOLOv8 weights
├── main.py                    # FastAPI inference server
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/your-username/AI-Traffic-Flow-Optimizer
cd AI-Traffic-Flow-Optimizer

# Create virtual environment
python -m venv .myenv
source .myenv/bin/activate        # Windows: .myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run the API

```bash
uvicorn main:app --reload --port 8000
```

Swagger UI → http://localhost:8000/docs

---

## Notebooks — Run in Order

| Notebook | Purpose |
|----------|---------|
| `01_dataset_eda.ipynb` | Explore dataset, check class balance, visualize samples |
| `02_model_training_yolov8.ipynb` | Train YOLOv8 on vehicle + emergency vehicle classes |
| `03_traffic_pipeline.ipynb` | Run full inference pipeline on traffic footage |
| `04_tracking_and_signal.ipynb` | Vehicle tracking + adaptive signal timing logic |

---

## Model

- **Architecture:** YOLOv8 (weights: `best (1).pt`)
- **Top-1 Accuracy:** 85% · **Top-5 Accuracy:** 100%
- **Classes:** TwoWheelers, ambulance, auto-rikshaw, bus, car, firetruck, police vehicle
- **Emergency classes:** ambulance · firetruck · police vehicle

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect/image` | Detect vehicles in an image + density grid |
| POST | `/detect/video` | Frame-by-frame analysis + timeline |
| POST | `/detect/batch` | Classify multiple images at once |
| GET  | `/health` | Model status check |

---

## Quick Example

```bash
# Test with the sample video
curl -X POST "http://localhost:8000/detect/video?sample_rate=5" \
  -F "file=@elements/traffic.mp4"
```

---

## Built With

- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — detection & classification
- [FastAPI](https://fastapi.tiangolo.com/) — inference API
- [OpenCV](https://opencv.org/) — video processing
- [PyTorch](https://pytorch.org/) — model backend
