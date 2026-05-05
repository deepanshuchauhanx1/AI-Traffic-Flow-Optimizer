import os
import cv2
import pickle
import numpy as np
import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("model.pkl")
model = None

EMERGENCY_CLASSES = {"ambulance", "firetruck", "police vehicle"}


def load_model():
    global model
    if not MODEL_PATH.exists():
        logger.warning(f"model.pkl not found at {MODEL_PATH.resolve()}")
        return
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info("Server ready.")
    yield
    logger.info("Server shutting down.")


# Single app instance — with lifespan and CORS together
app = FastAPI(
    title="Emergency Vehicle Detection API",
    description="Detects emergency vehicles in images and videos using YOLOv8.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def detect_in_frame(frame: np.ndarray) -> dict:
    if model is None:
        raise RuntimeError("Model not loaded")

    results = model.predict(source=frame, verbose=False)[0]

    emergency_detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            if cls_name in EMERGENCY_CLASSES:
                emergency_detections.append({"class": cls_name, "confidence": conf})

    count = len(emergency_detections)
    confidence = max((d["confidence"] for d in emergency_detections), default=0.0)

    return {
        "is_emergency": count > 0,
        "count": count,
        "confidence": confidence,
        "detections": emergency_detections,
    }


def density_label(count: int) -> str:
    if count == 0:
        return "None"
    elif count == 1:
        return "Low"
    elif count <= 3:
        return "Medium"
    return "High"


class ImageResult(BaseModel):
    vehicle_count: int
    density: str
    confidence: float
    is_emergency: bool
    detections: list
    message: str


class VideoResult(BaseModel):
    total_detections: int
    duration_seconds: float
    density_per_second: float
    density_label: str
    frames_analyzed: int
    message: str


@app.post("/detect/image", response_model=ImageResult)
async def detect_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place model.pkl next to main.py and restart.")

    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPG or PNG.")

    result = detect_in_frame(frame)

    return ImageResult(
        vehicle_count=result["count"],
        density=density_label(result["count"]),
        confidence=round(result["confidence"], 4),
        is_emergency=result["is_emergency"],
        detections=result["detections"],
        message="🚨 Emergency vehicle detected!" if result["is_emergency"] else "✅ No emergency vehicle found.",
    )


@app.post("/detect/video", response_model=VideoResult)
async def detect_video(file: UploadFile = File(...), sample_every_n_frames: int = 15):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place model.pkl next to main.py and restart.")

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video. Send a valid MP4 or AVI.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        sample_interval = max(1, sample_every_n_frames)
        frame_idx = 0
        detections = 0
        frames_checked = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                r = detect_in_frame(frame)
                detections += r["count"]
                frames_checked += 1
            frame_idx += 1

        cap.release()

        density_ps = round(detections / duration_sec, 4) if duration_sec > 0 else 0.0

        return VideoResult(
            total_detections=detections,
            duration_seconds=round(duration_sec, 2),
            density_per_second=density_ps,
            density_label=density_label(detections),
            frames_analyzed=frames_checked,
            message=f"Analyzed {frames_checked} frames over {round(duration_sec, 1)}s.",
        )

    finally:
        os.unlink(tmp_path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "model_accuracy": 0.8505,
    }


@app.get("/")
def root():
    return {"message": "Emergency Vehicle Detector API is running 🚨", "docs": "/docs"}