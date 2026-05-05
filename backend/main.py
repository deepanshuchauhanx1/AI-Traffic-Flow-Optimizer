"""
Emergency Vehicle Detection API
================================
A learning project - FastAPI backend that:
- Accepts image or video uploads
- Runs emergency vehicle detection using model.pkl
- Returns vehicle count + density info

Run with: uvicorn main:app --reload
"""

import os
import cv2
import pickle
import numpy as np
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🚨 Emergency Vehicle Detector",
    description="Detects emergency vehicles in images and videos using a trained ML model.",
    version="1.0.0"
)

# Allow all origins so the HTML frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Load the model once at startup ─────────────────────────────────────────────
MODEL_PATH = Path("model.pkl")
model = None

def load_model():
    global model
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            logger.error(f"⚠️ Error loading model: {e}")
    else:
        logger.warning(f"⚠️ Warning: {MODEL_PATH} not found. Place it next to main.py.")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code here runs BEFORE the server starts accepting requests
    load_model()
    logger.info("✅ Server started, model loaded!")
    
    yield  # ← Server is running while we're here
    
    # Code here runs AFTER the server shuts down (cleanup)
    logger.info("🛑 Server shutting down...")

# Pass lifespan to FastAPI
app = FastAPI(
    title="🚑 Emergency Vehicle Detection API",
    lifespan=lifespan   # ← add this
)
# ── Helper: preprocess a single frame for the model ────────────────────────────
def preprocess_frame(frame: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Resize + normalize a BGR frame into the format expected by the pkl model.
    Adjust target_size if your model was trained on a different input shape.
    """
    resized = cv2.resize(frame, target_size)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm    = rgb.astype(np.float32) / 255.0          # scale to [0, 1]
    flat    = norm.flatten().reshape(1, -1)            # flatten for sklearn-style models
    return flat

# ── Helper: run detection on one frame ─────────────────────────────────────────
def detect_in_frame(frame: np.ndarray) -> dict:
    """
    Returns { is_emergency: bool, confidence: float }
    Works with sklearn classifiers that have predict_proba,
    or plain predict.
    """
    if model is None:
        raise RuntimeError("Model not loaded")

    X = preprocess_frame(frame)

    # Try to get probability (most sklearn models support this)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        label = int(np.argmax(proba))
        confidence = float(np.max(proba))
    else:
        label = int(model.predict(X)[0])
        confidence = 1.0   # no probability available

    # label=1 → emergency vehicle detected (adjust if your labels differ)
    return {"is_emergency": bool(label == 1), "confidence": confidence}

# ── Helper: compute density label from count / area ────────────────────────────
def density_label(count: int, area_fraction: float = None) -> str:
    """Simple rule-based density description."""
    if count == 0:
        return "None"
    elif count == 1:
        return "Low"
    elif count <= 3:
        return "Medium"
    else:
        return "High"

# ── Response schemas ────────────────────────────────────────────────────────────
class ImageResult(BaseModel):
    vehicle_count: int
    density: str
    confidence: float
    message: str

class VideoResult(BaseModel):
    total_detections: int
    duration_seconds: float
    density_per_second: float
    density_label: str
    frames_analyzed: int
    message: str

# ── Endpoint 1: Image ──────────────────────────────────────────────────────────
@app.post("/detect/image", response_model=ImageResult, summary="Detect emergency vehicles in an image")
async def detect_image(file: UploadFile = File(...)):
    """
    Upload a JPG/PNG image.
    Returns how many emergency vehicles were detected and density info.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place model.pkl next to main.py and restart.")

    # Read upload into memory
    content = await file.read()
    np_arr  = np.frombuffer(content, np.uint8)
    frame   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPG or PNG.")

    result  = detect_in_frame(frame)
    count   = 1 if result["is_emergency"] else 0   # single-frame = 0 or 1
    density = density_label(count)

    return ImageResult(
        vehicle_count=count,
        density=density,
        confidence=round(result["confidence"], 4),
        message=f"{'🚨 Emergency vehicle detected!' if count else '✅ No emergency vehicle found.'}"
    )

# ── Endpoint 2: Video ──────────────────────────────────────────────────────────
@app.post("/detect/video", response_model=VideoResult, summary="Detect emergency vehicles in a video")
async def detect_video(file: UploadFile = File(...)):
    """
    Upload an MP4/AVI video.
    Samples 1 frame per second, returns total count & density per second.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place model.pkl next to main.py and restart.")

    # Save to a temp file (OpenCV needs a real file path)
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video. Send a valid MP4 or AVI.")

        fps            = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec   = total_frames / fps

        # Sample one frame per second to keep it fast
        sample_interval = max(1, int(fps))
        frame_idx       = 0
        detections      = 0
        frames_checked  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                r = detect_in_frame(frame)
                if r["is_emergency"]:
                    detections += 1
                frames_checked += 1
            frame_idx += 1

        cap.release()

        density_ps = round(detections / duration_sec, 4) if duration_sec > 0 else 0.0
        d_label    = density_label(detections, density_ps)

        return VideoResult(
            total_detections=detections,
            duration_seconds=round(duration_sec, 2),
            density_per_second=density_ps,
            density_label=d_label,
            frames_analyzed=frames_checked,
            message=f"Analyzed {frames_checked} frames over {round(duration_sec,1)}s."
        )

    finally:
        os.unlink(tmp_path)   # clean up temp file

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_accuracy": 0.8505
    }

# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Emergency Vehicle Detector API is running 🚨", "docs": "/docs"}