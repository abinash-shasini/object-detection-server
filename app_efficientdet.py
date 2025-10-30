from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
try:
    import tensorflow as tf
except Exception:
    tf = None
import logging
import base64
from typing import List

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load EfficientDet model once at startup
# NOTE: the model bundle in this workspace is at
# `efficientdet_d0_coco17_tpu-32/saved_model` (not under `models/`),
# so use that path. Wrap loading in try/except so the server still starts
# with a clear error if the model isn't present or fails to load.
MODEL_PATH = "efficientdet_d0_coco17_tpu-32/saved_model"
model = None
model_loaded = False
if tf is None:
    logging.warning("TensorFlow is not installed; model will not be loaded.")
else:
    try:
        model = tf.saved_model.load(MODEL_PATH)
        model_loaded = True
        logging.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        # Keep server running but mark model as not loaded. The /detect
        # endpoint will return 503 until this is resolved.
        logging.error(f"Error loading model from {MODEL_PATH}: {e}")
        model = None
        model_loaded = False

@app.get("/")
def home():
    return {"status": "Server running fine 🚀"}


@app.get("/health")
def health():
    return {"model_loaded": model_loaded, "model_path": MODEL_PATH}

@app.post("/detect")
async def detect_object(image: UploadFile = File(...), crop: bool = False, min_score: float = 0.7):
    # Ensure model is available
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img).astype(np.uint8)

    # Run inference
    input_tensor = tf.convert_to_tensor(img_array)[tf.newaxis, ...]
    detections = model(input_tensor)

    # Parse output
    boxes = detections['detection_boxes'][0].numpy().tolist()
    classes = detections['detection_classes'][0].numpy().astype(int).tolist()
    scores = detections['detection_scores'][0].numpy().tolist()

    # Filter results and optionally return server-side crops
    filtered = []
    height, width = img.height, img.width

    for box, cls, score in zip(boxes, classes, scores):
        if float(score) >= float(min_score):
            ymin, xmin, ymax, xmax = box
            left = int(max(0, xmin * width))
            top = int(max(0, ymin * height))
            right = int(min(width, xmax * width))
            bottom = int(min(height, ymax * height))

            # Calculate center point for frontend dot display
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2

            det = {
                "box": [ymin, xmin, ymax, xmax],  # normalized
                "box_pixels": [left, top, right, bottom],
                "center_point": [center_x, center_y],  # pixel coordinates for dot placement
                "class": int(cls),
                "score": round(float(score), 4),
            }

            if crop:
                try:
                    cropped = img.crop((left, top, right, bottom))
                    bio = io.BytesIO()
                    cropped.save(bio, format='JPEG', quality=85)
                    bio.seek(0)
                    b64 = base64.b64encode(bio.read()).decode('utf-8')
                    det['crop_b64'] = f"data:image/jpeg;base64,{b64}"
                except Exception as e:
                    det['crop_error'] = str(e)

            filtered.append(det)

    return {"detections": filtered, "original_size": {"width": width, "height": height}}