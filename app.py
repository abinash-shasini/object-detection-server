from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import logging
import base64
from typing import List

# Load YOLOv8 model via ultralytics
YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception:
    YOLO = None

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model (optional). We load in a try/except so the server starts if ultralytics not installed.
yolo_model = None
yolo_model_loaded = False
if YOLO is None:
    logging.info('ultralytics not installed; YOLO endpoint will be unavailable')
else:
    try:
        # using yolov8n for fast inference and smaller size (good for free hosting)
        # change to 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', or 'yolov8x.pt' for higher accuracy
        yolo_model = YOLO('yolov8n.pt')
        yolo_model_loaded = True
        logging.info('YOLOv8 model loaded')
    except Exception as e:
        logging.error(f'Failed to load YOLO model: {e}')
        yolo_model = None
        yolo_model_loaded = False

@app.get("/")
def home():
    return {"status": "Server running fine 🚀"}


@app.get("/health")
def health():
    return {"yolo_model_loaded": yolo_model_loaded}

@app.post("/detect")
async def detect_yolo(image: UploadFile = File(...), crop: bool = False, min_score: float = 0.7):
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLO model not loaded on server")

    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_arr = np.array(img)
    height, width = img.height, img.width

    # Run YOLO inference
    results = yolo_model(img_arr)
    r = results[0]
    names = r.names if hasattr(r, 'names') else {}

    boxes_xyxy = []
    scores = []
    classes = []
    if hasattr(r, 'boxes') and getattr(r, 'boxes') is not None:
        try:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            # If tensors are already numpy
            boxes_xyxy = getattr(r.boxes, 'xyxy', [])
            scores = getattr(r.boxes, 'conf', [])
            classes = getattr(r.boxes, 'cls', [])

    filtered = []
    for (x1, y1, x2, y2), score, cls in zip(boxes_xyxy, scores, classes):
        # Convert score to float and check against threshold
        confidence = float(score)
        if confidence < min_score:
            print(f"Skipping detection with confidence {confidence:.3f} (below threshold {min_score})")
            continue
        
        print(f"Accepting detection with confidence {confidence:.3f} (above threshold {min_score})")
        left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)
        ymin = float(y1 / height)
        xmin = float(x1 / width)
        ymax = float(y2 / height)
        xmax = float(x2 / width)

        # Calculate center point for frontend dot display
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

        det = {
            'box': [ymin, xmin, ymax, xmax],
            'box_pixels': [left, top, right, bottom],
            'center_point': [center_x, center_y],  # pixel coordinates for dot placement
            'class': int(cls),
            'score': round(confidence, 4),
            'label': names.get(int(cls), str(int(cls)))
        }

        if crop:
            try:
                cropped = img.crop((left, top, right, bottom))
                bio = io.BytesIO()
                cropped.save(bio, format='JPEG', quality=85)
                bio.seek(0)
                det['crop_b64'] = f"data:image/jpeg;base64,{base64.b64encode(bio.read()).decode('utf-8')}"
            except Exception as e:
                det['crop_error'] = str(e)

        filtered.append(det)

    return {"detections": filtered, "original_size": {"width": width, "height": height}}
