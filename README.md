# Object Detection FastAPI

This small FastAPI server exposes an endpoint `/detect` to run object detection using an EfficientDet saved model.

Quick start (macOS / zsh):

1. Create and activate a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies (TensorFlow is large; install separately if needed):

```bash
pip install -r requirements.txt
# then install tensorflow appropriate for your machine
pip install tensorflow
```

3. Start the server:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. Health endpoint:

- GET http://localhost:8000/health

5. Detect endpoint (form-data):

- POST http://localhost:8000/detect?crop=true&min_score=0.5
- Body form-data: key `image` (File)

Response contains detections with normalized boxes and `box_pixels`. When `crop=true`, each detection includes `crop_b64` with a data URL you can use directly in an `<img>`.

Troubleshooting

- If TensorFlow installation fails on macOS, consider using a conda environment or a compatible Python version. Tell me your `python3 --version` and I can suggest specific wheels.
