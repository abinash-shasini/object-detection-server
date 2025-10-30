# Object Detection API

FastAPI server with YOLOv8 object detection.

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/new)

## Deploy to Render

1. Fork this repository
2. Connect to Render
3. Set environment: `python`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Model status
- `POST /detect` - Object detection (supports `min_score` parameter)
