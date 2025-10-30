#!/bin/bash
# Local test script to verify everything works before deployment

echo "🧪 Testing local deployment simulation..."

# Test without venv (should fail)
echo "❌ Testing without venv (expected to fail):"
python3 -c "import ultralytics" 2>/dev/null && echo "✅ ultralytics works without venv" || echo "❌ ultralytics not available without venv (this is expected)"

# Test with venv
echo "✅ Testing with venv:"
source .venv/bin/activate

echo "📦 Current Python and packages:"
which python
python --version
pip list | grep -E "(ultralytics|fastapi|torch)"

echo "🧪 Testing imports:"
python -c "
import sys
print(f'Python: {sys.executable}')

try:
    from ultralytics import YOLO
    print('✅ YOLO import successful')
    
    # Test model loading
    model = YOLO('yolov8n.pt')
    print('✅ YOLOv8 model loaded successfully')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo "🚀 Starting local server for testing..."
echo "Visit http://localhost:8000/debug to check deployment readiness"
echo "Visit http://localhost:8000/health to check model status"
echo "Press Ctrl+C to stop"

uvicorn app:app --reload --host 0.0.0.0 --port 8000