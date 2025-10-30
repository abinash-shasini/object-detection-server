#!/bin/bash
# Build script for Render deployment

echo "🚀 Starting build process..."

# Upgrade pip
pip install --upgrade pip

# Install packages with verbose output for debugging
pip install -r requirements.txt --verbose

# List installed packages for debugging
echo "📦 Installed packages:"
pip list

# Test imports
echo "🧪 Testing critical imports..."
python -c "
try:
    import fastapi
    print('✅ FastAPI imported successfully')
except Exception as e:
    print(f'❌ FastAPI import failed: {e}')

try:
    import ultralytics
    print('✅ Ultralytics imported successfully')
except Exception as e:
    print(f'❌ Ultralytics import failed: {e}')

try:
    import torch
    print(f'✅ PyTorch imported successfully (version: {torch.__version__})')
except Exception as e:
    print(f'❌ PyTorch import failed: {e}')

try:
    from ultralytics import YOLO
    print('✅ YOLO class imported successfully')
except Exception as e:
    print(f'❌ YOLO class import failed: {e}')
"

echo "✅ Build process completed!"