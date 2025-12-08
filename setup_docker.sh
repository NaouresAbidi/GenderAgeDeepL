#!/bin/bash

echo "🔧 Setting up Docker environment..."

# Create directories
echo "📁 Creating directories..."
mkdir -p logs mlruns monitoring

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "⚠️  Dockerfile not found! Creating basic Dockerfile..."
    cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY best_age_gender_model_children_tuned.h5 .
COPY run_api.py .

EXPOSE 5000

CMD ["python", "run_api.py"]
EOF
fi

# Check if model exists
if [ ! -f "best_age_gender_model_children_tuned.h5" ]; then
    echo "⚠️  Model file not found, creating dummy model..."
    python -c "
import tensorflow as tf
model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
model.save('best_age_gender_model_children_tuned.h5')
print('✅ Dummy model created')
" 2>/dev/null || echo "❌ Python/TensorFlow not available for dummy model"
fi

echo "✅ Setup complete!"
echo "🚀 Now run: docker-compose up --build"