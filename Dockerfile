FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies in one layer
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY best_age_gender_model_children_tuned.h5 .
COPY run_api.py .

# Expose the API port
EXPOSE 5000

# Run the API
CMD ["python", "run_api.py"]
