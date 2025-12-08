FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY best_age_gender_model_children_tuned.h5 .
COPY run_api.py .

EXPOSE 5000

CMD ["python", "run_api.py"]