# Gender & Age Prediction from Facial Images

A deep learning project using a **Convolutional Neural Network (CNN)** to predict the age and gender of a person from a facial image. It includes a complete **TensorFlow** training pipeline and a **Flask** API for live inference.

## Overview

| Detail | Description | 
 | ----- | ----- | 
| **Framework** | TensorFlow / Keras | 
| **Tasks** | Age Prediction (Regression), Gender Prediction (Binary Classification) | 
| **Dataset** | UTKFace ($\sim 20,000$ labeled face images) | 
| **Deployment** | Flask REST API (`/predict` endpoint) | 

## Model Architecture

The model uses a custom CNN with **shared convolutional layers** and **two separate output heads** for age and gender.

| Component | Details | 
 | ----- | ----- | 
| **Shared Layers** | $4$ Convolutional Blocks (filters: $32 \rightarrow 256$) $\rightarrow$ Dense ($512$ units) | 
| **Age Output** | Linear Activation, **MAE** Loss | 
| **Gender Output** | Sigmoid Activation, **Binary Crossentropy** Loss | 
| **Optimizer** | Adam ($\text{lr} = 1\text{e-}4$) | 

### Preprocessing

All images are consistently processed across training and inference:

1. Convert to **Grayscale**.

2. Resize to $360 \times 360$.

3. Normalize pixel values to range $[-1, 1]$.

## Results Summary

| Metric | Value | 
 | ----- | ----- | 
| **Age MAE** | $\sim 4.2$ years | 
| **Gender Accuracy** | $\sim 91\%$ | 
| **Model Size** | $\sim 45$ MB | 

## Setup and Installation

### Requirements

This project requires **Python** $\ge 3.10$.

| Package | Version | 
 | ----- | ----- | 
| `tensorflow` | $\ge 2.14$ | 
| `flask` | (Any) | 
| `pandas`, `numpy`, `matplotlib`, `seaborn` | (Any) | 

To install all necessary dependencies, run:

```
pip install -r requirements.txt
```

## Running the Flask API

### 1. Start the API

Run the `api.py` file to start the inference server. The default port is $5000$.

```
python api.py
```

### 2. Prediction Request

Send a `POST` request with an image file to the `/predict` endpoint:

```
curl -X POST -F "file=@sample.jpg" [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)
```

**Example Response:**

```
{
  "age": 26,
  "gender": "Male",
  "gender_probability": 0.93
}
```

A health check endpoint is also available:

```
curl [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)
```

## Repository Structure

```
.
├── api.py                      # Flask API for inference
├── best_age_gender_model_children_tuned.h5    # Trained model weights (Download from Releases)
├── figures/                    # Visual outputs and plots
├── age-gender-detection-deepl-tuned-on-children.ipynb # Full training and evaluation notebook
├── PREPROCESS.md               # Detailed preprocessing documentation
└── README.md                   # Project documentation
```

***Note:*** *The* `best_age_gender_model_children_tuned.h5` *file is excluded from Git due to its size. Before running the API, please retrain the model using the provided notebook.*