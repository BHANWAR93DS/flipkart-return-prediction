# Flipkart Return Prediction

An end-to-end machine-learning project for predicting whether an e-commerce order is likely to be returned.

## Goal

Build a reproducible ML pipeline that transforms order/customer information into a return-risk prediction and exposes inference through an API.

## Architecture

~~~text
Input Data
  ↓
Validation & Preprocessing
  ↓
Feature Engineering
  ↓
Imbalance Handling
  ↓
Model Training
  ↓
Experiment Tracking
  ↓
Evaluation
  ↓
API Inference
  ↓
Docker
~~~

## Current Stack

Python · Pandas · NumPy · Scikit-learn · Imbalanced-learn · Joblib · MLflow · FastAPI · Pydantic · Uvicorn · Docker · Pytest · HTTPX

## Development Goals

- Keep training and inference preprocessing identical
- Track experiments with MLflow
- Expose predictions through an API
- Add automated tests
- Containerize the service
- Document the inference contract

## Quality Checklist

- [ ] Reproducible training command
- [ ] Dataset/data-access instructions
- [ ] Model evaluation report
- [ ] API request/response examples
- [ ] Unit tests
- [ ] Docker build/run instructions
- [ ] No credentials or secrets committed

This README documents the intended architecture without inventing unverified performance numbers.

**Skills:** Machine Learning · Classification · MLOps · APIs · Docker · Experiment Tracking
