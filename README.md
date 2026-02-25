# EV Battery Health Intelligence Platform

## 🚀 Quick Start (Restarting the Project)
To run the project again after shutting down your computer:

1.  **Simply double-click** the `start_dashboard.bat` file in this folder.
2.  It will automatically open two terminal windows (Backend & Frontend).
3.  Open your browser to `http://localhost:5173`.

---

## Manual Startup
If you prefer running terminals manually:
1.  **Backend**: `cd backend` -> `uvicorn main:app --reload`
2.  **Frontend**: `cd frontend` -> `npm run dev`

---


## Overview
This project implements a deployment-ready EV Battery Health Estimation system using a Teacher-Student Knowledge Distillation framework.

## Project Structure
- `data/`: Contains dataset files.
- `models/`: Trained model artifacts (`.pkl`).
- `backend/`: FastAPI application (`main.py`).
- `frontend/`: React + Vite + Tailwind dashboard.
- `results/`: Evaluation metrics and anomaly reports.
- `train_student_model.py`: Training pipeline script.
- `anomaly_detection.py`: Anomaly detection script.

## Setup & Running

### Prerequisites
- Python 3.9+
- Node.js & npm

### 1. Backend (FastAPI)
Navigate to the root directory and install Python dependencies:
```bash
pip install fastapi uvicorn pydantic scikit-learn pandas joblib matplotlib seaborn
```

Start the server:
```bash
cd backend
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### 2. Frontend (React)
Navigate to the frontend directory:
```bash
cd frontend
npm install
npm run dev
```
Access the Dashboard at `http://localhost:5173`.

## Features
- **SOH Estimation**: Estimates State of Health using minimal inputs.
- **Latency Features**: Infers internal states (Cycles, Temp) from simple inputs.
- **Anomaly Detection**: Flags potential battery anomalies.
