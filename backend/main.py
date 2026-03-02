from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib

# Import local modules
try:
    from .exam_difficulty_predictor import predict_difficulty
except ImportError:
    from exam_difficulty_predictor import predict_difficulty

from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now, restrict in prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model Store
MODELS = {}
TFIDF = None
SCALER = None
LE = None

# ─── Data Models ───
class MatchRequest(BaseModel):
    question: str
    student_scores: str

# ─── Startup ───
@app.on_event("startup")
async def startup_event():
    global MODELS, TFIDF, SCALER, LE
    print("Loading pre-trained models on startup...")
    model_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        MODELS["Logistic Regression"] = joblib.load(os.path.join(model_dir, "model_lr.joblib"))
        TFIDF = joblib.load(os.path.join(model_dir, "tfidf.joblib"))
        SCALER = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        LE = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
        print(f"Models loaded successfully from {model_dir}")
        print(f"  Model keys: {list(MODELS.keys())}")
    except Exception as e:
        import traceback
        print(f"[CRITICAL] Error loading models from {model_dir}:")
        traceback.print_exc()
        print("Please run backend/train_and_save.py first.")



# ─── Endpoints ───

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Intelligent Exam Analysis Backend"}

@app.post("/predict")
def predict(req: MatchRequest):
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded. Check server startup logs.")
    
    try:
        model = MODELS["Logistic Regression"]
        
        label, confidence, avg, var, pr = predict_difficulty(
            req.question,
            req.student_scores,
            model,
            TFIDF,
            SCALER,
            LE
        )
        
        # Calculate Disc Index here for frontend consistency
        disc_index = round(min(var / 500, 1.0), 2)
        
        return {
            "predicted_difficulty": label,
            "confidence": confidence,
            "avg_score": avg,
            "variance": var,
            "pass_rate": pr,
            "disc_index": disc_index
        }
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] /predict failed:\n{error_detail}")
        raise HTTPException(status_code=500, detail=str(e))

