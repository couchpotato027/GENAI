from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any

# Import local modules
# Note: In production/Docker, ensure these are in PYTHONPATH or relative import works
try:
    from .exam_difficulty_predictor import (
        load_all_splits, assign_difficulty_labels, simulate_student_scores,
        build_features, train_models, predict_difficulty
    )
    from .assessment_agent import AssessmentAnalysisAgent
    from .pedagogical_agent import PedagogicalRetrievalAgent
    from .improvement_agent import AssessmentImprovementAgent
    from .justification_agent import JustificationAgent
    from .question_classifier import classify_question
except ImportError:
    # Fallback for local testing if running from within backend dir
    from exam_difficulty_predictor import (
        load_all_splits, assign_difficulty_labels, simulate_student_scores,
        build_features, train_models, predict_difficulty
    )
    from assessment_agent import AssessmentAnalysisAgent
    from pedagogical_agent import PedagogicalRetrievalAgent
    from improvement_agent import AssessmentImprovementAgent
    from justification_agent import JustificationAgent
    from question_classifier import classify_question

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

class AnalysisRequest(BaseModel):
    question: str
    avg_score: float
    variance: float
    pass_rate: float
    predicted_difficulty: str
    disc_index: float

# ─── Startup ───
@app.on_event("startup")
async def startup_event():
    global MODELS, TFIDF, SCALER, LE
    print("Training models on startup...")
    # Load and prep data
    train_df, _, _ = load_all_splits()
    train_df = assign_difficulty_labels(train_df, seed=42)
    train_df = simulate_student_scores(train_df, seed=42)
    
    # Features
    X_train, tfidf, scaler = build_features(train_df, fit=True)
    TFIDF = tfidf
    SCALER = scaler
    
    # Encoder
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["difficulty"])
    LE = le
    
    # Train
    MODELS = train_models(X_train, y_train)
    print("Models ready.")

# ─── Helpers to Parse Agent Outputs ───
def parse_agent1(text: str) -> Dict[str, Any]:
    # Extract "Assessment Quality: ..."
    quality_match = re.search(r"Assessment Quality:\s*(.+)", text)
    quality = quality_match.group(1).strip() if quality_match else "Unknown"
    
    # Extract Observation bullets
    issues = []
    lines = text.split('\n')
    for line in lines:
        if line.strip().startswith('- '):
            issues.append(line.strip()[2:])
            
    return {"quality": quality, "issues": issues}

def parse_agent2(text: str) -> Dict[str, Any]:
    principles = []
    lines = text.split('\n')
    for line in lines:
        if line.strip().startswith('- '):
            principles.append(line.strip()[2:])
    return {"principles": principles}

def parse_agent3(text: str) -> Dict[str, Any]:
    improvements = []
    rewritten = ""
    
    # Extract bullets for improvements
    lines = text.split('\n')
    for line in lines:
        # Match numbered lists 1. 2. 
        if re.match(r"^\d+\.", line.strip()):
            improvements.append(line.strip())
            
    # Extract "Rewritten Question:"
    # usually at the end
    if "Rewritten Question:" in text:
        parts = text.split("Rewritten Question:")
        if len(parts) > 1:
            rewritten = parts[1].strip()
            
    return {"improvements": improvements, "rewritten": rewritten}

def parse_agent4(text: str) -> Dict[str, Any]:
    # Justification sections
    # - Discrimination analysis:
    # - Difficulty calibration:
    # - Learning outcome alignment:
    
    disc = ""
    diff = ""
    lo = ""
    
    current_section = None
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if "Discrimination analysis:" in line:
            current_section = "disc"
            continue
        elif "Difficulty calibration:" in line:
            current_section = "diff"
            continue
        elif "Learning outcome alignment:" in line:
            current_section = "lo"
            continue
            
        if current_section == "disc" and line:
            disc += line + " "
        elif current_section == "diff" and line:
            diff += line + " "
        elif current_section == "lo" and line:
            lo += line + " "
            
    return {
        "justDisc": disc.strip(),
        "justDiff": diff.strip(),
        "justLO": lo.strip()
    }

# ─── Endpoints ───

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Intelligent Exam Analysis Backend"}

@app.post("/predict")
def predict(req: MatchRequest):
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
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

@app.post("/analyze")
def analyze(req: AnalysisRequest):
    # Reconstruct stats dict
    stats = {
        "avg_score": req.avg_score,
        "pass_rate": req.pass_rate,
        "variance": req.variance
    }
    
    # 1. Assessment Agent
    agent1 = AssessmentAnalysisAgent()
    analysis_text = agent1.analyze(req.question, req.predicted_difficulty, req.disc_index, stats)
    parsed_a1 = parse_agent1(analysis_text)
    
    # 2. Pedagogical
    agent2 = PedagogicalRetrievalAgent()
    pedagogy_text = agent2.retrieve_principles(analysis_text, stats)
    parsed_a2 = parse_agent2(pedagogy_text)
    
    # 3. Improvement
    agent3 = AssessmentImprovementAgent()
    imp_text = agent3.improve(req.question, analysis_text, pedagogy_text, stats)
    parsed_a3 = parse_agent3(imp_text)
    
    # 4. Justification
    agent4 = JustificationAgent()
    rewritten = parsed_a3["rewritten"]
    just_text = agent4.justify(req.question, rewritten, analysis_text, pedagogy_text, stats)
    parsed_a4 = parse_agent4(just_text)
    
    return {
        "agent1": parsed_a1,
        "agent2": parsed_a2,
        "agent3": parsed_a3,
        "agent4": parsed_a4
    }
