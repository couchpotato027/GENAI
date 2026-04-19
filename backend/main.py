from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

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

    # ── Initialize Milestone 2: Agentic Pipeline ──
    _init_agentic_pipeline()


def _init_agentic_pipeline():
    """Initialize the agentic assessment pipeline components."""
    print("\n[Milestone 2] Initializing Agentic Assessment Pipeline...")

    # 1. Inject ML components into agent nodes
    try:
        from agent.nodes import set_ml_components
        if MODELS and TFIDF and SCALER and LE:
            set_ml_components(MODELS["Logistic Regression"], TFIDF, SCALER, LE)
            print("  ✓ ML components injected into agent")
        else:
            print("  ✗ ML models not available — agent ML node will fail gracefully")
    except Exception as e:
        print(f"  ✗ Agent ML setup error: {e}")

    # 2. Build RAG knowledge base
    try:
        from rag.retriever import PedagogyRetriever
        from agent.nodes import set_rag_retriever

        retriever = PedagogyRetriever()
        retriever.build_index()
        set_rag_retriever(retriever)
        print("  ✓ RAG retriever initialized")
    except Exception as e:
        print(f"  ✗ RAG setup error: {e}")

    # 3. Initialize LLM client
    try:
        from llm.groq_client import GroqLLMClient
        from agent.nodes import set_llm_client

        client = GroqLLMClient()
        if client.is_available:
            set_llm_client(client)
            print("  ✓ LLM client initialized (Groq/Qwen3-32B)")
        else:
            print("  ⚠ LLM client unavailable — agent will use rule-based fallback")
    except Exception as e:
        print(f"  ✗ LLM setup error: {e}")

    print("[Milestone 2] Agentic pipeline ready.\n")



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


# ─── Milestone 2: Agentic Assessment Endpoint ───

@app.post("/agent/analyze")
def agent_analyze(req: MatchRequest):
    """
    Run the full agentic assessment pipeline.
    
    Pipeline: Input → ML Prediction → Interpretation → RAG → LLM → Structured Report
    
    Returns a structured JSON report with:
    - summary, difficulty_analysis, learning_gaps, question_issues,
    - recommendations, pedagogical_references, ethical_notice
    - ml_metrics, score_dominance flag, execution_log
    """
    try:
        from agent import run_agent_pipeline

        result = run_agent_pipeline(
            question_text=req.question,
            student_scores=req.student_scores,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] /agent/analyze failed:\n{error_detail}")
        raise HTTPException(status_code=500, detail=str(e))
