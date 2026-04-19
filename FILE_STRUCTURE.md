# 📁 Project File Structure

## Root
| File | Purpose |
| :--- | :--- |
| `README.md` | Project overview, milestones, and evaluation criteria |
| `FILE_STRUCTURE.md` | This file — explains what each file does |
| `milestone1_exam_analysis.ipynb` | Jupyter Notebook walkthrough of the entire Milestone 1 ML pipeline |

---

## `backend/` — FastAPI Server & ML Pipeline

### Core ML Pipeline
| File | Purpose |
| :--- | :--- |
| `exam_difficulty_predictor.py` | **Core ML engine** — loads the SciQ dataset, assigns difficulty labels, simulates student scores, builds TF-IDF + numeric features, trains Logistic Regression & Decision Tree models, and provides the `predict_difficulty()` function |
| `train_and_save.py` | **Pre-training script** — runs the full training pipeline once and saves the fitted model, TF-IDF vectorizer, scaler, and label encoder as `.joblib` files for fast loading in production |

### Pre-Trained Model Artifacts
| File | Purpose |
| :--- | :--- |
| `model_lr.joblib` | Saved Logistic Regression model (trained on 11,679 questions) |
| `tfidf.joblib` | Saved TF-IDF vectorizer (5,000 features) |
| `scaler.joblib` | Saved StandardScaler for numeric features |
| `label_encoder.joblib` | Saved LabelEncoder mapping (Easy / Medium / Hard) |

### API Server
| File | Purpose |
| :--- | :--- |
| `main.py` | **FastAPI application** — loads pre-trained models on startup, exposes `/predict` and `/agent/analyze` endpoints |
| `requirements.txt` | Python dependencies (fastapi, scikit-learn, langgraph, groq, faiss-cpu, etc.) |
| `render.yaml` | Render.com deployment configuration |
| `.env` | Environment variables (GROQ_API_KEY) — not committed to git |
| `.env.example` | Template for environment variables |

### Dataset
| Path | Purpose |
| :--- | :--- |
| `data/SciQ/` | SciQ dataset (~13,679 MCQs) split into `train.json`, `valid.json`, `test.json` |

---

## `backend/agent/` — LangGraph Agentic Pipeline (Milestone 2)

| File | Purpose |
| :--- | :--- |
| `__init__.py` | Package init — exports `run_agent_pipeline` |
| `state.py` | `AgentState` TypedDict definition — shared state contract between all 6 pipeline nodes |
| `nodes.py` | **All 6 node implementations**: input validation, ML analysis, rule-based interpretation, RAG retrieval, LLM reasoning with fallback, output formatting |
| `graph.py` | **LangGraph StateGraph** definition — wires nodes into sequential pipeline and provides the `run_agent_pipeline()` entry point |

---

## `backend/rag/` — RAG Knowledge Base (Milestone 2)

| File | Purpose |
| :--- | :--- |
| `__init__.py` | Package init — exports `PedagogyRetriever` |
| `knowledge_base.py` | **Curated pedagogical knowledge base** — 10 documents covering Bloom's taxonomy, MCQ design, common flaws, difficulty calibration, fairness, and item analysis |
| `retriever.py` | **FAISS-backed retriever** — chunks documents, embeds with sentence-transformers (all-MiniLM-L6-v2), stores in FAISS index, retrieves top-k relevant passages. Falls back to keyword search if embedding model unavailable |

---

## `backend/llm/` — LLM Integration (Milestone 2)

| File | Purpose |
| :--- | :--- |
| `__init__.py` | Package init — exports `GroqLLMClient` |
| `groq_client.py` | **Groq API client** — interfaces with Qwen3-32B model, enforces structured JSON output, temperature 0.2, retry logic with exponential backoff, handles Qwen `<think>` tags |

---

## `frontend/` — Static Web UI (Vercel)

| File | Purpose |
| :--- | :--- |
| `index.html` | Main HTML page — input panel, ML results, model evaluation, **AI Assessment Assistant panel**, and About Us |
| `app.js` | Frontend logic — form validation, API calls to `/predict` and `/agent/analyze`, ML results rendering, **agentic analysis with progress animation and result cards** |
| `styles.css` | Complete design system — neobrutalist retro academic theme with all component styles including **agent cards, progress bars, ethical notices** |
| `vercel.json` | Vercel deployment configuration with rewrites for both `/predict` and `/agent/analyze` |
| `photos/` | Team member photos for the About Us section |

---

## How It All Fits Together

```
User enters question + scores in frontend (index.html / app.js)
        │
        ├──► [Milestone 1] POST /predict → ML prediction → Results display
        │
        └──► [Milestone 2] POST /agent/analyze → LangGraph pipeline:
                │
                ├── Input Node (validation)
                ├── ML Analysis Node (calls predict_difficulty)
                ├── Interpretation Node (rule-based reasoning)
                ├── RAG Retriever Node (FAISS search)
                ├── LLM Reasoning Node (Groq/Qwen3-32B)
                └── Output Formatter Node (structured JSON)
                        │
                        ▼
                AI Assessment Report with recommendations
```
