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
| `main.py` | **FastAPI application** — loads pre-trained models on startup, exposes `/predict` endpoint that accepts a question + student scores and returns difficulty prediction with confidence |
| `requirements.txt` | Python dependencies (fastapi, scikit-learn, pandas, joblib, uvicorn) |
| `render.yaml` | Render.com deployment configuration |

### Dataset
| Path | Purpose |
| :--- | :--- |
| `data/SciQ/` | SciQ dataset (~13,679 MCQs) split into `train.json`, `valid.json`, `test.json` |

---

## `frontend/` — Static Web UI (Vercel)

| File | Purpose |
| :--- | :--- |
| `index.html` | Main HTML page — input panel, results display, model evaluation section, and About Us |
| `app.js` | Frontend logic — form validation, API calls to `/predict`, results rendering, offline eval display |
| `styles.css` | Complete design system — neobrutalist retro academic theme with all component styles |
| `vercel.json` | Vercel deployment configuration |
| `photos/` | Team member photos for the About Us section |

---

## How It All Fits Together

```
User enters question + scores in frontend (index.html / app.js)
        │
        ▼
Frontend sends POST /predict to backend (main.py)
        │
        ▼
Backend loads pre-trained model (*.joblib) and runs predict_difficulty()
        │
        ▼
Returns: { difficulty, confidence, avg_score, pass_rate, variance }
        │
        ▼
Frontend renders results + shows offline model evaluation metrics
```
