# Intelligent Exam Question Difficulty Predictor & AI Assessment Assistant

## Project Description

This system predicts the difficulty level (Easy, Medium, or Hard) of exam questions using classical machine learning. It combines natural language processing of question text with student performance statistics to produce a difficulty classification along with model confidence scores.

**Milestone 2** extends this into an **Agentic AI Assessment Design Assistant** that reasons about assessment quality, retrieves pedagogical best practices using RAG, and generates structured improvement suggestions powered by LangGraph and Qwen3-32B via Groq.

---

## How the System Works

### Milestone 1: ML Prediction Pipeline

```
User enters a question + student scores in the web UI
        │
        ▼
Frontend sends POST /predict to the FastAPI backend
        │
        ▼
Backend vectorizes the question text (TF-IDF) and
scales the numeric features (avg score, variance, pass rate)
        │
        ▼
Combined feature vector is fed to a trained Logistic Regression model
        │
        ▼
Returns: predicted difficulty, confidence %, and score summary
        │
        ▼
Frontend renders the result alongside offline evaluation metrics
```

### Milestone 2: Agentic Assessment Pipeline

```
User clicks "Run AI Assessment Analysis" in the AI Assistant panel
        │
        ▼
Frontend sends POST /agent/analyze to FastAPI
        │
        ▼
LangGraph pipeline executes 6 sequential nodes:
        │
  1. Input Node — validates question + scores
  2. ML Analysis Node — calls existing predict_difficulty()
  3. Interpretation Node — rule-based reasoning about WHY
  4. RAG Retriever Node — queries FAISS for pedagogy docs
  5. LLM Reasoning Node — Groq/Qwen3-32B generates insights
  6. Output Formatter Node — validates structured JSON
        │
        ▼
Returns structured report: summary, difficulty analysis,
learning gaps, question issues, recommendations,
pedagogical references, ethical notice
```

### ML Pipeline Overview

1. **Text Preprocessing** — Question text is cleaned and tokenized.
2. **TF-IDF Vectorization** — Text is converted into a 5,000-dimensional feature vector.
3. **Numeric Feature Engineering** — Three student performance statistics are computed: average score, score variance, and pass rate.
4. **Feature Concatenation** — The final feature vector is:
   ```
   [ TF-IDF text features (5000) | avg_score | variance | pass_rate ]
   ```
5. **Standard Scaling** — Numeric features are scaled for model compatibility.
6. **Classification** — A Logistic Regression model predicts difficulty as Easy, Medium, or Hard.

---

## Dataset

The system uses the **SciQ dataset**, which contains approximately **13,679 science multiple-choice questions** with answer options and supporting explanations.

Since the SciQ dataset does not include real student responses, **student performance scores (average score, variance, pass rate) are simulated transparently during training**. The simulation assigns score distributions based on difficulty labels derived from question characteristics.

During live usage, the system accepts **real user-entered scores**, making predictions based on actual performance data provided at inference time.

---

## Machine Learning Approach

Two classifiers were trained and evaluated during development:

| Model               | Role                              |
| :------------------ | :-------------------------------- |
| Logistic Regression | Primary model used in production  |
| Decision Tree       | Trained for comparison during evaluation |

Both models were trained on the combined TF-IDF + numeric feature vector using the scikit-learn library. The Logistic Regression model was selected for deployment based on evaluation performance.

### A Note on Feature Dominance

The model may produce the same difficulty prediction for an empty question with specific scores as it does for a full question with the same scores. This occurs because **student performance features (avg_score, variance, pass_rate) tend to dominate** the prediction over text features alone. This is expected behavior in educational analytics, where empirical student outcomes are strong indicators of question difficulty regardless of question wording.

---

## Agentic Assessment Pipeline (Milestone 2)

### Agent Architecture

The agentic pipeline uses **LangGraph** with explicit state management. Six nodes execute sequentially:

| Node | Function | Technology |
|:---|:---|:---|
| Input Node | Validates question text and scores | Python |
| ML Analysis Node | Runs existing `predict_difficulty()` | Scikit-learn |
| Interpretation Node | Rule-based reasoning about prediction drivers | Python (deterministic) |
| RAG Retriever Node | Queries FAISS index for pedagogy best practices | FAISS + sentence-transformers |
| LLM Reasoning Node | Generates structured improvement suggestions | Groq API / Qwen3-32B |
| Output Formatter Node | Validates and normalizes the final report | Python |

### RAG Knowledge Base

The system includes a curated knowledge base covering:
- Bloom's Taxonomy (all 6 cognitive levels)
- MCQ design best practices
- Common question flaws (ambiguity, guessing cues, poor distractors)
- Difficulty calibration strategies
- Assessment fairness and bias avoidance
- Item discrimination and statistical quality indicators

Documents are chunked, embedded using `all-MiniLM-L6-v2`, and indexed in FAISS for fast retrieval.

### Responsible AI

- All outputs include an ethical disclaimer
- Recommendations are grounded in retrieved pedagogical documents
- The system explicitly states when predictions are driven by numeric features
- A rule-based fallback ensures the system works even without LLM access

---

## Evaluation Metrics

The model is evaluated offline using standard classification metrics:

- **Accuracy** — Overall proportion of correct predictions
- **Precision** — Per-class correctness of positive predictions
- **Recall** — Per-class coverage of actual positives
- **Confusion Matrix** — Detailed breakdown of predictions vs. actual labels

These metrics are displayed in the frontend's Model Evaluation section so users can assess model reliability.

---

## Technology Stack

| Component            | Technology                          |
| :------------------- | :---------------------------------- |
| Language             | Python                              |
| ML Library           | Scikit-learn                        |
| Data Processing      | Pandas, NumPy                       |
| Text Vectorization   | TF-IDF (via Scikit-learn)           |
| Backend API          | FastAPI + Uvicorn                   |
| Frontend             | HTML, CSS, JavaScript               |
| Model Persistence    | Joblib                              |
| Agent Framework      | LangGraph                           |
| Vector Store / RAG   | FAISS + sentence-transformers       |
| LLM                  | Qwen3-32B via Groq API             |
| Backend Deployment   | Render (free tier)                  |
| Frontend Deployment  | Vercel                              |

---

## API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| GET | `/` | Health check |
| POST | `/predict` | ML difficulty prediction (Milestone 1) |
| POST | `/agent/analyze` | Full agentic assessment analysis (Milestone 2) |

### Example Request (Agent)

```bash
curl -X POST http://localhost:8000/agent/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What type of organism is commonly used in preparation of foods such as cheese and yogurt?",
    "student_scores": "85,90,78,92,88,76,95,80,70,82"
  }'
```

### Example Response

```json
{
  "summary": "This question is classified as 'Easy' with high pass rate...",
  "difficulty_analysis": "The ML model predicted 'Easy' driven by...",
  "learning_gaps": ["..."],
  "question_issues": ["Low discrimination index..."],
  "recommendations": ["Increase cognitive complexity...", "Revise distractors..."],
  "pedagogical_references": ["Bloom's Taxonomy: ...", "Item Analysis: ..."],
  "ethical_notice": "This analysis is provided as decision support...",
  "ml_metrics": { "predicted_difficulty": "Easy", "confidence": 0.99, ... },
  "score_dominance": true
}
```

---

## Setup Instructions

1. Clone the repository
2. Create `backend/.env`:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
3. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
4. Run the backend:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
5. Open `frontend/index.html` in a browser (or deploy to Vercel)

---

## Limitations

- Student scores used during training are simulated, not collected from real exam administrations.
- The TF-IDF approach treats questions as bags of words and does not capture semantic meaning or question structure.
- Student performance features dominate predictions, which means text-only predictions (without scores) are less reliable.
- The model is trained exclusively on science MCQs and may not generalize well to other subjects or question formats.

---

## Future Improvements

- Incorporating real student response data would improve model accuracy.
- Replacing TF-IDF with contextual embeddings (e.g., sentence transformers) could improve text feature quality.
- Adding multi-turn agent interaction for iterative question improvement.