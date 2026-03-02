# Intelligent Exam Question Difficulty Predictor

## Project Description

This system predicts the difficulty level (Easy, Medium, or Hard) of exam questions using classical machine learning. It combines natural language processing of question text with student performance statistics to produce a difficulty classification along with model confidence scores.

The project was built as part of an academic AI/ML course and currently implements Milestone 1 — a complete ML-based prediction pipeline with a deployed web interface. Milestone 2 will extend the system into an agentic pedagogical assistant.

---

## How the System Works

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
| Backend Deployment   | Render (free tier)                  |
| Frontend Deployment  | Vercel                              |

---

## Limitations

- Student scores used during training are simulated, not collected from real exam administrations.
- The TF-IDF approach treats questions as bags of words and does not capture semantic meaning or question structure.
- Student performance features dominate predictions, which means text-only predictions (without scores) are less reliable.
- The model is trained exclusively on science MCQs and may not generalize well to other subjects or question formats.

---

## Future Improvements

- **Milestone 2** will extend this system into an agentic pedagogical assistant that reasons about difficulty factors, retrieves best practices using RAG, and generates structured recommendations for educators.
- Incorporating real student response data would improve model accuracy.
- Replacing TF-IDF with contextual embeddings (e.g., sentence transformers) could improve text feature quality.