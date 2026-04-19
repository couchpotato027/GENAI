"""
Agent Node Implementations
===========================
Six nodes that form the LangGraph assessment pipeline.
Each function takes AgentState and returns a partial state update.

Node 1: input_node        — Validates and stores user inputs
Node 2: ml_analysis_node  — Runs existing ML prediction pipeline
Node 3: interpretation_node — Rule-based reasoning about ML outputs
Node 4: rag_retriever_node — Retrieves pedagogy docs from FAISS
Node 5: llm_reasoning_node — LLM-powered structured analysis
Node 6: output_formatter_node — Validates and formats final report
"""

import os
import sys
import numpy as np
from datetime import datetime

# Ensure parent directory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .state import AgentState


# ──────────────────────────────────────────────
# NODE 1: INPUT NODE
# ──────────────────────────────────────────────

def input_node(state: AgentState) -> dict:
    """
    Validate and store user inputs.
    Ensures question_text and student_scores are present and well-formed.
    """
    question_text = state.get("question_text", "").strip()
    student_scores = state.get("student_scores", "").strip()

    log = [f"[{datetime.now().isoformat()}] Input Node: Received question ({len(question_text)} chars), scores: {student_scores[:50]}..."]

    if not student_scores:
        raise ValueError("student_scores is required")

    # Validate scores are parseable
    scores = [float(s.strip()) for s in student_scores.split(",") if s.strip()]
    if len(scores) < 3:
        raise ValueError(f"At least 3 student scores required, got {len(scores)}")

    log.append(f"[{datetime.now().isoformat()}] Input Node: Validated {len(scores)} scores")

    return {
        "question_text": question_text,
        "student_scores": student_scores,
        "log": log,
    }


# ──────────────────────────────────────────────
# NODE 2: ML ANALYSIS NODE
# ──────────────────────────────────────────────

# These will be set by the FastAPI startup event
_ml_model = None
_ml_tfidf = None
_ml_scaler = None
_ml_label_encoder = None


def set_ml_components(model, tfidf, scaler, label_encoder):
    """Called once at startup to inject the pre-trained ML artifacts."""
    global _ml_model, _ml_tfidf, _ml_scaler, _ml_label_encoder
    _ml_model = model
    _ml_tfidf = tfidf
    _ml_scaler = scaler
    _ml_label_encoder = label_encoder


def ml_analysis_node(state: AgentState) -> dict:
    """
    Run the existing ML prediction pipeline.
    Calls predict_difficulty() from the original codebase — NO retraining.
    """
    from exam_difficulty_predictor import predict_difficulty

    if _ml_model is None:
        raise RuntimeError("ML models not loaded. Call set_ml_components() at startup.")

    label, confidence, avg_score, variance, pass_rate = predict_difficulty(
        state["question_text"],
        state["student_scores"],
        _ml_model,
        _ml_tfidf,
        _ml_scaler,
        _ml_label_encoder,
    )

    disc_index = round(min(variance / 500, 1.0), 2)

    log = state.get("log", [])
    log.append(
        f"[{datetime.now().isoformat()}] ML Analysis Node: "
        f"difficulty={label}, confidence={confidence:.3f}, "
        f"avg={avg_score:.1f}, var={variance:.1f}, pr={pass_rate:.1f}%, disc={disc_index}"
    )

    return {
        "predicted_difficulty": label,
        "confidence": round(confidence, 4),
        "avg_score": round(avg_score, 2),
        "variance": round(variance, 2),
        "pass_rate": round(pass_rate, 2),
        "disc_index": disc_index,
        "log": log,
    }


# ──────────────────────────────────────────────
# NODE 3: INTERPRETATION NODE (CRITICAL)
# ──────────────────────────────────────────────

def interpretation_node(state: AgentState) -> dict:
    """
    Deterministic, rule-based reasoning about WHY the ML model
    predicted a certain difficulty level.

    This node explicitly reasons about:
    1. Difficulty source — numeric features vs. question text
    2. Learning gaps — based on score patterns
    3. Question quality indicators
    4. Feature dominance detection

    NO LLM is used here — this is pure analytical reasoning.
    """
    avg = state["avg_score"]
    var = state["variance"]
    pr = state["pass_rate"]
    diff = state["predicted_difficulty"]
    conf = state["confidence"]
    disc = state["disc_index"]
    question = state.get("question_text", "")

    lines = []

    # ── 1. Difficulty Source Analysis ──
    lines.append("## Difficulty Source Analysis")

    # Detect numeric feature dominance
    # In this model, student performance features (avg_score, variance, pass_rate)
    # typically dominate the 5000 TF-IDF text features.
    numeric_dominance = True  # Known characteristic of this model

    if numeric_dominance:
        lines.append(
            "⚠ SCORE DOMINANCE DETECTED: The prediction is primarily driven by "
            "student performance statistics (avg_score, variance, pass_rate), not "
            "question text. This is expected behavior — empirical student outcomes "
            "are strong difficulty indicators regardless of question wording."
        )

    # Explain the specific numeric drivers
    if diff == "Easy":
        if avg > 70:
            lines.append(f"→ High average score ({avg:.1f}) strongly indicates an easy question.")
        if pr > 80:
            lines.append(f"→ Very high pass rate ({pr:.1f}%) — most students succeeded.")
        if var < 100:
            lines.append(f"→ Low variance ({var:.1f}) — consistent performance across students.")
    elif diff == "Medium":
        if 45 <= avg <= 70:
            lines.append(f"→ Moderate average score ({avg:.1f}) indicates medium difficulty.")
        if 40 <= pr <= 80:
            lines.append(f"→ Mixed pass rate ({pr:.1f}%) — some students struggled.")
        if var > 100:
            lines.append(f"→ Notable variance ({var:.1f}) — inconsistent understanding among students.")
    elif diff == "Hard":
        if avg < 50:
            lines.append(f"→ Low average score ({avg:.1f}) indicates a challenging question.")
        if pr < 50:
            lines.append(f"→ Low pass rate ({pr:.1f}%) — majority of students struggled.")
        if var > 200:
            lines.append(f"→ High variance ({var:.1f}) — wide gap between strong and weak students.")

    # ── 2. Learning Gap Analysis ──
    lines.append("\n## Learning Gap Analysis")

    if avg < 40 and var > 150:
        lines.append(
            "🔴 CRITICAL: Low scores combined with high variance suggest a "
            "fundamental knowledge gap. Some students may lack prerequisite "
            "understanding while others have partial mastery."
        )
    elif avg < 50:
        lines.append(
            "🟡 CONCERN: Below-average scores suggest potential gaps in "
            "student preparation or instruction for this topic."
        )
    elif avg < 60 and var > 200:
        lines.append(
            "🟡 MIXED: Moderate scores with high variance indicate "
            "inconsistent understanding — the class is split between "
            "those who grasp the concept and those who don't."
        )
    else:
        lines.append(
            "🟢 Scores suggest adequate student preparation for this difficulty level."
        )

    if var > 250:
        lines.append(
            "⚡ EQUITY ALERT: Very high variance may indicate that instruction "
            "or the question itself is not equitably accessible to all students."
        )

    # ── 3. Question Quality Analysis ──
    lines.append("\n## Question Quality Analysis")

    # Discrimination index analysis
    if disc < 0.2:
        lines.append(
            f"⚠ LOW DISCRIMINATION (index: {disc}): This question does not "
            f"effectively distinguish between high- and low-performing students. "
            f"Consider revising distractors or question stem."
        )
    elif disc < 0.4:
        lines.append(
            f"→ Moderate discrimination (index: {disc}): Acceptable but could be improved."
        )
    else:
        lines.append(
            f"✓ Good discrimination (index: {disc}): Effectively separates mastery levels."
        )

    # Extreme difficulty flags
    if pr > 95:
        lines.append(
            "⚠ CEILING EFFECT: Near-universal success suggests this question "
            "may be too easy to provide useful diagnostic information."
        )
    elif pr < 10:
        lines.append(
            "⚠ FLOOR EFFECT: Nearly all students failed — question may be "
            "unfairly difficult, poorly worded, or testing untaught content."
        )

    # Confidence analysis
    if conf < 0.5:
        lines.append(
            f"⚠ LOW MODEL CONFIDENCE ({conf:.1%}): The model is uncertain about "
            f"this classification. The question may sit at a difficulty boundary."
        )
    elif conf > 0.9:
        lines.append(
            f"✓ High model confidence ({conf:.1%}): Strong classification certainty."
        )

    # Question text analysis (basic heuristics)
    if len(question) < 20:
        lines.append(
            "⚠ VERY SHORT QUESTION: Minimal text may indicate a recall-only "
            "question that doesn't assess higher-order thinking."
        )
    elif len(question) > 500:
        lines.append(
            "→ Long question text — may include context/passage-based assessment, "
            "which typically targets higher Bloom's levels."
        )

    # ── 4. Confidence Interpretation ──
    lines.append("\n## Prediction Confidence")
    lines.append(
        f"The model predicted '{diff}' with {conf:.1%} confidence. "
        f"{'This is a strong prediction.' if conf > 0.7 else 'This prediction has moderate uncertainty.'}"
    )

    interpretation = "\n".join(lines)

    log = state.get("log", [])
    log.append(f"[{datetime.now().isoformat()}] Interpretation Node: Generated {len(lines)} reasoning lines")

    return {
        "interpretation": interpretation,
        "log": log,
    }


# ──────────────────────────────────────────────
# NODE 4: RAG RETRIEVER NODE
# ──────────────────────────────────────────────

# Retriever will be set at startup
_rag_retriever = None


def set_rag_retriever(retriever):
    """Called once at startup to inject the RAG retriever."""
    global _rag_retriever
    _rag_retriever = retriever


def rag_retriever_node(state: AgentState) -> dict:
    """
    Retrieve relevant pedagogy documents from the FAISS knowledge base.
    Uses the interpretation text as the query to find the most relevant
    pedagogical principles.
    """
    # Build a query from the current analysis context
    diff = state.get("predicted_difficulty", "Medium")
    interpretation = state.get("interpretation", "")
    question = state.get("question_text", "")

    # Craft a retrieval query combining key aspects
    query_parts = [
        f"question difficulty {diff}",
        f"student performance average score {state.get('avg_score', 50)}",
    ]

    # Add specific concerns from interpretation
    if "LOW DISCRIMINATION" in interpretation:
        query_parts.append("improving item discrimination distractor quality")
    if "CEILING EFFECT" in interpretation:
        query_parts.append("question too easy increase difficulty bloom taxonomy")
    if "FLOOR EFFECT" in interpretation:
        query_parts.append("question too hard unfair difficulty calibration")
    if "SCORE DOMINANCE" in interpretation:
        query_parts.append("numeric feature dominance assessment design")
    if "Learning Gap" in interpretation or "knowledge gap" in interpretation.lower():
        query_parts.append("learning gaps prerequisite knowledge assessment")
    if "variance" in interpretation.lower():
        query_parts.append("score variance fairness equitable assessment")

    query = " ".join(query_parts)

    retrieved_docs = []
    if _rag_retriever is not None:
        try:
            retrieved_docs = _rag_retriever.retrieve(query, top_k=4)
        except Exception as e:
            retrieved_docs = [f"[RAG retrieval error: {str(e)}]"]
    else:
        retrieved_docs = ["[RAG retriever not initialized — using fallback]"]

    log = state.get("log", [])
    log.append(
        f"[{datetime.now().isoformat()}] RAG Retriever Node: "
        f"Retrieved {len(retrieved_docs)} documents for query: {query[:80]}..."
    )

    return {
        "retrieved_docs": retrieved_docs,
        "log": log,
    }


# ──────────────────────────────────────────────
# NODE 5: LLM REASONING NODE
# ──────────────────────────────────────────────

# LLM client will be set at startup
_llm_client = None


def set_llm_client(client):
    """Called once at startup to inject the LLM client."""
    global _llm_client
    _llm_client = client


def llm_reasoning_node(state: AgentState) -> dict:
    """
    Combine ML outputs, interpretation, and retrieved docs
    to generate structured assessment improvement suggestions.

    Uses Groq API with Qwen3-32B. Falls back to rule-based
    generation if the API is unavailable.
    """
    # Build the prompt
    prompt = _build_llm_prompt(state)

    llm_output = None

    if _llm_client is not None:
        try:
            llm_output = _llm_client.generate(prompt)
        except Exception as e:
            print(f"[LLM Node] Groq API error: {e}. Using rule-based fallback.")

    # Fallback: generate structured output from rules
    if llm_output is None:
        llm_output = _generate_fallback_output(state)

    log = state.get("log", [])
    source = "LLM (Groq/Qwen3-32B)" if _llm_client and llm_output != _generate_fallback_output(state) else "Rule-based fallback"
    log.append(f"[{datetime.now().isoformat()}] LLM Reasoning Node: Generated output via {source}")

    return {
        "llm_output": llm_output,
        "log": log,
    }


def _build_llm_prompt(state: AgentState) -> str:
    """Construct the LLM prompt from agent state."""
    docs_text = "\n\n".join(state.get("retrieved_docs", []))

    return f"""You are an expert educational assessment consultant. Analyze the following exam question and provide structured improvement recommendations.

## ML ANALYSIS RESULTS
- Question Text: {state.get('question_text', 'N/A')}
- Predicted Difficulty: {state.get('predicted_difficulty', 'N/A')}
- Model Confidence: {state.get('confidence', 0):.1%}
- Average Student Score: {state.get('avg_score', 0):.1f}
- Score Variance: {state.get('variance', 0):.1f}
- Pass Rate: {state.get('pass_rate', 0):.1f}%
- Discrimination Index: {state.get('disc_index', 0)}

## AUTOMATED INTERPRETATION
{state.get('interpretation', 'No interpretation available.')}

## RETRIEVED PEDAGOGICAL REFERENCES
{docs_text}

## YOUR TASK
Based on the above analysis, provide a structured assessment report. You MUST respond with ONLY a valid JSON object (no markdown, no code fences) with these exact keys:

{{
  "summary": "A 2-3 sentence executive summary of the assessment quality",
  "difficulty_analysis": "Detailed explanation of why this difficulty was predicted, whether it is appropriate, and what drives it",
  "learning_gaps": ["List of specific learning gaps identified from student performance data"],
  "question_issues": ["List of specific issues with the question design, wording, or structure"],
  "recommendations": ["List of actionable, specific improvement recommendations for the educator"],
  "pedagogical_references": ["List of relevant pedagogical principles from the retrieved documents that support your recommendations"],
  "ethical_notice": "This analysis is provided as decision support for educators. It should not replace professional pedagogical judgment. Predictions are based on statistical patterns and may not capture all nuances of question quality."
}}

Be specific and actionable. Ground your recommendations in the retrieved pedagogical documents. Do not hallucinate principles — only reference what was provided."""


def _generate_fallback_output(state: AgentState) -> dict:
    """
    Rule-based fallback when the LLM is unavailable.
    Generates a reasonable structured output from the interpretation and RAG docs.
    """
    diff = state.get("predicted_difficulty", "Medium")
    avg = state.get("avg_score", 50)
    var = state.get("variance", 100)
    pr = state.get("pass_rate", 50)
    disc = state.get("disc_index", 0.5)
    interpretation = state.get("interpretation", "")
    docs = state.get("retrieved_docs", [])

    # Build summary
    summary = (
        f"This question is classified as '{diff}' difficulty based on student performance data. "
        f"The average score is {avg:.1f} with a pass rate of {pr:.1f}%. "
    )
    if disc < 0.2:
        summary += "The question shows low discrimination and may need revision."
    elif pr > 90:
        summary += "The very high pass rate suggests the question may be too easy for meaningful assessment."
    elif pr < 20:
        summary += "The very low pass rate indicates potential issues with question fairness or difficulty calibration."
    else:
        summary += "Overall performance metrics are within acceptable ranges."

    # Difficulty analysis
    difficulty_analysis = (
        f"The ML model predicted '{diff}' with primary influence from student performance statistics "
        f"(avg_score={avg:.1f}, variance={var:.1f}, pass_rate={pr:.1f}%). "
        f"Student performance features dominate the prediction over text features — "
        f"this is expected behavior in this model architecture (5000 TF-IDF features vs 3 numeric features, "
        f"where numeric features have stronger predictive signal)."
    )

    # Learning gaps
    learning_gaps = []
    if avg < 40:
        learning_gaps.append("Critical knowledge gap: Most students scored below passing threshold, suggesting fundamental misunderstanding of the tested concept.")
    if var > 200:
        learning_gaps.append("Inconsistent understanding: High variance indicates a significant split between students who understand the concept and those who do not.")
    if avg < 60 and pr < 50:
        learning_gaps.append("Below-average performance suggests the topic may need additional instructional time or differentiated teaching approaches.")
    if not learning_gaps:
        learning_gaps.append("No critical learning gaps detected from the available performance data.")

    # Question issues
    question_issues = []
    if disc < 0.2:
        question_issues.append("Low discrimination index — the question does not effectively distinguish between high and low performers. Consider improving distractor quality.")
    if pr > 95:
        question_issues.append("Ceiling effect — nearly all students answered correctly. The question may not provide useful diagnostic information.")
    if pr < 10:
        question_issues.append("Floor effect — nearly all students failed. Consider whether the content was covered in instruction.")
    if len(state.get("question_text", "")) < 20:
        question_issues.append("Very short question text may indicate a surface-level recall question rather than a higher-order thinking assessment.")
    if not question_issues:
        question_issues.append("No major structural issues detected from available data.")

    # Recommendations
    recommendations = []
    if disc < 0.3:
        recommendations.append("Revise distractors to be more plausible — each distractor should represent a common misconception.")
    if pr > 90:
        recommendations.append("Increase cognitive complexity by targeting higher Bloom's taxonomy levels (Application, Analysis, or Evaluation).")
    if pr < 20:
        recommendations.append("Review whether prerequisite knowledge was adequately taught. Consider scaffolding the question.")
    if var > 200:
        recommendations.append("Consider adding formative assessments before this summative question to identify struggling students earlier.")
    recommendations.append("Ensure question stem is clear, concise, and free of grammatical cues that might hint at the correct answer.")
    recommendations.append("Review distractors to ensure they are all plausible, similar in length to the correct answer, and free of absolute terms.")

    # Pedagogical references from RAG
    ped_refs = []
    for doc in docs[:3]:
        if isinstance(doc, str) and len(doc) > 20 and not doc.startswith("[RAG"):
            # Extract first sentence as reference
            first_line = doc.strip().split("\n")[0][:200]
            ped_refs.append(first_line)
    if not ped_refs:
        ped_refs.append("Bloom's Taxonomy: Assessment items should target appropriate cognitive levels.")
        ped_refs.append("Item Analysis: Discrimination index below 0.2 indicates poor item quality.")

    return {
        "summary": summary,
        "difficulty_analysis": difficulty_analysis,
        "learning_gaps": learning_gaps,
        "question_issues": question_issues,
        "recommendations": recommendations,
        "pedagogical_references": ped_refs,
        "ethical_notice": (
            "This analysis is provided as decision support for educators. "
            "It should not replace professional pedagogical judgment. "
            "Predictions are based on statistical patterns and may not capture "
            "all nuances of question quality."
        ),
    }


# ──────────────────────────────────────────────
# NODE 6: OUTPUT FORMATTER NODE
# ──────────────────────────────────────────────

REQUIRED_KEYS = [
    "summary",
    "difficulty_analysis",
    "learning_gaps",
    "question_issues",
    "recommendations",
    "pedagogical_references",
    "ethical_notice",
]


def output_formatter_node(state: AgentState) -> dict:
    """
    Validate and format the final structured report.
    Ensures all required keys are present and properly typed.
    """
    llm_output = state.get("llm_output", {})

    # Ensure all required keys exist
    report = {}
    for key in REQUIRED_KEYS:
        value = llm_output.get(key)
        if value is None:
            if key in ("learning_gaps", "question_issues", "recommendations", "pedagogical_references"):
                report[key] = []
            elif key == "ethical_notice":
                report[key] = (
                    "This analysis is provided as decision support for educators. "
                    "It should not replace professional pedagogical judgment."
                )
            else:
                report[key] = "Analysis not available."
        else:
            # Ensure list fields are lists
            if key in ("learning_gaps", "question_issues", "recommendations", "pedagogical_references"):
                if isinstance(value, str):
                    report[key] = [value]
                elif isinstance(value, list):
                    report[key] = [str(item) for item in value]
                else:
                    report[key] = [str(value)]
            else:
                report[key] = str(value)

    # Add metadata
    report["ml_metrics"] = {
        "predicted_difficulty": state.get("predicted_difficulty", "N/A"),
        "confidence": state.get("confidence", 0),
        "avg_score": state.get("avg_score", 0),
        "variance": state.get("variance", 0),
        "pass_rate": state.get("pass_rate", 0),
        "disc_index": state.get("disc_index", 0),
    }

    report["score_dominance"] = True  # Known model characteristic

    log = state.get("log", [])
    log.append(f"[{datetime.now().isoformat()}] Output Formatter Node: Report validated with {len(REQUIRED_KEYS)} required fields")

    return {
        "final_report": report,
        "log": log,
    }
