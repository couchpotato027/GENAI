"""
Agent State Definition
======================
TypedDict that flows through the LangGraph pipeline.
Each node reads from and writes to this shared state.
"""

from typing import TypedDict, List, Optional


class AgentState(TypedDict, total=False):
    """Shared state for the LangGraph assessment agent pipeline."""

    # ── Input (set by Input Node) ──
    question_text: str
    student_scores: str

    # ── ML Analysis (set by ML Analysis Node) ──
    predicted_difficulty: str
    confidence: float
    avg_score: float
    variance: float
    pass_rate: float
    disc_index: float

    # ── Interpretation (set by Interpretation Node) ──
    interpretation: str

    # ── RAG Retrieval (set by RAG Retriever Node) ──
    retrieved_docs: List[str]

    # ── LLM Output (set by LLM Reasoning Node) ──
    llm_output: dict

    # ── Final Report (set by Output Formatter Node) ──
    final_report: dict

    # ── Logging ──
    log: List[str]
