"""
LangGraph Agent Pipeline
=========================
Defines the StateGraph that wires the 6 nodes into a sequential pipeline:

    Input → ML Analysis → Interpretation → RAG Retrieval → LLM Reasoning → Output

Uses LangGraph's StateGraph with explicit state management.
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    input_node,
    ml_analysis_node,
    interpretation_node,
    rag_retriever_node,
    llm_reasoning_node,
    output_formatter_node,
)


def build_agent_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph for the assessment agent.
    
    Pipeline flow:
        input → ml_analysis → interpretation → rag_retrieval → llm_reasoning → output_formatter → END
    """
    graph = StateGraph(AgentState)

    # Add all 6 nodes
    graph.add_node("input", input_node)
    graph.add_node("ml_analysis", ml_analysis_node)
    graph.add_node("interpretation", interpretation_node)
    graph.add_node("rag_retrieval", rag_retriever_node)
    graph.add_node("llm_reasoning", llm_reasoning_node)
    graph.add_node("output_formatter", output_formatter_node)

    # Wire nodes sequentially
    graph.set_entry_point("input")
    graph.add_edge("input", "ml_analysis")
    graph.add_edge("ml_analysis", "interpretation")
    graph.add_edge("interpretation", "rag_retrieval")
    graph.add_edge("rag_retrieval", "llm_reasoning")
    graph.add_edge("llm_reasoning", "output_formatter")
    graph.add_edge("output_formatter", END)

    return graph


# Compile the graph once at module level
_compiled_graph = None


def get_compiled_graph():
    """Get or create the compiled LangGraph pipeline."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_agent_graph()
        _compiled_graph = graph.compile()
    return _compiled_graph


def run_agent_pipeline(question_text: str, student_scores: str) -> dict:
    """
    Run the full agentic assessment pipeline.
    
    Parameters
    ----------
    question_text  : The exam question string.
    student_scores : Comma-separated student scores.
    
    Returns
    -------
    dict with the structured assessment report and execution log.
    """
    compiled = get_compiled_graph()

    # Initialize state with user inputs
    initial_state = {
        "question_text": question_text,
        "student_scores": student_scores,
        "log": [],
    }

    # Run the pipeline
    final_state = compiled.invoke(initial_state)

    # Return the final report + log
    result = final_state.get("final_report", {})
    result["execution_log"] = final_state.get("log", [])

    return result
