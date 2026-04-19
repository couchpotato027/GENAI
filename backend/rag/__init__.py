"""
RAG Module — Pedagogy Knowledge Base & FAISS Retrieval
======================================================
Provides retrieval-augmented generation capability using
a curated pedagogical knowledge base indexed with FAISS.
"""

from .retriever import PedagogyRetriever

__all__ = ["PedagogyRetriever"]
