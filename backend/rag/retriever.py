"""
Pedagogy RAG Retriever
========================
Chunks, embeds, and indexes the pedagogical knowledge base
using sentence-transformers + FAISS for fast similarity search.

Falls back to keyword-based retrieval if the embedding model
cannot be loaded (e.g., offline environment).
"""

import os
import re
import json
import numpy as np
from typing import List, Optional

from .knowledge_base import PEDAGOGY_DOCUMENTS


class PedagogyRetriever:
    """
    RAG retriever backed by FAISS index of pedagogical documents.
    
    Usage:
        retriever = PedagogyRetriever()
        retriever.build_index()
        docs = retriever.retrieve("question too easy bloom taxonomy", top_k=4)
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self.index = None
        self.embed_model = None
        self._use_fallback = False

    def build_index(self):
        """Chunk documents, embed them, and build FAISS index."""
        print("[RAG] Building pedagogy knowledge base index...")

        # Step 1: Chunk documents
        self.chunks = self._chunk_documents(PEDAGOGY_DOCUMENTS)
        print(f"[RAG] Created {len(self.chunks)} chunks from {len(PEDAGOGY_DOCUMENTS)} documents")

        # Step 2: Try to load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[RAG] Loaded embedding model: all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[RAG] WARNING: Could not load embedding model: {e}")
            print("[RAG] Falling back to keyword-based retrieval")
            self._use_fallback = True
            return

        # Step 3: Embed all chunks
        try:
            embeddings = self.embed_model.encode(
                self.chunks,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings = np.array(embeddings, dtype="float32")
            print(f"[RAG] Embedded {len(self.chunks)} chunks (dim={embeddings.shape[1]})")
        except Exception as e:
            print(f"[RAG] WARNING: Embedding failed: {e}")
            self._use_fallback = True
            return

        # Step 4: Build FAISS index
        try:
            import faiss
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim with normalized vectors)
            self.index.add(embeddings)
            print(f"[RAG] FAISS index built with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"[RAG] WARNING: FAISS index build failed: {e}")
            self._use_fallback = True

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        """
        Retrieve top-k relevant pedagogy passages for a query.
        
        Uses FAISS similarity search if available, falls back
        to keyword matching otherwise.
        """
        if not self.chunks:
            return ["No pedagogy knowledge base available."]

        if self._use_fallback or self.index is None:
            return self._keyword_retrieve(query, top_k)

        try:
            # Embed query
            query_vec = self.embed_model.encode(
                [query],
                normalize_embeddings=True,
            )
            query_vec = np.array(query_vec, dtype="float32")

            # Search FAISS
            scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks):
                    results.append(self.chunks[idx])

            return results if results else self._keyword_retrieve(query, top_k)

        except Exception as e:
            print(f"[RAG] FAISS search failed: {e}, using keyword fallback")
            return self._keyword_retrieve(query, top_k)

    def _chunk_documents(self, documents: List[str]) -> List[str]:
        """Split documents into overlapping chunks for better retrieval."""
        chunks = []
        for doc in documents:
            # Split by paragraphs first
            paragraphs = [p.strip() for p in doc.split("\n\n") if p.strip()]
            
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    current_chunk += ("\n\n" + para if current_chunk else para)
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

        # Remove very short chunks
        chunks = [c for c in chunks if len(c) > 50]
        return chunks

    def _keyword_retrieve(self, query: str, top_k: int) -> List[str]:
        """Fallback: simple keyword-based retrieval using term overlap."""
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        scored = []
        for i, chunk in enumerate(self.chunks):
            chunk_terms = set(re.findall(r'\w+', chunk.lower()))
            overlap = len(query_terms & chunk_terms)
            # Normalize by chunk length to avoid bias toward long chunks
            score = overlap / (len(chunk_terms) + 1)
            scored.append((score, i))

        scored.sort(reverse=True)
        return [self.chunks[idx] for _, idx in scored[:top_k]]
