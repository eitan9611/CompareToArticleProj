"""
pipeline — Core RAG pipeline modules (ingestion, retrieval, generation).
"""

from pipeline.ingestion import ingest_dataset
from pipeline.retrieval import retrieve
from pipeline.generation import LlamaGenerator

__all__ = ["ingest_dataset", "retrieve", "LlamaGenerator"]
