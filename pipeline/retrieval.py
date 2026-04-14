"""
retrieval.py — Passage Retrieval from ChromaDB
===============================================

Queries a ChromaDB collection using an E5-embedded query to retrieve
the top-K most relevant passages for a given question.
"""

import logging
from typing import List

log = logging.getLogger(__name__)


def retrieve(
        query: str,
        embedding_model,
        chroma_collection,
        top_k: int = 10,
) -> List[str]:
    """
    Retrieve the top-K most relevant passages for a query.

    Parameters
    ----------
    query : str
        The question to search for.
    embedding_model : SentenceTransformer
        The embedding model (must be the same as used during ingestion).
    chroma_collection : chromadb.Collection
        The ChromaDB collection to search.
    top_k : int
        Number of passages to retrieve (default: 10, per the paper).

    Returns
    -------
    List[str]
        The top-K passage texts, ordered by relevance (most relevant first).
    """

    # E5 requires "query: " prefix for query encoding
    query_embedding = embedding_model.encode(
        [f"query: {query}"],
        show_progress_bar=False
    ).tolist()

    # Query ChromaDB
    results = chroma_collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, chroma_collection.count()),
        include=["documents"]
    )

    # Extract passage texts from results
    # ChromaDB returns results as List[List[str]], we want the inner list
    passages = results["documents"][0] if results["documents"] else []

    return passages
