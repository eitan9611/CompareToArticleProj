"""
ingestion.py — Document Ingestion Pipeline
===========================================

Processes documents through the GMM chunker, embeds them with the
E5 embedding model, and indexes them into ChromaDB collections.

Each dataset gets its own ChromaDB collection for isolated retrieval.
"""

import logging
from typing import List

from langchain_core.documents import Document
from tqdm import tqdm

log = logging.getLogger(__name__)


def ingest_dataset(
        dataset_name: str,
        documents: List[Document],
        chunker,
        embedding_model,
        chroma_client,
        batch_size: int = 64,
) -> None:
    """
    Ingest a list of documents into ChromaDB.

    Pipeline:  Document → GMM Chunk → Embed → Upsert to ChromaDB

    Parameters
    ----------
    dataset_name : str
        Name for the ChromaDB collection (e.g., "squad", "covidqa").
    documents : List[Document]
        LangChain Documents to process.
    chunker : GmmChunker
        The GMM-based chunker instance.
    embedding_model : SentenceTransformer
        The embedding model for encoding chunks.
    chroma_client : chromadb.ClientAPI
        The ChromaDB client instance.
    batch_size : int
        Number of embeddings to upsert in a single batch.
    """

    # Create or get the collection for this dataset
    # Delete existing collection to ensure fresh ingestion
    try:
        chroma_client.delete_collection(name=dataset_name)
        log.info(f"Deleted existing collection '{dataset_name}'.")
    except Exception:
        pass  # Collection doesn't exist yet

    collection = chroma_client.get_or_create_collection(
        name=dataset_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )

    print(f"\n[Ingestion] Processing {len(documents)} documents for '{dataset_name}'...")

    all_chunks = []
    all_ids = []
    all_metadatas = []

    # ── Step 1: Chunk all documents ────────────────────────────
    for doc in tqdm(documents, desc=f"Chunking {dataset_name}"):
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            chunk_id = (
                f"{dataset_name}_{chunk.metadata.get('parent_doc_id', 'unk')}"
                f"_c{chunk.metadata.get('chunk_index', 0)}"
            )
            all_chunks.append(chunk.page_content)
            all_ids.append(chunk_id)
            all_metadatas.append(chunk.metadata)

    print(f"[Ingestion] Generated {len(all_chunks)} chunks from "
          f"{len(documents)} documents "
          f"(avg {len(all_chunks) / max(len(documents), 1):.1f} chunks/doc).")

    # ── Step 2: Embed all chunks ───────────────────────────────
    print(f"[Ingestion] Embedding {len(all_chunks)} chunks...")

    # E5 requires "passage: " prefix for document encoding
    prefixed_chunks = [f"passage: {text}" for text in all_chunks]
    all_embeddings = embedding_model.encode(
        prefixed_chunks,
        show_progress_bar=True,
        batch_size=batch_size,
    )
    all_embeddings = all_embeddings.tolist()

    # ── Step 3: Upsert into ChromaDB in batches ───────────────
    print(f"[Ingestion] Indexing into ChromaDB collection '{dataset_name}'...")

    for start in range(0, len(all_chunks), batch_size):
        end = min(start + batch_size, len(all_chunks))
        collection.add(
            ids=all_ids[start:end],
            embeddings=all_embeddings[start:end],
            documents=all_chunks[start:end],
            metadatas=all_metadatas[start:end],
        )

    print(f"[Ingestion] ✓ Indexed {collection.count()} chunks "
          f"into '{dataset_name}' collection.\n")
