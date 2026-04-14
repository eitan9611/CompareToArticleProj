"""
gmm_chunker.py — GMM-based Semantic Chunker (Refactored for LangChain)
======================================================================

This module is a refactored version of OurStrategy.py (Rag_System/Chunking/gmm.py).

WHAT CHANGED:
  - Removed all Rag_System internal imports (ChunkingStrategy base, register_chunker,
    Document/Chunk from Rag_System.core, SentenceTransformer from Rag_System.embeddings).
  - Input:  langchain_core.documents.Document  (uses .page_content for text)
  - Output: List[langchain_core.documents.Document]  (one per chunk, with metadata)
  - Embedding model is injected via constructor (dependency injection).
  - Sentence tokenization fallback uses nltk.sent_tokenize instead of Rag_System.Utils.

WHAT DID NOT CHANGE:
  - All GMM clustering logic is preserved exactly.
  - Elbow Method (K-Means) for optimal cluster count.
  - UMAP / PCA dimensionality reduction.
  - Soft assignment with probability threshold and margin.
  - Semantic gap filtering and smart window expansion.
  - Overlapping window merging.
  - All default hyperparameters.
"""

from typing import List, Tuple, Optional
import numpy as np
import logging

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from langchain_core.documents import Document

# OPTIMIZATION: UMAP (Better manifold preservation)
try:
    import umap
except ImportError:
    umap = None

# Spacy sentencizer
try:
    import spacy
except ImportError:
    spacy = None

# NLTK fallback for sentence tokenization
try:
    import nltk
    nltk.download('punkt_tab', quiet=True)
except ImportError:
    nltk = None

log = logging.getLogger(__name__)


class GmmChunker:
    """
    GMM-based semantic chunker (Sentence-level).

    Uses Gaussian Mixture Models with soft assignment to cluster sentences
    by semantic similarity, then builds contiguous text chunks from the
    clustered sentence indices. Supports automatic cluster count detection
    via the Elbow Method (K-Means SSE analysis).

    Parameters
    ----------
    embedding_model : SentenceTransformer
        A pre-loaded SentenceTransformer instance for encoding sentences.
    num_clusters : int or None
        Fixed number of clusters. If None, uses Elbow Method to auto-detect.
    probability_threshold : float
        Minimum probability for soft-assigning a sentence to a secondary cluster.
    soft_assignment_margin : float
        Maximum gap between primary and secondary cluster probabilities
        for soft assignment to occur.
    max_gap_threshold : int
        Maximum allowed index gap between consecutive sentences in the same
        sub-group before splitting into separate chunks.
    semantic_gap_threshold : float
        Minimum cosine similarity between consecutive sentences to keep
        them in the same sub-group.
    window_expansion_k : int
        Number of expansion steps to try when growing chunk boundaries.
    max_sentences_per_chunk : int
        Hard cap on the number of sentences per chunk.
    max_chunk_words : int
        Hard cap on the number of words per chunk (chunks exceeding this
        are dropped as they are likely noise / concatenation artifacts).
    """

    def __init__(
            self,
            embedding_model,  # SentenceTransformer instance (injected)

            # Clustering
            num_clusters: Optional[int] = None,
            probability_threshold: float = 0.85,
            soft_assignment_margin: float = 0.15,

            # Windowing
            max_gap_threshold: int = 1,
            semantic_gap_threshold: float = 0.75,
            window_expansion_k: int = 1,

            # Chunk size safety
            max_sentences_per_chunk: int = 12,
            max_chunk_words: int = 300,

            # Compatibility
            max_cluster_size: int = 1500,
    ):
        self.embedder = embedding_model

        self.num_clusters = num_clusters
        self.probability_threshold = float(probability_threshold)
        self.soft_assignment_margin = float(soft_assignment_margin)

        self.max_gap_threshold = int(max_gap_threshold)
        self.semantic_gap_threshold = float(semantic_gap_threshold)
        self.window_expansion_k = int(window_expansion_k)

        self.max_sentences_per_chunk = int(max_sentences_per_chunk)
        self.max_chunk_words = int(max_chunk_words)
        self.max_cluster_size = int(max_cluster_size)

        # Initialize spacy sentencizer if available
        if spacy:
            try:
                self.nlp = spacy.load(
                    "en_core_web_sm",
                    disable=["ner", "tagger", "parser", "lemmatizer"]
                )
                self.nlp.add_pipe("sentencizer")
            except OSError:
                log.warning("spacy model 'en_core_web_sm' not found. "
                            "Falling back to NLTK sentence tokenizer.")
                self.nlp = None
        else:
            self.nlp = None

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a LangChain Document into semantically coherent sub-documents.

        Parameters
        ----------
        document : langchain_core.documents.Document
            The input document to chunk. Text is read from .page_content.

        Returns
        -------
        List[Document]
            A list of LangChain Document objects, one per chunk. Each chunk's
            metadata includes: chunk_index, parent_doc_id, plus all original
            metadata from the parent document.
        """
        text = document.page_content
        doc_id = document.metadata.get("doc_id", "unknown")

        # ── Step 1: Sentence Tokenization ──────────────────────
        sentences = self._get_sentences(text)

        if len(sentences) < 2:
            # Single-sentence document → return as-is
            return [Document(
                page_content=text,
                metadata={
                    **document.metadata,
                    "chunk_index": 0,
                    "parent_doc_id": doc_id,
                }
            )]

        # ── Step 2: Encode sentences ───────────────────────────
        # The E5 model expects "passage: " prefix for document encoding
        prefixed_sentences = [f"passage: {s}" for s in sentences]
        embeddings = self.embedder.encode(prefixed_sentences, show_progress_bar=False)
        embeddings = np.asarray(embeddings)
        n_sent = len(sentences)

        # ── Step 3: Determine cluster count (Elbow Method) ─────
        if self.num_clusters is None:
            n_clusters = self._find_optimal_k(embeddings)
            log.info(f"Elbow Method selected {n_clusters} clusters for doc {doc_id}")
        else:
            n_clusters = min(int(self.num_clusters), n_sent)

        # ── Step 4: Dimensionality reduction (UMAP / PCA) ─────
        n_samples = embeddings.shape[0]
        reduced_embeddings = embeddings
        use_umap = (umap is not None) and (n_samples > 10)

        if use_umap:
            try:
                reducer = umap.UMAP(
                    n_neighbors=min(n_samples - 1, 15),
                    n_components=min(n_samples - 2, 16),
                    min_dist=0.0,
                    metric="cosine",
                    random_state=42,
                    n_jobs=1
                )
                reduced_embeddings = reducer.fit_transform(embeddings)
            except Exception:
                use_umap = False

        if not use_umap:
            pca = PCA(n_components=min(n_samples, 32), random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings)

        # ── Step 5: GMM with Soft Assignment ───────────────────
        clustered_indices = {}
        try:
            gmm = GaussianMixture(
                n_components=n_clusters, random_state=42, n_init=5
            )
            gmm.fit(reduced_embeddings)
            probs = gmm.predict_proba(reduced_embeddings)

            for idx, p in enumerate(probs):
                primary = int(np.argmax(p))
                primary_prob = float(p[primary])
                clustered_indices.setdefault(primary, []).append(idx)

                # Soft assignment: also assign to secondary clusters if close enough
                for c_id, prob_val in enumerate(p):
                    if c_id == primary:
                        continue
                    if (prob_val > self.probability_threshold and
                            (primary_prob - prob_val) < self.soft_assignment_margin):
                        clustered_indices.setdefault(int(c_id), []).append(idx)

        except Exception as e:
            log.warning(f"GMM fitting failed for doc {doc_id}: {e}. "
                        f"Falling back to single cluster.")
            clustered_indices = {0: list(range(n_sent))}

        # ── Step 6: Build windows with semantic gap filtering ──
        final_windows: List[Tuple[int, int]] = []
        for indices in clustered_indices.values():
            if not indices:
                continue
            indices = sorted(set(indices))

            sub_groups = []
            current_group = [indices[0]]
            for i in range(1, len(indices)):
                prev, curr = indices[i - 1], indices[i]
                gap = curr - prev
                sim = self._cos_sim(embeddings[prev], embeddings[curr])

                if gap > self.max_gap_threshold or sim < self.semantic_gap_threshold:
                    sub_groups.append(current_group)
                    current_group = []
                current_group.append(curr)
            if current_group:
                sub_groups.append(current_group)

            for group in sub_groups:
                s, e = group[0], group[-1]

                # Smart Expansion: grow boundaries if neighbors are similar
                for _ in range(self.window_expansion_k):
                    if (s > 0 and
                            self._cos_sim(embeddings[s], embeddings[s - 1])
                            >= self.semantic_gap_threshold):
                        s -= 1
                    if (e < n_sent - 1 and
                            self._cos_sim(embeddings[e], embeddings[e + 1])
                            >= self.semantic_gap_threshold):
                        e += 1

                # Enforce max sentences per chunk
                if (e - s + 1) > self.max_sentences_per_chunk:
                    e = s + self.max_sentences_per_chunk - 1
                final_windows.append((s, e))

        # ── Step 7: Merge overlapping windows → final chunks ──
        final_windows.sort()
        merged = []
        if final_windows:
            curr_s, curr_e = final_windows[0]
            for next_s, next_e in final_windows[1:]:
                if next_s <= curr_e:
                    curr_e = max(curr_e, next_e)
                else:
                    merged.append((curr_s, curr_e))
                    curr_s, curr_e = next_s, next_e
            merged.append((curr_s, curr_e))

        # ── Step 8: Create LangChain Document objects ──────────
        chunks = []
        for i, (start, end) in enumerate(merged):
            chunk_text = " ".join(sentences[start:end + 1]).strip()

            # Skip chunks that exceed word limit (likely noise)
            if self.max_chunk_words and len(chunk_text.split()) > self.max_chunk_words:
                continue

            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "parent_doc_id": doc_id,
                }
            )
            chunks.append(chunk_doc)

        # Fallback: if no valid chunks produced, return the full document
        if not chunks:
            chunks = [Document(
                page_content=text,
                metadata={
                    **document.metadata,
                    "chunk_index": 0,
                    "parent_doc_id": doc_id,
                }
            )]

        return chunks

    # ──────────────────────────────────────────────────────────
    # Internal Methods (logic preserved from OurStrategy.py)
    # ──────────────────────────────────────────────────────────

    def _find_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        Compute the Elbow Point using KMeans SSE to dynamically determine
        how many topics (clusters) exist in the document.
        """
        n_samples = embeddings.shape[0]
        if n_samples < 4:
            return max(1, n_samples // 2)

        # Normalize for cosine-like behavior with KMeans (Euclidean distance)
        norm_embeddings = normalize(embeddings)

        # Search up to 12 clusters or number of sentences
        max_k = min(n_samples - 1, 12)
        sse = []

        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans.fit(norm_embeddings)
            sse.append(kmeans.inertia_)

        if len(sse) < 3:
            return 2

        # Find the "elbow" — the point where rate of SSE improvement drops
        deltas = np.diff(sse)
        double_deltas = np.diff(deltas)
        optimal_k = int(np.argmax(double_deltas) + 2)
        return optimal_k

    def _get_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spacy (preferred) or NLTK (fallback).
        Filters out very short sentences (≤5 chars) which are typically noise.
        """
        if self.nlp:
            doc = self.nlp(text)
            sentences = [
                sent.text.strip() for sent in doc.sents
                if len(sent.text.strip()) > 5
            ]
            return sentences

        # Fallback: NLTK sentence tokenizer
        if nltk:
            raw = nltk.sent_tokenize(text)
            return [s.strip() for s in raw if len(s.strip()) > 5]

        # Last resort: split on periods
        log.warning("No sentence tokenizer available. Splitting on periods.")
        return [s.strip() + '.' for s in text.split('.')
                if len(s.strip()) > 5]

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(
            cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
        )
