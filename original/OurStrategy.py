# Rag_System/Chunking/gmm.py

from typing import List, Tuple, Optional
import numpy as np
import logging

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans  # נוסף עבור Elbow Method
from sklearn.preprocessing import normalize  # נוסף עבור שיפור חישובי מרחק

# OPTIMIZATION: UMAP (Better manifold preservation)
try:
    import umap
except ImportError:
    umap = None

# Spacy
try:
    import spacy
except ImportError:
    spacy = None

from Rag_System.Chunking.base import ChunkingStrategy, register_chunker
from Rag_System.core.document import Document, Chunk

try:
    from Rag_System.embeddings.sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: SentenceTransformer not found. GMM Chunker will fail if initialized.")
    SentenceTransformer = None

log = logging.getLogger(__name__)


@register_chunker("gmm")
class Gmm_Chunker(ChunkingStrategy):
    """
    GMM-based semantic chunker (Sentence-level).

    שינויים חדשים:
    - שימוש ב-Elbow Method (K-Means) לקביעת מספר אשכולות אופטימלי כאשר num_clusters=None.
    - שילוב לוגיקת צמצום מימדים (UMAP/PCA).
    - סינון פערים סמנטיים (Fact-Gap) והרחבה חכמה.
    """

    def __init__(
            self,
            model_name: str,
            device: str,

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
            chunks_retrive: int = 6
    ):
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformer is required for Gmm_Chunker.")

        self.embedder = SentenceTransformer(model_name=model_name, device=device)

        self.chunks_retrive = chunks_retrive
        self.num_clusters = num_clusters
        self.probability_threshold = float(probability_threshold)
        self.soft_assignment_margin = float(soft_assignment_margin)

        self.max_gap_threshold = int(max_gap_threshold)
        self.semantic_gap_threshold = float(semantic_gap_threshold)
        self.window_expansion_k = int(window_expansion_k)

        self.max_sentences_per_chunk = int(max_sentences_per_chunk)
        self.max_chunk_words = int(max_chunk_words)
        self.max_cluster_size = int(max_cluster_size)

        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser", "lemmatizer"])
                self.nlp.add_pipe("sentencizer")
            except OSError:
                self.nlp = None
        else:
            self.nlp = None

    def strategy_name(self) -> str:
        return "gmm"

    def _find_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        מחשב את ה-Elbow Point באמצעות KMeans ו-SSE.
        עוזר לקבוע כמה נושאים (Clusters) יש במסמך בצורה דינמית.
        """
        n_samples = embeddings.shape[0]
        if n_samples < 4:
            return max(1, n_samples // 2)

        # נרמול לטובת דמיון קוסינוס (KMeans עובד על מרחק אוקלידי)
        norm_embeddings = normalize(embeddings)

        # חיפוש עד 12 קלאסטרים או כמות המשפטים
        max_k = min(n_samples - 1, 12)
        sse = []

        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans.fit(norm_embeddings)
            sse.append(kmeans.inertia_)

        if len(sse) < 3:
            return 2

        # מציאת הנקודה בה קצב שיפור ה-SSE דועך (המרפק)
        deltas = np.diff(sse)
        double_deltas = np.diff(deltas)
        optimal_k = int(np.argmax(double_deltas) + 2)
        return optimal_k

    def _get_sentence_spans(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
            spans = []
            start_offset = 0
            for sent in sentences:
                start = text.find(sent, start_offset)
                if start == -1: start = start_offset
                end = start + len(sent)
                spans.append((start, end))
                start_offset = end
            return sentences, spans

        from Rag_System.Utils.tokenization import get_sentence_tokenizer
        tokenizer = get_sentence_tokenizer()
        spans = list(tokenizer.span_tokenize(text))
        valid_sentences = [text[s:e] for s, e in spans if len(text[s:e].strip()) > 5]
        return valid_sentences, spans

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])

    def _chunk_impl(self, document: Document) -> List[Chunk]:
        # 1) הכנת טקסט ו-Embeddings
        if hasattr(document, "sentences") and document.sentences:
            sentences = document.sentences
        else:
            sentences, _ = self._get_sentence_spans(document.text)
            document.sentences = sentences

        if len(sentences) < 2:
            return [self._create_simple_chunk(document, document.text, 0)]

        if hasattr(document, "sentence_embeddings") and document.sentence_embeddings is not None:
            embeddings = document.sentence_embeddings
        else:
            embeddings = self.embedder.encode(sentences, task="document")

        embeddings = np.asarray(embeddings)
        n_sent = len(sentences)

        # 2) קביעת מספר אשכולות - כאן נכנס ה-ELBOW
        if self.num_clusters is None:
            n_clusters = self._find_optimal_k(embeddings)
            log.info(f"Elbow Method selected {n_clusters} clusters for doc {document.id}")
        else:
            n_clusters = min(int(self.num_clusters), n_sent)

        # 3) צמצום מימדים (UMAP/PCA) לשיפור ביצועי GMM
        n_samples = embeddings.shape[0]
        reduced_embeddings = embeddings
        use_umap = (umap is not None) and (n_samples > 10)

        if use_umap:
            try:
                reducer = umap.UMAP(n_neighbors=min(n_samples - 1, 15), n_components=min(n_samples - 2, 16),
                                    min_dist=0.0, metric="cosine", random_state=42, n_jobs=1)
                reduced_embeddings = reducer.fit_transform(embeddings)
            except Exception:
                use_umap = False

        if not use_umap:
            pca = PCA(n_components=min(n_samples, 32), random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings)

        # 4) הרצת GMM עם Soft Assignment
        clustered_indices = {}
        try:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
            gmm.fit(reduced_embeddings)
            probs = gmm.predict_proba(reduced_embeddings)

            for idx, p in enumerate(probs):
                primary = int(np.argmax(p))
                primary_prob = float(p[primary])
                clustered_indices.setdefault(primary, []).append(idx)

                for c_id, prob_val in enumerate(p):
                    if c_id == primary: continue
                    if (prob_val > self.probability_threshold and
                            (primary_prob - prob_val) < self.soft_assignment_margin):
                        clustered_indices.setdefault(int(c_id), []).append(idx)
        except Exception as e:
            clustered_indices = {0: list(range(n_sent))}

        # 5) בניית חלונות וסינון פערים סמנטיים (Fact-Gap Filter)
        final_windows: List[Tuple[int, int]] = []
        for indices in clustered_indices.values():
            if not indices: continue
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
            if current_group: sub_groups.append(current_group)

            for group in sub_groups:
                s, e = group[0], group[-1]
                # Smart Expansion
                for _ in range(self.window_expansion_k):
                    if s > 0 and self._cos_sim(embeddings[s], embeddings[s - 1]) >= self.semantic_gap_threshold:
                        s -= 1
                    if e < n_sent - 1 and self._cos_sim(embeddings[e],
                                                        embeddings[e + 1]) >= self.semantic_gap_threshold:
                        e += 1

                if (e - s + 1) > self.max_sentences_per_chunk:
                    e = s + self.max_sentences_per_chunk - 1
                final_windows.append((s, e))

        # 6) מיזוג חלונות חופפים ויצירת צ'אנקים סופיים
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

        chunks = []
        for i, (start, end) in enumerate(merged):
            text = " ".join(sentences[start:end + 1]).strip()
            if self.max_chunk_words and len(text.split()) > self.max_chunk_words:
                continue
            c = Chunk(id=f"{document.id}_gmm_{i}", text=text, document_id=document.id,
                      chunk_index=i, metadata=document.metadata.copy())
            c.embedding = [0.0] * embeddings.shape[1]
            chunks.append(c)

        return chunks if chunks else [self._create_simple_chunk(document, document.text, 0)]

    def _create_simple_chunk(self, doc: Document, text: str, idx: int) -> Chunk:
        return Chunk(id=f"{doc.id}_simple_{idx}", text=text, document_id=doc.id, chunk_index=idx,
                     metadata=doc.metadata.copy())