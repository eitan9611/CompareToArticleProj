"""
nq_loader.py — NarrativeQA Dataset Loader
==========================================

Loads 100 balanced samples (50 books + 50 movies) from the NarrativeQA
test split using seed 42, matching the paper's experimental protocol.

KEY CHANGE from original: Adds a 'kind' field (book/movie) to both
Document metadata and evaluation entries so that metrics can be computed
separately for NarrativeQA (Books) and NarrativeQA (Movies).

Returns:
    documents:      List of LangChain Document objects (summaries for ingestion)
    evaluation_set: List of dicts with {question, ground_truth, doc_id, kind}
"""

import random
from datasets import load_dataset
from langchain_core.documents import Document


class NQLoader:
    """
    Loads and samples from the NarrativeQA test split.

    The paper used 100 balanced samples (50 books + 50 movies) from
    the test set with seed 42. Each document uses the summary text
    as the context for retrieval (since full texts are very long).
    """

    def __init__(self, total_samples: int = 100, seed: int = 42):
        print(f"[NQLoader] Loading NarrativeQA test split "
              f"(balanced {total_samples // 2}/{total_samples // 2}, seed={seed})...")

        dataset = load_dataset("narrativeqa", split="test")

        # Separate books and movies for balanced sampling
        books = [row for row in dataset if row['document']['kind'] == 'book']
        movies = [row for row in dataset if row['document']['kind'] == 'movie']

        print(f"[NQLoader] Found {len(books)} books, {len(movies)} movies in test set.")

        # Reproducible balanced sampling
        random.seed(seed)
        sampled_books = random.sample(books, total_samples // 2)
        sampled_movies = random.sample(movies, total_samples // 2)

        self.samples = sampled_books + sampled_movies
        print(f"[NQLoader] Sampled {len(self.samples)} total "
              f"({len(sampled_books)} books + {len(sampled_movies)} movies).")

    def get_data(self):
        """
        Returns:
            documents:      List[Document] — unique document summaries for ingestion
            evaluation_set: List[dict]     — question/answer pairs with kind field
        """
        documents = []
        evaluation_set = []
        processed_docs = {}  # Track already-indexed document IDs

        for i, row in enumerate(self.samples):
            doc_id = row['document']['id']
            context = row['document']['summary']['text']
            kind = row['document']['kind']  # 'book' or 'movie'

            # Only index each document once (multiple questions may share a doc)
            if doc_id not in processed_docs:
                processed_docs[doc_id] = True
                documents.append(Document(
                    page_content=context,
                    metadata={
                        "source": "narrative_qa",
                        "doc_id": doc_id,
                        "kind": kind,  # NEW: enables separate book/movie scoring
                    }
                ))

            evaluation_set.append({
                "question": row['question']['text'],
                "ground_truth": row['answers'][0]['text'],
                "doc_id": doc_id,
                "kind": kind,  # NEW: enables separate book/movie scoring
            })

        print(f"[NQLoader] {len(documents)} unique documents, "
              f"{len(evaluation_set)} evaluation pairs.")
        return documents, evaluation_set
