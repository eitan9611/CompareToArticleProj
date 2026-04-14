"""
covidqa_loader.py — CovidQA Dataset Loader
==========================================

Loads all 124 question-answer pairs from the CovidQA dataset,
matching the paper's experimental protocol (no sub-sampling).

Returns:
    documents:      List of LangChain Document objects (unique contexts)
    evaluation_set: List of dicts with {question, ground_truth, doc_id}
"""

from datasets import load_dataset
from langchain_core.documents import Document


class CovidQALoader:
    """
    Loads the full CovidQA (covid_qa_deepset) dataset.

    The paper used all 124 question-answer pairs without sub-sampling.
    Contexts are de-duplicated to avoid indexing the same passage twice.
    """

    def __init__(self):
        print("[CovidQALoader] Loading full CovidQA dataset (124 samples)...")
        self.dataset = load_dataset("covid_qa_deepset", split="train")
        print(f"[CovidQALoader] Loaded {len(self.dataset)} samples.")

    def get_data(self):
        """
        Returns:
            documents:      List[Document] — unique contexts for ingestion
            evaluation_set: List[dict]     — question/answer pairs for evaluation
        """
        documents = []
        evaluation_set = []
        processed_contexts = {}  # De-duplicate identical contexts

        for i, row in enumerate(self.dataset):
            context = row['context']

            # Only create a new Document if this context hasn't been seen before
            if context not in processed_contexts:
                doc_id = f"covid_doc_{len(processed_contexts)}"
                processed_contexts[context] = doc_id
                documents.append(Document(
                    page_content=context,
                    metadata={"source": "covid_qa", "doc_id": doc_id}
                ))

            evaluation_set.append({
                "question": row['question'],
                "ground_truth": row['answers']['text'][0],
                "doc_id": processed_contexts[context]
            })

        print(f"[CovidQALoader] {len(documents)} unique documents, "
              f"{len(evaluation_set)} evaluation pairs.")
        return documents, evaluation_set
