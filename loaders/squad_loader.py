"""
squad_loader.py — SQuAD Dataset Loader
=======================================

Loads 150 samples from the SQuAD validation split using seed 433,
matching the paper's experimental protocol.

Returns:
    documents:      List of LangChain Document objects (contexts for ingestion)
    evaluation_set: List of dicts with {question, ground_truth, doc_id}
"""

import random
from datasets import load_dataset
from langchain_core.documents import Document


class SquadLoader:
    """
    Loads and samples from the SQuAD v1 validation split.

    The paper used 150 randomly sampled question-context pairs from
    the validation set with seed 433 for reproducibility.
    """

    def __init__(self, sample_size: int = 150, seed: int = 433):
        print(f"[SquadLoader] Loading SQuAD validation split "
              f"(sample_size={sample_size}, seed={seed})...")

        dataset = load_dataset("squad", split="validation")

        # Reproducible random sampling matching the paper's protocol
        random.seed(seed)
        indices = random.sample(range(len(dataset)), sample_size)
        self.samples = dataset.select(indices)

        print(f"[SquadLoader] Loaded {len(self.samples)} samples.")

    def get_data(self):
        """
        Returns:
            documents:      List[Document] — unique contexts for ingestion
            evaluation_set: List[dict]     — question/answer pairs for evaluation
        """
        documents = []
        evaluation_set = []

        for i, row in enumerate(self.samples):
            doc_id = f"squad_{i}"

            # Create a LangChain Document for the context passage
            documents.append(Document(
                page_content=row['context'],
                metadata={"source": "squad", "doc_id": doc_id}
            ))

            # Create the evaluation entry
            evaluation_set.append({
                "question": row['question'],
                "ground_truth": row['answers']['text'][0],
                "doc_id": doc_id
            })

        return documents, evaluation_set
