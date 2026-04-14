import random
from datasets import load_dataset
from langchain_core.documents import Document


class SquadLoader:
    def __init__(self, sample_size=150, seed=433):
        print(f"Loading SQuAD validation split (Seed: {seed})...")
        # המאמר השתמש ב-Validation set לניסויים [cite: 375]
        dataset = load_dataset("squad", split="validation")

        # דגימה אקראית לפי ה-Seed מהמאמר [cite: 376]
        random.seed(seed)
        indices = random.sample(range(len(dataset)), sample_size)
        self.samples = dataset.select(indices)

    def get_data(self):
        documents = []
        evaluation_set = []

        for i, row in enumerate(self.samples):
            doc_id = f"squad_{i}"
            # יצירת מסמך LangChain עבור ה-Index
            documents.append(Document(
                page_content=row['context'],
                metadata={"source": "squad", "doc_id": doc_id}
            ))

            # יצירת סט שאלות להערכה
            evaluation_set.append({
                "question": row['question'],
                "ground_truth": row['answers']['text'][0],
                "doc_id": doc_id
            })

        return documents, evaluation_set