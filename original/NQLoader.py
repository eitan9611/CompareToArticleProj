import random
from datasets import load_dataset
from langchain_core.documents import Document


class NQLoader:
    def __init__(self, total_samples=100, seed=42):
        print(f"Loading NarrativeQA test split (Balanced 50/50, Seed: {seed})...")
        # המאמר השתמש ב-Test set [cite: 455]
        dataset = load_dataset("narrativeqa", split="test")

        # הפרדה לספרים וסרטים לצורך דגימה מאוזנת [cite: 460]
        books = [row for row in dataset if row['document']['kind'] == 'book']
        movies = [row for row in dataset if row['document']['kind'] == 'movie']

        random.seed(seed)
        sampled_books = random.sample(books, total_samples // 2)
        sampled_movies = random.sample(movies, total_samples // 2)

        self.samples = sampled_books + sampled_movies

    def get_data(self):
        documents = []
        evaluation_set = []
        processed_docs = {}

        for i, row in enumerate(self.samples):
            doc_id = row['document']['id']
            context = row['document']['summary']['text']

            if doc_id not in processed_docs:
                processed_docs[doc_id] = True
                documents.append(Document(
                    page_content=context,
                    metadata={"source": "narrative_qa", "doc_id": doc_id}
                ))

            evaluation_set.append({
                "question": row['question']['text'],
                # לוקחים את התשובה הראשונה מרשימת התשובות [cite: 459]
                "ground_truth": row['answers'][0]['text'],
                "doc_id": doc_id
            })

        return documents, evaluation_set