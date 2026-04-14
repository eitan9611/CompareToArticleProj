from datasets import load_dataset
from langchain_core.documents import Document


class CovidQALoader:
    def __init__(self):
        print("Loading full CovidQA dataset (124 samples)...")
        # המאמר השתמש בכל ה-Dataset 
        self.dataset = load_dataset("covid_qa_deepset", split="train")

    def get_data(self):
        documents = []
        evaluation_set = []
        processed_contexts = {}

        for i, row in enumerate(self.dataset):
            context = row['context']
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

        return documents, evaluation_set