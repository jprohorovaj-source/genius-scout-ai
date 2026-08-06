
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from loguru import logger
import torch


class VectorStoreManager:
    """
    Локальная векторная база знаний на основе FAISS.
    """

    def __init__(self,
                 model_name="intfloat/multilingual-e5-large"):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Инициализация VectorStore ({device})")

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

        self.dimension = self.model.get_embedding_dimension()

        self.index = faiss.IndexFlatIP(self.dimension)

        self.documents = []

    def add_papers(self, papers):

        if not papers:
            return

        texts = [
            f"passage: {paper.title}. {paper.summary}"
            for paper in papers
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        self.index.add(
            np.asarray(embeddings, dtype=np.float32)
        )

        self.documents.extend(papers)

        logger.info(
            f"Добавлено документов: {len(papers)} | Всего: {len(self.documents)}"
        )

    def search(self,
               query,
               top_k=5):

        if self.index.ntotal == 0:
            return []

        query_vector = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True
        )

        scores, ids = self.index.search(
            np.asarray(query_vector, dtype=np.float32),
            top_k
        )

        results = []

        for score, idx in zip(scores[0], ids[0]):

            if idx == -1:
                continue

            results.append({
                "score": float(score),
                "paper": self.documents[idx]
            })

        logger.info(f"Найдено документов: {len(results)}")

        return results


print("✅ vector_store.py успешно создан.")
