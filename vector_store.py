%%writefile vector_store.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
import torch

from loguru import logger
from sentence_transformers import SentenceTransformer

from collector import ResearchPaper
from config import EMBEDDING_MODEL, TOP_K


class VectorStoreManager:
    """
    Локальное семантическое хранилище GeniusScout AI.

    Архитектура:

        ResearchPaper
             ↓
        E5 Embedding
             ↓
        L2 Normalization
             ↓
        FAISS IndexFlatIP
             ↓
        Semantic Search

    Используется multilingual-e5-large, поэтому
    база поддерживает как английские, так и русские запросы.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
    ):
        # ====================================================
        # Device
        # ====================================================

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"VectorStore: устройство "
            f"{self.device.upper()}"
        )

        # ====================================================
        # Embedding model
        # ====================================================

        logger.info(
            f"Загрузка embedding-модели: "
            f"{model_name}"
        )

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

        self.dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        logger.info(
            f"Размерность embedding: "
            f"{self.dimension}"
        )

        # ====================================================
        # FAISS
        # ====================================================

        # После L2-нормализации Inner Product
        # эквивалентен cosine similarity.
        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        # ====================================================
        # Документы
        # ====================================================

        self.documents: List[ResearchPaper] = []

        logger.info(
            "FAISS VectorStore успешно создан."
        )

    # ========================================================
    # TEXT BUILDING
    # ========================================================

    @staticmethod
    def _build_passage(
        paper: ResearchPaper
    ) -> str:
        """
        Формирует passage для E5.

        E5 рекомендует использовать префикс:
        passage:
        """

        return (
            "passage: "
            f"Title: {paper.title}\n"
            f"Abstract: {paper.summary}"
        )

    @staticmethod
    def _build_query(
        query: str
    ) -> str:
        """
        Формирует поисковый запрос для E5.
        """

        return f"query: {query}"

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def _encode_documents(
        self,
        papers: List[ResearchPaper],
    ) -> np.ndarray:
        """
        Создаёт embedding-вектора документов.
        """

        texts = [
            self._build_passage(paper)
            for paper in papers
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def _encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Создаёт embedding поискового запроса.
        """

        text = self._build_query(query)

        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_papers(
        self,
        papers: List[ResearchPaper],
    ) -> None:
        """
        Добавляет научные публикации
        в локальный FAISS индекс.
        """

        if not papers:

            logger.warning(
                "VectorStore: получен пустой "
                "список документов."
            )

            return

        logger.info(
            f"VectorStore: добавление "
            f"{len(papers)} документов..."
        )

        embeddings = self._encode_documents(
            papers
        )

        self.index.add(
            embeddings
        )

        self.documents.extend(
            papers
        )

        logger.info(
            f"VectorStore: добавлено "
            f"{len(papers)} документов."
        )

        logger.info(
            f"VectorStore: всего документов "
            f"в индексе = {self.index.ntotal}"
        )

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Выполняет семантический поиск
        по локальной базе знаний.

        Возвращает:

        [
            {
                "paper": ResearchPaper,
                "score": float
            }
        ]
        """

        if self.index.ntotal == 0:

            logger.warning(
                "VectorStore: FAISS индекс пуст."
            )

            return []

        # ----------------------------------------------------
        # Ограничиваем top_k размером индекса
        # ----------------------------------------------------

        top_k = min(
            top_k,
            self.index.ntotal,
        )

        logger.info(
            f"VectorStore: поиск "
            f"'{query}' | top_k={top_k}"
        )

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_embedding = self._encode_query(
            query
        )

        # ----------------------------------------------------
        # FAISS search
        # ----------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):

            if index_id < 0:
                continue

            paper = self.documents[
                int(index_id)
            ]

            results.append(
                {
                    "paper": paper,
                    "score": float(score),
                }
            )

        logger.info(
            f"VectorStore: найдено "
            f"{len(results)} документов."
        )

        return results

    # ========================================================
    # SEARCH WITH MINIMUM SCORE
    # ========================================================

    def search_relevant(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """
        Семантический поиск с дополнительным
        порогом cosine similarity.

        Это второй уровень защиты после FAISS retrieval.

        Например:

            min_score=0.50

        означает, что документы с cosine similarity
        ниже 0.50 будут отброшены.
        """

        results = self.search(
            query=query,
            top_k=top_k,
        )

        filtered = [
            result
            for result in results
            if result["score"] >= min_score
        ]

        logger.info(
            f"VectorStore: после score-фильтра "
            f"{len(filtered)} / {len(results)}"
        )

        return filtered

    # ========================================================
    # SAVE INDEX
    # ========================================================

    def save(
        self,
        index_path: str = "data/faiss_index.bin",
    ) -> None:
        """
        Сохраняет FAISS индекс на диск.
        """

        path = Path(index_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(path),
        )

        logger.info(
            f"FAISS индекс сохранён: "
            f"{path}"
        )

    # ========================================================
    # LOAD INDEX
    # ========================================================

    def load(
        self,
        index_path: str = "data/faiss_index.bin",
    ) -> bool:
        """
        Загружает FAISS индекс с диска.

        Возвращает True, если загрузка успешна.
        """

        path = Path(index_path)

        if not path.exists():

            logger.warning(
                f"FAISS индекс не найден: "
                f"{path}"
            )

            return False

        try:

            self.index = faiss.read_index(
                str(path)
            )

            logger.info(
                f"FAISS индекс загружен: "
                f"{path}"
            )

            logger.info(
                f"Документов в индексе: "
                f"{self.index.ntotal}"
            )

            return True

        except Exception as e:

            logger.error(
                f"Ошибка загрузки FAISS: {e}"
            )

            return False

    # ========================================================
    # STATISTICS
    # ========================================================

    def stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику VectorStore.
        """

        return {
            "documents": len(
                self.documents
            ),
            "vectors": self.index.ntotal,
            "dimension": self.dimension,
            "model": EMBEDDING_MODEL,
            "device": self.device,
            "metric": "cosine_similarity",
        }

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:
        """
        Полностью очищает локальный индекс.
        """

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.documents = []

        logger.info(
            "VectorStore полностью очищен."
        )


print(
    "✅ vector_store.py успешно создан."
)
