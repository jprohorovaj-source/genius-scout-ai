
from typing import List

import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from collector import ResearchPaper
from config import CROSS_ENCODER_MODEL, SEMANTIC_THRESHOLD


class SemanticGate:
    """
    Семантический шлюз второго уровня.

    Выполняет интеллектуальную проверку найденных публикаций
    при помощи Cross-Encoder и пропускает только наиболее
    релевантные документы из области Machine Learning.
    """

    def __init__(
        self,
        relevance_threshold: float = SEMANTIC_THRESHOLD,
        model_name: str = CROSS_ENCODER_MODEL,
    ):

        self.threshold = relevance_threshold

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"SemanticGate: загрузка модели '{model_name}' "
            f"на устройство {self.device.upper()}..."
        )

        self.model = CrossEncoder(
            model_name,
            device=self.device,
        )

        logger.info("SemanticGate успешно инициализирован.")

    def filter_papers(
        self,
        query: str,
        papers: List[ResearchPaper],
    ) -> List[ResearchPaper]:
        """
        Выполняет глубокую повторную проверку релевантности найденных статей
        через совместное кодирование пары 'запрос - документ'.
        """

        if not papers:
            return []

        logger.info(
            f"SemanticGate: проверка {len(papers)} публикаций..."
        )

        # Формируем пары для совместного анализа Cross-Encoder
        pairs = [
            (query, f"{paper.title}. {paper.summary}")
            for paper in papers
        ]

        scores = self.model.predict(pairs)

        passed = []

        for paper, score in zip(papers, scores):

            logger.info(
                f"[Gate] score={score:.3f} | {paper.title[:80]}"
            )

            if score >= self.threshold:
                passed.append(paper)

        logger.info(
            f"SemanticGate пропустил "
            f"{len(passed)} из {len(papers)} документов."
        )

        return passed


print("✅ gate.py успешно создан.")
