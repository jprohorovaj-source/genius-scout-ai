from __future__ import annotations

from typing import List

import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from collector import ResearchPaper
from config import CROSS_ENCODER_MODEL, SEMANTIC_THRESHOLD


class SemanticGate:
    """
    Семантический шлюз GeniusScout AI.

    Второй уровень фильтрации после ArXiv Collector.

    Задача:
        query + scientific paper
                ↓
          Cross-Encoder
                ↓
        relevance score
                ↓
        threshold filter
                ↓
        relevant papers
    """

    def __init__(
        self,
        relevance_threshold: float = SEMANTIC_THRESHOLD,
        model_name: str = CROSS_ENCODER_MODEL,
    ):

        self.threshold = relevance_threshold

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"SemanticGate: загрузка модели "
            f"'{model_name}' "
            f"на {self.device.upper()}..."
        )

        self.model = CrossEncoder(
            model_name,
            device=self.device,
        )

        logger.info(
            f"SemanticGate успешно инициализирован. "
            f"Threshold={self.threshold}"
        )

    def filter_papers(
        self,
        query: str,
        papers: List[ResearchPaper],
    ) -> List[ResearchPaper]:
        """
        Проверяет релевантность научных публикаций.

        Cross-Encoder получает пару:

            (пользовательский запрос,
             title + abstract)

        и рассчитывает semantic relevance score.
        """

        if not papers:
            logger.warning(
                "SemanticGate: получен пустой список."
            )
            return []

        logger.info(
            f"SemanticGate: проверка "
            f"{len(papers)} публикаций."
        )

        pairs = [
            (
                query,
                f"{paper.title}. {paper.summary}",
            )
            for paper in papers
        ]

        try:

            scores = self.model.predict(
                pairs
            )

        except Exception as e:

            logger.exception(
                f"Ошибка Cross-Encoder: {e}"
            )

            return []

        passed: List[ResearchPaper] = []

        for paper, score in zip(
            papers,
            scores,
        ):

            score = float(score)

            logger.info(
                f"[Gate] "
                f"score={score:.4f} | "
                f"{paper.title[:100]}"
            )

            if score >= self.threshold:

                passed.append(
                    paper
                )

                logger.info(
                    f"[Gate] PASS | "
                    f"{paper.title[:80]}"
                )

            else:

                logger.info(
                    f"[Gate] REJECT | "
                    f"{paper.title[:80]}"
                )

        logger.info(
            f"SemanticGate: пропущено "
            f"{len(passed)} из {len(papers)}."
        )

        return passed


print(
    "✅ gate.py успешно загружен."
)
