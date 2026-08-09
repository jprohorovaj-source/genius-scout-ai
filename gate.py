%%writefile gate.py

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from collector import ResearchPaper, ContextDisambiguator
from config import CROSS_ENCODER_MODEL


class SemanticGate:
    """
    Семантический шлюз GeniusScout AI.

    Выполняет каскадную проверку релевантности:

    1. Переводит пользовательский запрос
       в академический английский.

    2. Вычисляет semantic score через Cross-Encoder.

    3. Проверяет наличие тематических признаков
       из taxonomy.json.

    4. Использует строгий порог релевантности.

    5. Отбрасывает документы с явно слабой
       семантической связью.

    ВАЖНО:
    Cross-Encoder ms-marco-MiniLM-L-6-v2 возвращает
    logits, а НЕ вероятность от 0 до 1.
    Поэтому threshold = -20.0 является слишком мягким.
    """

    # ---------------------------------------------------------
    # Основные параметры фильтрации
    # ---------------------------------------------------------

    # Основной минимальный semantic score.
    #
    # Вместо старого -20.0 используем значительно
    # более строгий порог.
    MIN_SCORE = 0.0

    # Для документов без тематических ключевых слов
    # требуется более высокий score.
    STRICT_SCORE = 2.0

    # Минимальная разница между документом и слабым
    # кандидатом при наличии нескольких результатов.
    MIN_MARGIN = 0.5

    # Максимальная доля нерелевантных документов,
    # которые допускаются только при отсутствии
    # тематических признаков.
    ALLOW_WEAK_WITHOUT_KEYWORDS = False

    def __init__(
        self,
        relevance_threshold: float = MIN_SCORE,
        model_name: str = CROSS_ENCODER_MODEL,
        taxonomy_path: str = "taxonomy.json",
    ):
        self.threshold = relevance_threshold
        self.strict_threshold = self.STRICT_SCORE

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"SemanticGate: загрузка модели "
            f"'{model_name}' "
            f"на устройство {self.device.upper()}..."
        )

        self.model = CrossEncoder(
            model_name,
            device=self.device
        )

        self.disambiguator = ContextDisambiguator(
            taxonomy_path=taxonomy_path
        )

        self.taxonomy = self._load_taxonomy(
            taxonomy_path
        )

        logger.info(
            "SemanticGate успешно инициализирован."
        )

        logger.info(
            f"Semantic threshold: {self.threshold}"
        )

        logger.info(
            f"Strict threshold: {self.strict_threshold}"
        )

    # =========================================================
    # TAXONOMY
    # =========================================================

    @staticmethod
    def _load_taxonomy(
        taxonomy_path: str
    ) -> Dict:
        """
        Загружает taxonomy.json.
        """

        path = Path(taxonomy_path)

        if not path.exists():
            logger.warning(
                f"taxonomy.json не найден: {path}"
            )
            return {}

        try:
            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            if not isinstance(data, dict):
                logger.warning(
                    "taxonomy.json имеет "
                    "некорректную структуру."
                )
                return {}

            logger.info(
                f"Taxonomy загружена: "
                f"{len(data)} категорий."
            )

            return data

        except Exception as e:

            logger.error(
                f"Ошибка загрузки taxonomy.json: {e}"
            )

            return {}

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_text(
        text: str
    ) -> str:
        """
        Нормализация текста для проверки
        тематических ключевых слов.
        """

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9\s-]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # =========================================================
    # QUERY PREPARATION
    # =========================================================

    def _prepare_query(
        self,
        query: str
    ) -> Tuple[str, Optional[str], List[str]]:
        """
        Подготавливает запрос:

        Returns:
            query_en
            matched_topic
            taxonomy_keywords
        """

        query_en = (
            self.disambiguator
            ._translate_to_academic_english(query)
        )

        query_en = query_en.strip()

        matched_topic = None
        keywords: List[str] = []

        query_lower = query_en.lower()

        # -----------------------------------------------------
        # Ищем соответствующую тему в taxonomy.json
        # -----------------------------------------------------

        for topic, data in self.taxonomy.items():

            topic_lower = topic.lower()

            if topic_lower in query_lower:
                matched_topic = topic
                break

            for keyword in data.get(
                "keywords",
                []
            ):

                if keyword.lower() in query_lower:
                    matched_topic = topic
                    break

            if matched_topic:
                break

        # -----------------------------------------------------
        # Если тема найдена — собираем keywords
        # -----------------------------------------------------

        if matched_topic:

            data = self.taxonomy[
                matched_topic
            ]

            keywords = [
                str(keyword).lower()
                for keyword in data.get(
                    "keywords",
                    []
                )
            ]

        # -----------------------------------------------------
        # Специальная обработка Decision Trees
        # -----------------------------------------------------

        if (
            "decision tree" in query_lower
            or "decision trees" in query_lower
        ):

            matched_topic = (
                matched_topic
                or "decision_tree"
            )

            keywords = list(
                set(
                    keywords
                    + [
                        "decision tree",
                        "decision trees",
                        "decision tree learning",
                        "decision tree classifier",
                        "classification tree",
                        "tree-based model",
                    ]
                )
            )

        logger.info(
            f"SemanticGate query: "
            f"'{query}' -> '{query_en}'"
        )

        if matched_topic:
            logger.info(
                f"SemanticGate topic: "
                f"'{matched_topic}'"
            )

        return (
            query_en,
            matched_topic,
            keywords
        )

    # =========================================================
    # KEYWORD ANALYSIS
    # =========================================================

    def _keyword_matches(
        self,
        text: str,
        keywords: List[str]
    ) -> List[str]:
        """
        Возвращает найденные тематические ключевые слова.
        """

        normalized_text = (
            self._normalize_text(text)
        )

        matches = []

        for keyword in keywords:

            normalized_keyword = (
                self._normalize_text(keyword)
            )

            if not normalized_keyword:
                continue

            if normalized_keyword in normalized_text:
                matches.append(keyword)

        return list(set(matches))

    # =========================================================
    # DOCUMENT SCORING
    # =========================================================

    def _score_documents(
        self,
        query_en: str,
        papers: List[ResearchPaper]
    ) -> List[float]:
        """
        Вычисляет Cross-Encoder score
        для каждой публикации.
        """

        pairs = []

        for paper in papers:

            document = (
                f"Title: {paper.title}. "
                f"Abstract: {paper.summary}"
            )

            pairs.append(
                (
                    query_en,
                    document
                )
            )

        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )

        return [
            float(score)
            for score in scores
        ]

    # =========================================================
    # MAIN FILTER
    # =========================================================

    def filter_papers(
        self,
        query: str,
        papers: List[ResearchPaper],
    ) -> List[ResearchPaper]:
        """
        Выполняет каскадную семантическую фильтрацию.
        """

        if not papers:
            logger.warning(
                "SemanticGate получил пустой список."
            )
            return []

        logger.info(
            f"SemanticGate: проверка "
            f"{len(papers)} публикаций..."
        )

        # -----------------------------------------------------
        # 1. Перевод запроса
        # -----------------------------------------------------

        (
            query_en,
            matched_topic,
            keywords
        ) = self._prepare_query(query)

        # -----------------------------------------------------
        # 2. Cross-Encoder scoring
        # -----------------------------------------------------

        scores = self._score_documents(
            query_en,
            papers
        )

        # -----------------------------------------------------
        # 3. Определяем лучший score
        # -----------------------------------------------------

        max_score = max(scores)

        logger.info(
            f"SemanticGate: максимальный score = "
            f"{max_score:.3f}"
        )

        passed = []

        # -----------------------------------------------------
        # 4. Анализируем каждый документ
        # -----------------------------------------------------

        for paper, score in zip(
            papers,
            scores
        ):

            document_text = (
                f"{paper.title} "
                f"{paper.summary}"
            )

            matched_keywords = (
                self._keyword_matches(
                    document_text,
                    keywords
                )
            )

            keyword_count = len(
                matched_keywords
            )

            # -------------------------------------------------
            # Проверка semantic score
            # -------------------------------------------------

            semantic_pass = (
                score >= self.threshold
            )

            # -------------------------------------------------
            # Проверка тематических keywords
            # -------------------------------------------------

            keyword_pass = (
                keyword_count > 0
            )

            # -------------------------------------------------
            # Strict mode
            #
            # Если статья не содержит ни одного
            # тематического ключевого слова,
            # требуем более высокий semantic score.
            # -------------------------------------------------

            if not keyword_pass:

                strict_pass = (
                    score >= self.strict_threshold
                )

            else:

                strict_pass = True

            # -------------------------------------------------
            # Margin относительно лучшего документа
            # -------------------------------------------------

            margin = (
                max_score - score
            )

            # Для лучшего документа margin = 0.
            # Поэтому он не блокируется.
            margin_pass = (
                score == max_score
                or margin <= self.MIN_MARGIN
                or keyword_pass
            )

            # -------------------------------------------------
            # Финальное решение
            # -------------------------------------------------

            accepted = (
                semantic_pass
                and strict_pass
                and margin_pass
            )

            # -------------------------------------------------
            # Логирование
            # -------------------------------------------------

            logger.info(
                "\n"
                f"[Gate] score={score:.3f} | "
                f"keywords={keyword_count} | "
                f"margin={margin:.3f}\n"
                f"       title={paper.title[:100]}\n"
                f"       matched={matched_keywords}\n"
                f"       decision="
                f"{'PASS' if accepted else 'REJECT'}"
            )

            if accepted:

                passed.append(paper)

            else:

                logger.info(
                    f"[Gate] REJECTED: "
                    f"{paper.title}"
                )

        # -----------------------------------------------------
        # 5. Финальная сортировка
        # -----------------------------------------------------

        # Повторно считаем score для сохранения
        # порядка документов по релевантности.
        paper_scores = {
            paper.id: score
            for paper, score in zip(
                papers,
                scores
            )
        }

        passed.sort(
            key=lambda paper: paper_scores.get(
                paper.id,
                -999
            ),
            reverse=True
        )

        # -----------------------------------------------------
        # 6. Итоговая статистика
        # -----------------------------------------------------

        logger.info(
            "=================================================="
        )

        logger.info(
            f"SemanticGate RESULT: "
            f"{len(passed)} / {len(papers)}"
        )

        logger.info(
            f"Query EN: {query_en}"
        )

        logger.info(
            f"Topic: {matched_topic}"
        )

        logger.info(
            f"Keywords: {keywords}"
        )

        logger.info(
            "=================================================="
        )

        return passed


print(
    "✅ gate.py успешно обновлен: "
    "каскадная семантическая фильтрация."
)
