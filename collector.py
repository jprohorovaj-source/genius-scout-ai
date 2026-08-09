from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import arxiv
from deep_translator import GoogleTranslator
from loguru import logger


# ============================================================
# Research Paper
# ============================================================

@dataclass
class ResearchPaper:
    """
    Структура данных научной публикации ArXiv.
    """

    id: str
    title: str
    summary: str
    authors: List[str]
    published_date: str
    pdf_url: str


# ============================================================
# Context Disambiguator
# ============================================================

class ContextDisambiguator:
    """
    Подготавливает пользовательский запрос
    для академического поиска ArXiv.

    Использует taxonomy.json для определения
    научного домена и связанных терминов.
    """

    def __init__(
        self,
        taxonomy_path: str = "taxonomy.json",
    ):
        self.taxonomy_path = Path(
            taxonomy_path
        )

        self.taxonomy = (
            self._load_taxonomy()
        )

        logger.info(
            "ContextDisambiguator успешно "
            "инициализирован."
        )

    # ========================================================
    # TAXONOMY
    # ========================================================

    def _load_taxonomy(self) -> Dict:
        """
        Загружает taxonomy.json.
        """

        if not self.taxonomy_path.exists():

            logger.warning(
                f"taxonomy.json не найден: "
                f"{self.taxonomy_path}"
            )

            return {}

        try:

            with open(
                self.taxonomy_path,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            logger.info(
                f"Taxonomy загружена: "
                f"{len(data)} доменов."
            )

            return data

        except Exception as e:

            logger.error(
                f"Ошибка загрузки taxonomy.json: "
                f"{e}"
            )

            return {}

    # ========================================================
    # LANGUAGE DETECTION
    # ========================================================

    @staticmethod
    def _is_russian(
        text: str
    ) -> bool:
        """
        Определяет наличие кириллицы.
        """

        return bool(
            re.search(
                r"[а-яА-ЯёЁ]",
                text
            )
        )

    # ========================================================
    # TRANSLATION
    # ========================================================

    def _translate_to_academic_english(
        self,
        query: str,
    ) -> str:
        """
        Переводит запрос на английский язык.

        Для английского запроса перевод не выполняется.
        """

        query = query.strip()

        if not query:
            return query

        if not self._is_russian(query):

            logger.info(
                f"Запрос уже на английском: "
                f"'{query}'"
            )

            return query

        try:

            translated = GoogleTranslator(
                source="ru",
                target="en",
            ).translate(query)

            translated = (
                translated.strip()
                if translated
                else query
            )

            logger.info(
                f"Перевод запроса: "
                f"'{query}' → '{translated}'"
            )

            return translated

        except Exception as e:

            logger.warning(
                f"Не удалось перевести запрос "
                f"'{query}': {e}"
            )

            return query

    # ========================================================
    # TAXONOMY MATCHING
    # ========================================================

    def _find_taxonomy_topic(
        self,
        query_en: str,
    ) -> Optional[str]:
        """
        Определяет наиболее подходящую тему
        из taxonomy.json.
        """

        query_lower = query_en.lower()

        # ----------------------------------------------------
        # Сначала ищем прямое совпадение названия темы
        # ----------------------------------------------------

        for topic in self.taxonomy:

            if topic.lower() in query_lower:

                return topic

        # ----------------------------------------------------
        # Затем ищем совпадение по keywords
        # ----------------------------------------------------

        for topic, data in self.taxonomy.items():

            keywords = data.get(
                "keywords",
                []
            )

            for keyword in keywords:

                if keyword.lower() in query_lower:

                    return topic

        return None

    # ========================================================
    # QUERY ENRICHMENT
    # ========================================================

    def enrich_query(
        self,
        user_query: str,
    ) -> str:
        """
        Переводит пользовательский запрос
        и формирует академический поисковый запрос.

        ВАЖНО:

        Старый универсальный фильтр

            machine learning OR
            deep learning OR
            neural networks ...

        больше НЕ используется.

        Вместо него используется taxonomy.json.
        """

        query_en = (
            self._translate_to_academic_english(
                user_query
            )
        )

        topic = self._find_taxonomy_topic(
            query_en
        )

        # ----------------------------------------------------
        # Если тема найдена в taxonomy
        # ----------------------------------------------------

        if topic:

            data = self.taxonomy[
                topic
            ]

            exact_query = data.get(
                "exact_query",
                ""
            )

            categories = data.get(
                "categories",
                []
            )

            logger.info(
                f"Taxonomy topic: '{topic}'"
            )

            logger.info(
                f"ArXiv categories: "
                f"{categories}"
            )

            # ------------------------------------------------
            # Используем exact_query,
            # если он определён.
            # ------------------------------------------------

            if exact_query:

                enriched = (
                    f"({exact_query})"
                )

            else:

                enriched = (
                    f'"{query_en}"'
                )

            logger.info(
                f"Сформирован taxonomy-запрос: "
                f"'{enriched}'"
            )

            return enriched

        # ----------------------------------------------------
        # Если темы нет в taxonomy
        # ----------------------------------------------------

        logger.info(
            "Тема отсутствует в taxonomy.json. "
            "Используется прямой академический запрос."
        )

        return (
            f'"{query_en}"'
        )


# ============================================================
# ArXiv Collector
# ============================================================

class ArxivCollector:
    """
    Сборщик научных публикаций ArXiv.

    Выполняет:

        Query
          ↓
        Translation
          ↓
        Taxonomy
          ↓
        ArXiv Search
          ↓
        Metadata filtering
          ↓
        ResearchPaper
    """

    def __init__(
        self,
        max_results: int = 10,
        taxonomy_path: str = "taxonomy.json",
    ):

        self.max_results = max_results

        self.disambiguator = (
            ContextDisambiguator(
                taxonomy_path=taxonomy_path
            )
        )

        logger.info(
            "ArxivCollector успешно "
            "инициализирован."
        )

    # ========================================================
    # CATEGORY DETECTION
    # ========================================================

    def _get_categories(
        self,
        query_en: str,
    ) -> List[str]:
        """
        Возвращает категории ArXiv,
        соответствующие taxonomy.
        """

        topic = (
            self.disambiguator
            ._find_taxonomy_topic(
                query_en
            )
        )

        if not topic:
            return []

        data = (
            self.disambiguator.taxonomy
            .get(topic, {})
        )

        return data.get(
            "categories",
            []
        )

    # ========================================================
    # FORBIDDEN TERMS
    # ========================================================

    @staticmethod
    def _is_forbidden(
        title: str,
        summary: str,
    ) -> bool:
        """
        Базовая защита от явно нерелевантных
        биомедицинских публикаций.

        Это НЕ основной semantic filter.
        Основной фильтр находится в gate.py.
        """

        forbidden_terms = [
            "gene",
            "protein",
            "patient",
            "clinical",
            "cell biology",
            "disease",
            "tumor",
            "rna",
            "dna",
        ]

        text = (
            f"{title} {summary}"
            .lower()
        )

        return any(
            term in text
            for term in forbidden_terms
        )

    # ========================================================
    # FETCH PAPERS
    # ========================================================

    async def fetch_papers(
        self,
        query: str,
    ) -> List[ResearchPaper]:
        """
        Получает научные публикации с ArXiv.
        """

        logger.info(
            f"ArxivCollector: "
            f"исследование темы '{query}'"
        )

        # ----------------------------------------------------
        # Перевод
        # ----------------------------------------------------

        query_en = (
            self.disambiguator
            ._translate_to_academic_english(
                query
            )
        )

        # ----------------------------------------------------
        # Taxonomy
        # ----------------------------------------------------

        refined_query = (
            self.disambiguator
            .enrich_query(query)
        )

        categories = (
            self._get_categories(
                query_en
            )
        )

        logger.info(
            f"Academic query: '{query_en}'"
        )

        logger.info(
            f"ArXiv query: '{refined_query}'"
        )

        if categories:

            logger.info(
                f"Taxonomy categories: "
                f"{categories}"
            )

        # ----------------------------------------------------
        # ArXiv search
        # ----------------------------------------------------

        search = arxiv.Search(
            query=refined_query,
            max_results=self.max_results * 3,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        client = arxiv.Client()

        papers: List[
            ResearchPaper
        ] = []

        try:

            for result in client.results(
                search
            ):

                # --------------------------------------------
                # Metadata
                # --------------------------------------------

                title = (
                    result.title
                    .strip()
                )

                summary = (
                    result.summary
                    .strip()
                )

                # --------------------------------------------
                # Базовая фильтрация
                # --------------------------------------------

                if self._is_forbidden(
                    title,
                    summary,
                ):

                    logger.info(
                        f"[Collector] "
                        f"REJECT biomedical: "
                        f"{title}"
                    )

                    continue

                # --------------------------------------------
                # ResearchPaper
                # --------------------------------------------

                paper = ResearchPaper(
                    id=(
                        result.entry_id
                        .split("/")[-1]
                    ),
                    title=title,
                    summary=summary,
                    authors=[
                        author.name
                        for author
                        in result.authors
                    ],
                    published_date=(
                        result.published
                        .strftime(
                            "%Y-%m-%d"
                        )
                    ),
                    pdf_url=result.pdf_url,
                )

                papers.append(
                    paper
                )

                logger.info(
                    f"[Collector] "
                    f"ACCEPT: {title}"
                )

                if len(papers) >= (
                    self.max_results
                ):

                    break

            logger.info(
                f"ArXiv: собрано "
                f"{len(papers)} публикаций."
            )

            return papers

        except Exception as e:

            logger.exception(
                "Ошибка при обращении "
                "к ArXiv API."
            )

            return []


print(
    "✅ collector.py успешно обновлен."
)
