from __future__ import annotations

import json
import re
import time
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
# Rate Limiter
# ============================================================

class RateLimiter:
    """
    Защита от частотных ограничений (Rate Limit) ArXiv API.
    """

    def __init__(self, min_interval: float = 3.0):
        self.min_interval = min_interval
        self.last_call_time = 0.0

    def check_and_wait(self):
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            logger.warning(f"Rate limit: пауза перед запросом к ArXiv {wait_time:.1f} сек.")
            time.sleep(wait_time)
        self.last_call_time = time.time()


# ============================================================
# Context Disambiguator
# ============================================================

class ContextDisambiguator:
    """
    Подготавливает пользовательский запрос для академического поиска ArXiv,
    используя taxonomy.json для определения домена и категорий.
    """

    def __init__(
        self,
        taxonomy_path: str = "taxonomy.json",
    ):
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy = self._load_taxonomy()
        logger.info("ContextDisambiguator успешно инициализирован.")

    def _load_taxonomy(self) -> Dict:
        if not self.taxonomy_path.exists():
            logger.warning(f"taxonomy.json не найден: {self.taxonomy_path}")
            return {}

        try:
            with open(self.taxonomy_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            logger.info(f"Taxonomy загружена: {len(data)} доменов.")
            return data
        except Exception as e:
            logger.error(f"Ошибка загрузки taxonomy.json: {e}")
            return {}

    @staticmethod
    def _is_russian(text: str) -> bool:
        return bool(re.search(r"[а-яА-ЯёЁ]", text))

    def _translate_to_academic_english(self, query: str) -> str:
        query = query.strip()
        if not query:
            return query

        if not self._is_russian(query):
            logger.info(f"Запрос уже на английском: '{query}'")
            return query

        try:
            translated = GoogleTranslator(source="ru", target="en").translate(query)
            translated = translated.strip() if translated else query
            logger.info(f"Перевод запроса: '{query}' → '{translated}'")
            return translated
        except Exception as e:
            logger.warning(f"Не удалось перевести запрос '{query}': {e}")
            return query

    def _find_taxonomy_topic(self, query_en: str) -> Optional[str]:
        query_lower = query_en.lower()

        for topic in self.taxonomy:
            if topic.lower() in query_lower:
                return topic

        for topic, data in self.taxonomy.items():
            keywords = data.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    return topic

        return None

    def enrich_query(self, user_query: str) -> str:
        query_en = self._translate_to_academic_english(user_query)
        topic = self._find_taxonomy_topic(query_en)

        if topic:
            data = self.taxonomy[topic]
            exact_query = data.get("exact_query", "")
            categories = data.get("categories", [])

            logger.info(f"Taxonomy topic: '{topic}'")
            logger.info(f"ArXiv categories: {categories}")

            if exact_query:
                enriched = f"({exact_query})"
            else:
                enriched = f'"{query_en}"'

            logger.info(f"Сформирован taxonomy-запрос: '{enriched}'")
            return enriched

        logger.info("Тема отсутствует в taxonomy.json. Используется прямой академический запрос.")
        return f'"{query_en}"'


# ============================================================
# ArXiv Collector
# ============================================================

class ArxivCollector:
    """
    Сборщик научных публикаций ArXiv с фильтрацией по категориям и защитой лимитов.
    """

    def __init__(
        self,
        max_results: int = 10,
        taxonomy_path: str = "taxonomy.json",
    ):
        self.max_results = max_results
        self.disambiguator = ContextDisambiguator(taxonomy_path=taxonomy_path)
        self.rate_limiter = RateLimiter(min_interval=3.0)
        logger.info("ArxivCollector успешно инициализирован.")

    def _get_categories(self, query_en: str) -> List[str]:
        topic = self.disambiguator._find_taxonomy_topic(query_en)
        if not topic:
            return []
        data = self.disambiguator.taxonomy.get(topic, {})
        return data.get("categories", [])

    @staticmethod
    def _is_forbidden(title: str, summary: str) -> bool:
        forbidden_terms = [
            "gene", "protein", "patient", "clinical",
            "cell biology", "disease", "tumor", "rna", "dna",
        ]
        text = f"{title} {summary}".lower()
        return any(term in text for term in forbidden_terms)

    async def fetch_papers(self, query: str) -> List[ResearchPaper]:
        logger.info(f"ArxivCollector: исследование темы '{query}'")

        query_en = self.disambiguator._translate_to_academic_english(query)
        refined_query = self.disambiguator.enrich_query(query)
        categories = self._get_categories(query_en)

        logger.info(f"Academic query: '{query_en}'")

        # Интегрируем строгие категории из таксономии в поисковый запрос ArXiv
        final_query = refined_query
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            final_query = f"({refined_query}) AND ({cat_query})"

        logger.info(f"Финальный запрос в ArXiv API: '{final_query}'")

        # Контролируем частоту запросов к API
        self.rate_limiter.check_and_wait()

        search = arxiv.Search(
            query=final_query,
            max_results=self.max_results * 3,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        client = arxiv.Client()
        papers: List[ResearchPaper] = []

        try:
            for result in client.results(search):
                title = result.title.strip()
                summary = result.summary.strip()

                if self._is_forbidden(title, summary):
                    logger.info(f"[Collector] REJECT biomedical: {title}")
                    continue

                paper = ResearchPaper(
                    id=result.entry_id.split("/")[-1],
                    title=title,
                    summary=summary,
                    authors=[author.name for author in result.authors],
                    published_date=result.published.strftime("%Y-%m-%d"),
                    pdf_url=result.pdf_url,
                )

                papers.append(paper)
                logger.info(f"[Collector] ACCEPT: {title}")

                if len(papers) >= self.max_results:
                    break

            logger.info(f"ArXiv: собрано {len(papers)} публикаций.")
            return papers

        except Exception as e:
            logger.exception("Ошибка при обращении к ArXiv API.")
            return []
