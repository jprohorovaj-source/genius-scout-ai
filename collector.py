%%writefile collector.py

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arxiv
from deep_translator import GoogleTranslator
from loguru import logger


@dataclass
class ResearchPaper:
    """Структура данных для научной публикации с ArXiv."""

    id: str
    title: str
    summary: str
    authors: List[str]
    published_date: str
    pdf_url: str


class ContextDisambiguator:
    """
    Подготавливает пользовательский запрос для академического поиска
    в ArXiv.

    Основные задачи:
    1. Определить язык пользовательского запроса.
    2. Перевести русский запрос на академический английский.
    3. Найти соответствующую область в taxonomy.json.
    4. Использовать специализированный ArXiv-запрос из taxonomy.json.
    5. Для неизвестных тем сформировать безопасный общий запрос.
    """

    # Частые русско-английские академические соответствия.
    # Они нужны, чтобы короткие ML-запросы переводились
    # именно в научную терминологию.
    ACADEMIC_ALIASES = {
        "деревья решений": "decision trees",
        "дерево решений": "decision tree",
        "случайный лес": "random forest",
        "случайные леса": "random forests",
        "градиентный бустинг": "gradient boosting",
        "нейронные сети": "neural networks",
        "нейронная сеть": "neural network",
        "глубокое обучение": "deep learning",
        "машинное обучение": "machine learning",
        "обучение с подкреплением": "reinforcement learning",
        "компьютерное зрение": "computer vision",
        "обработка естественного языка": "natural language processing",
        "большие языковые модели": "large language models",
        "языковые модели": "language models",
        "трансформеры": "transformers",
        "трансформер": "transformer",
        "мультимодальные модели": "multimodal models",
        "векторные представления": "embeddings",
        "эмбеддинги": "embeddings",
        "кластеризация": "clustering",
        "классификация": "classification",
        "регрессия": "regression",
    }

    # Для известных академических тем, которых пока нет
    # отдельной записью в taxonomy.json.
    TAXONOMY_FALLBACKS = {
        "decision_tree": {
            "categories": ["cs.LG", "stat.ML"],
            "exact_query": (
                'all:"decision tree" OR '
                'all:"decision trees" OR '
                'all:"decision tree learning" OR '
                'all:"decision tree classifier"'
            ),
        }
    }

    def __init__(self, taxonomy_path: str = "taxonomy.json"):
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy = self._load_taxonomy()

        logger.info(
            f"ContextDisambiguator инициализирован. "
            f"Загружено таксономий: {len(self.taxonomy)}"
        )

    def _load_taxonomy(self) -> Dict:
        """Загружает taxonomy.json."""

        if not self.taxonomy_path.exists():
            logger.warning(
                f"Файл taxonomy.json не найден: {self.taxonomy_path}. "
                "Будет использован fallback-поиск."
            )
            return {}

        try:
            with open(
                self.taxonomy_path,
                "r",
                encoding="utf-8"
            ) as file:
                taxonomy = json.load(file)

            if not isinstance(taxonomy, dict):
                logger.warning(
                    "taxonomy.json имеет некорректную структуру."
                )
                return {}

            return taxonomy

        except Exception as e:
            logger.error(
                f"Ошибка загрузки taxonomy.json: {e}"
            )
            return {}

    @staticmethod
    def _contains_cyrillic(text: str) -> bool:
        """Проверяет наличие кириллицы."""

        return bool(
            re.search(r"[а-яА-ЯёЁ]", text)
        )

    def _translate_to_academic_english(
        self,
        user_query: str
    ) -> str:
        """
        Переводит пользовательский запрос на английский.

        Сначала проверяются известные академические соответствия.
        Если соответствие не найдено — используется GoogleTranslator.
        """

        query_clean = user_query.strip()
        query_lower = query_clean.lower()

        # 1. Прямое академическое соответствие.
        if query_lower in self.ACADEMIC_ALIASES:
            translated = self.ACADEMIC_ALIASES[query_lower]

            logger.info(
                f"Академический словарь: "
                f"'{query_clean}' -> '{translated}'"
            )

            return translated

        # 2. Если запрос уже на английском,
        # перевод не требуется.
        if not self._contains_cyrillic(query_clean):
            logger.info(
                f"Запрос уже на английском: '{query_clean}'"
            )
            return query_clean

        # 3. Автоматический перевод.
        try:
            translated = GoogleTranslator(
                source="ru",
                target="en"
            ).translate(query_clean)

            translated = translated.strip()

            logger.info(
                f"Перевод запроса: "
                f"'{query_clean}' -> '{translated}'"
            )

            return translated

        except Exception as e:
            logger.error(
                f"Ошибка перевода запроса: {e}"
            )

            # Даже при ошибке возвращаем исходный запрос,
            # чтобы pipeline не падал.
            return query_clean

    def _find_taxonomy_entry(
        self,
        query_en: str
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Ищет подходящую запись в taxonomy.json.

        Поиск выполняется по:
        - названию темы;
        - domain;
        - keywords.
        """

        query_lower = query_en.lower()

        # Сначала проверяем название самой категории.
        for topic, data in self.taxonomy.items():

            topic_lower = topic.lower()

            if topic_lower in query_lower:
                return topic, data

        # Затем ищем совпадения по keywords.
        for topic, data in self.taxonomy.items():

            keywords = data.get("keywords", [])

            for keyword in keywords:

                keyword_lower = keyword.lower()

                if keyword_lower in query_lower:
                    return topic, data

        return None, None

    def _build_category_filter(
        self,
        categories: List[str]
    ) -> str:
        """
        Формирует фильтр категорий ArXiv.

        Например:

        (cat:cs.LG OR cat:stat.ML)
        """

        if not categories:
            return ""

        category_query = " OR ".join(
            f"cat:{category}"
            for category in categories
        )

        return f"({category_query})"

    def _build_generic_query(
        self,
        query_en: str
    ) -> str:
        """
        Формирует запрос для темы,
        которой пока нет в taxonomy.json.
        """

        # Специальный случай Decision Trees.
        if (
            "decision tree" in query_en.lower()
            or "decision trees" in query_en.lower()
        ):
            fallback = self.TAXONOMY_FALLBACKS["decision_tree"]

            category_filter = self._build_category_filter(
                fallback["categories"]
            )

            exact_query = fallback["exact_query"]

            return f"({exact_query}) AND {category_filter}"

        # Общий поиск по названию/аннотации.
        #
        # В отличие от старого кода здесь НЕТ:
        #
        # machine learning OR deep learning OR ...
        #
        # Запрос ограничивается только самой темой.
        escaped_query = query_en.replace('"', "")

        category_filter = self._build_category_filter(
            [
                "cs.LG",
                "cs.AI",
                "stat.ML"
            ]
        )

        return (
            f'(all:"{escaped_query}") '
            f"AND {category_filter}"
        )

    def enrich_query(
        self,
        user_query: str
    ) -> str:
        """
        Преобразует пользовательский запрос
        в академический запрос ArXiv.
        """

        query_en = self._translate_to_academic_english(
            user_query
        )

        # Специальная обработка Decision Trees.
        if query_en.lower() in {
            "decision tree",
            "decision trees"
        }:
            refined_query = self._build_generic_query(
                query_en
            )

            logger.info(
                f"Распознана тема Decision Trees."
            )

            logger.info(
                f"Итоговый запрос ArXiv: "
                f"'{refined_query}'"
            )

            return refined_query

        # Пытаемся найти тему в taxonomy.json.
        topic, taxonomy_entry = self._find_taxonomy_entry(
            query_en
        )

        if taxonomy_entry:

            categories = taxonomy_entry.get(
                "categories",
                []
            )

            exact_query = taxonomy_entry.get(
                "exact_query",
                ""
            )

            if exact_query:

                category_filter = (
                    self._build_category_filter(
                        categories
                    )
                )

                refined_query = (
                    f"({exact_query}) "
                    f"AND {category_filter}"
                )

                logger.info(
                    f"Найдена taxonomy-категория: "
                    f"'{topic}'"
                )

                logger.info(
                    f"Итоговый запрос ArXiv: "
                    f"'{refined_query}'"
                )

                return refined_query

        # Если специальной категории нет,
        # формируем общий академический запрос.
        refined_query = self._build_generic_query(
            query_en
        )

        logger.info(
            "Специализированная категория "
            "в taxonomy.json не найдена."
        )

        logger.info(
            f"Итоговый запрос ArXiv: "
            f"'{refined_query}'"
        )

        return refined_query


class ArxivCollector:
    """
    Получает научные публикации из ArXiv.
    """

    def __init__(
        self,
        max_results: int = 10
    ):
        self.max_results = max_results

        self.disambiguator = ContextDisambiguator()

        logger.info(
            "ArxivCollector успешно инициализирован."
        )

    async def fetch_papers(
        self,
        query: str
    ) -> List[ResearchPaper]:
        """
        Получает публикации ArXiv по академически
        обогащённому запросу.
        """

        refined_query = (
            self.disambiguator.enrich_query(query)
        )

        # Технический защитный фильтр.
        #
        # Это НЕ поисковый фильтр.
        # Он применяется только после ответа ArXiv
        # для очевидного удаления биомедицинского мусора.
        forbidden_terms = [
            "gene",
            "protein",
            "patient",
            "clinical",
            "cell biology",
            "tumor",
            "cancer",
            "rna",
            "dna",
        ]

        search = arxiv.Search(
            query=refined_query,
            max_results=self.max_results * 3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers: List[ResearchPaper] = []

        client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3
        )

        try:

            logger.info(
                f"Запрос к ArXiv: {refined_query}"
            )

            for result in client.results(search):

                title_lower = result.title.lower()
                summary_lower = result.summary.lower()

                # Техническая страховка от очевидного
                # биомедицинского шума.
                is_forbidden = any(
                    term in title_lower
                    or term in summary_lower
                    for term in forbidden_terms
                )

                if is_forbidden:

                    logger.info(
                        "Отфильтрована потенциально "
                        f"нерелевантная статья: "
                        f"{result.title}"
                    )

                    continue

                paper = ResearchPaper(
                    id=result.entry_id.split("/")[-1],
                    title=result.title.strip(),
                    summary=result.summary.strip(),
                    authors=[
                        author.name
                        for author in result.authors
                    ],
                    published_date=(
                        result.published.strftime(
                            "%Y-%m-%d"
                        )
                    ),
                    pdf_url=result.pdf_url
                )

                papers.append(paper)

                logger.info(
                    f"[ArXiv] "
                    f"{len(papers)}. "
                    f"{paper.title}"
                )

                if len(papers) >= self.max_results:
                    break

            logger.info(
                f"ArXiv: собрано публикаций "
                f"{len(papers)}"
            )

            return papers

        except Exception as e:

            logger.error(
                f"Ошибка при запросе к ArXiv API: {e}"
            )

            return []


print(
    "✅ collector.py успешно обновлен: "
    "taxonomy.json + академический перевод + "
    "расширенный ArXiv-поиск."
)
