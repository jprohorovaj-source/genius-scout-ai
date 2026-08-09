import arxiv
from loguru import logger
from deep_translator import GoogleTranslator
from dataclasses import dataclass
from typing import List


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
    def __init__(self):
        logger.info("ContextDisambiguator успешно инициализирован.")

    def enrich_query(self, user_query: str) -> str:
        """
        Переводит запрос на английский (если нужно) и формирует
        поисковый запрос для ArXiv с учетом длины ключевых слов.
        """
        query_lower = user_query.lower().strip()

        if any(ch in 'абвгдежзийклмнопрстуфхцчшщъыьэюя' for ch in query_lower):
            try:
                translated = GoogleTranslator(source='ru', target='en').translate(user_query)
                logger.info(f"Перевод запроса с ru на en: '{user_query}' -> '{translated}'")
                query_en = translated
            except Exception as e:
                logger.error(f"Ошибка перевода: {e}")
                query_en = user_query
        else:
            query_en = user_query

        # Если запрос короткий (например, "Gemini", "Llama", "FAISS"), ищем напрямую без жестких ограничений
        if len(query_en.split()) <= 2:
            enriched = query_en
        else:
            ml_strict_keywords = "(machine learning OR deep learning OR neural networks OR algorithm)"
            enriched = f"({query_en}) AND {ml_strict_keywords}"

        logger.info(f"Итоговый обогащенный поисковый запрос для ArXiv: '{enriched}'")
        return enriched


class ArxivCollector:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.disambiguator = ContextDisambiguator()
        logger.info("ArxivCollector успешно инициализирован.")

    async def fetch_papers(self, query: str) -> List[ResearchPaper]:
        """
        Собирает статьи с ArXiv с фильтрацией нерелевантных доменов.
        """
        refined_query = self.disambiguator.enrich_query(query)

        # Стоп-слова для проверки на уровне метаданных (биология, медицина)
        forbidden_terms = ['gene', 'protein', 'patient', 'clinical', 'cell', 'biol', 'disease', 'tumor', 'rna', 'dna']

        search = arxiv.Search(
            query=refined_query,
            max_results=self.max_results * 2,  # запрашиваем с запасом для отсева
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        client = arxiv.Client()

        try:
            for result in client.results(search):
                title_lower = result.title.lower()
                summary_lower = result.summary.lower()

                is_forbidden = any(term in title_lower or term in summary_lower for term in forbidden_terms)
                if is_forbidden:
                    logger.info(f"Отфильтрована нерелевантная (биологическая/медицинская) статья: {result.title}")
                    continue

                paper = ResearchPaper(
                    id=result.entry_id.split('/')[-1],
                    title=result.title,
                    summary=result.summary,
                    authors=[author.name for author in result.authors],
                    published_date=result.published.strftime("%Y-%m-%d"),
                    pdf_url=result.pdf_url
                )
                papers.append(paper)

                if len(papers) >= self.max_results:
                    break

            logger.info(f"Собрано релевантных ML-статей: {len(papers)}")
            return papers

        except Exception as e:
            logger.error(f"Ошибка при запросе к ArXiv API: {e}")
            return []


print("✅ collector.py успешно обновлен.")
