
from loguru import logger

from collector import ArxivCollector
from gate import SemanticGate
from vector_store import VectorStoreManager

from config import ARXIV_TOP_RESULTS


class GeniusScoutAgent:
    """
    Центральный интеллектуальный агент GeniusScout AI.

    Последовательно выполняет:

        1. Поиск научных публикаций.
        2. Семантическую фильтрацию.
        3. Индексацию документов.
        4. Генерацию экспертного отчета.
    """

    def __init__(self):

        logger.info("Инициализация GeniusScoutAgent...")

        self.collector = ArxivCollector(
            max_results=ARXIV_TOP_RESULTS
        )

        self.semantic_gate = SemanticGate()

        self.vector_store = VectorStoreManager()

        logger.info("GeniusScoutAgent успешно инициализирован.")

    async def run_audit(self, query: str) -> str:

        logger.info(f"Запуск исследования темы: {query}")

        # -------------------------------------------------
        # Этап 1. Поиск публикаций
        # -------------------------------------------------

        papers = await self.collector.fetch_papers(query)

        if len(papers) == 0:
            return (
                "# Результат исследования\n\n"
                "По заданному запросу публикации не найдены."
            )

        # -------------------------------------------------
        # Этап 2. Semantic Gate
        # -------------------------------------------------

        papers = self.semantic_gate.filter_papers(
            query,
            papers
        )

        if len(papers) == 0:
            return (
                "# Результат исследования\n\n"
                "После семантической проверки релевантные публикации отсутствуют."
            )

        # -------------------------------------------------
        # Этап 3. Индексация
        # -------------------------------------------------

        self.vector_store.add_papers(papers)

        # -------------------------------------------------
        # Этап 4. Формирование отчета
        # -------------------------------------------------

        report = []

        report.append(f"# GeniusScout AI")
        report.append("")
        report.append(f"## Тема исследования")
        report.append(f"**{query}**")
        report.append("")
        report.append("---")
        report.append("")
        report.append("## Проверенные научные публикации")
        report.append("")

        for i, paper in enumerate(papers, start=1):

            report.append(f"### {i}. {paper.title}")
            report.append("")
            report.append(f"**Авторы:** {paper.authors}")
            report.append("")
            report.append(f"**Дата публикации:** {paper.published_date}")
            report.append("")
            report.append("**Аннотация:**")
            report.append("")
            report.append(paper.summary)
            report.append("")
            report.append(f"**PDF:** {paper.pdf_url}")
            report.append("")
            report.append("---")
            report.append("")

        report.append("## Заключение")
        report.append("")
        report.append(
            "В отчет включены только публикации, успешно прошедшие "
            "семантическую проверку Semantic Gate."
        )
        report.append("")
        report.append(
            "Все статьи сохранены в локальной векторной базе знаний FAISS "
            "и доступны для дальнейшего семантического поиска."
        )
        report.append("")
        report.append("---")
        report.append("")
        report.append(
            "© 2026 GeniusScout AI • Developed by Юлия Прохорова"
        )

        logger.info("Отчет успешно сформирован.")

        return "\n".join(report)


print("✅ agent_core.py успешно создан.")
