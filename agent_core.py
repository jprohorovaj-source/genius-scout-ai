from loguru import logger
from deep_translator import GoogleTranslator

from collector import ArxivCollector
from gate import SemanticGate
from vector_store import VectorStoreManager
from config import ARXIV_TOP_RESULTS


class GeniusScoutAgent:
    """
    Центральный интеллектуальный агент GeniusScout AI.
    """

    def __init__(self):
        logger.info("Инициализация GeniusScoutAgent...")
        self.collector = ArxivCollector(max_results=ARXIV_TOP_RESULTS)
        self.semantic_gate = SemanticGate()
        self.vector_store = VectorStoreManager()
        logger.info("GeniusScoutAgent успешно инициализирован.")

    async def run_audit(self, query: str) -> str:
        logger.info(f"Запуск исследования темы: {query}")

        papers = await self.collector.fetch_papers(query)

        if len(papers) == 0:
            return (
                "# Результат исследования\n\n"
                f"По запросу **«{query}»** публикации на ArXiv не найдены. "
                "Попробуйте уточнить запрос (например, вместо 'Gemini' указать 'Gemini model' или 'multimodal language models')."
            )

        papers = self.semantic_gate.filter_papers(query, papers)

        if len(papers) == 0:
            return (
                "# Результат исследования\n\n"
                "Публикации найдены, но после семантической проверки релевантные статьи отсутствуют."
            )

        self.vector_store.add_papers(papers)

        report = []
        report.append("# GeniusScout AI")
        report.append("")
        report.append("## Тема исследования")
        report.append(f"**{query}**")
        report.append("")
        report.append("---")
        report.append("")
        report.append("## Проверенные научные публикации")
        report.append("")

        translator = GoogleTranslator(source='en', target='ru')

        for i, paper in enumerate(papers, start=1):
            # Переводим название и аннотацию на русский язык для удобства чтения
            try:
                ru_title = translator.translate(paper.title)
            except Exception:
                ru_title = paper.title

            try:
                ru_summary = translator.translate(paper.summary)
            except Exception:
                ru_summary = paper.summary

            report.append(f"### {i}. {ru_title}")
            report.append(f"*(Оригинал: {paper.title})*")
            report.append("")
            report.append(f"**Авторы:** {', '.join(paper.authors) if isinstance(paper.authors, list) else paper.authors}")
            report.append("")
            report.append(f"**Дата публикации:** {paper.published_date}")
            report.append("")
            report.append("**Аннотация (на русском языке):**")
            report.append("")
            report.append(ru_summary)
            report.append("")
            report.append(f"**PDF:** {paper.pdf_url}")
            report.append("")
            report.append("---")
            report.append("")

        report.append("## Заключение")
        report.append("")
        report.append("В отчет включены только публикации, успешно прошедшие семантическую проверку Semantic Gate.")
        report.append("Все статьи сохранены в локальной векторной базе знаний FAISS.")
        report.append("")
        report.append("© 2026 GeniusScout AI • Developed by Юлия Прохорова")

        logger.info("Отчет успешно сформирован с переводом.")
        return "\n".join(report)


print("✅ agent_core.py успешно обновлен с переводом отчетов.")
