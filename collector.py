import arxiv
from loguru import logger

class ResearchPaper:
    """Контейнер данных для научной публикации из ArXiv."""
    def __init__(self, title, summary, authors, published_date, pdf_url, entry_id):
        self.title = title
        self.summary = summary
        self.authors = authors
        self.published_date = published_date
        self.pdf_url = pdf_url
        self.entry_id = entry_id


class ArxivCollector:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    @staticmethod
    def enhance_query_for_programming(user_query: str) -> str:
        """
        Автоматически расширяет пользовательский запрос контекстом программирования и ML,
        чтобы исключить омонимы (например, вулкан Этна вместо библиотеки ETNA).
        """
        query_lower = user_query.lower()
        
        tech_markers = [
            'python', 'ml', 'ai', 'learning', 'code', 'programming', 
            'library', 'model', 'neural', 'data', 'algorithm', 'software',
            'framework', 'time series', 'forecast', 'nlp', 'cv'
        ]
        
        if not any(term in query_lower for term in tech_markers):
            enhanced = f"({user_query}) AND (python OR programming OR software OR algorithm OR 'machine learning' OR 'deep learning' OR library OR framework OR code)"
            logger.info(f"Запрос автоматически расширен контекстом программирования: {enhanced}")
            return enhanced
        
        return user_query

    async def fetch_papers(self, query: str):
        """
        Асинхронный метод сбора научных публикаций из ArXiv API 
        с учетом фильтрации по программированию и ML.
        """
        processed_query = self.enhance_query_for_programming(query)
        logger.info(f"В библиотеке `arxiv` отправляется запрос: {processed_query}")
        
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=processed_query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in client.results(search):
                paper = ResearchPaper(
                    title=result.title,
                    summary=result.summary,
                    authors=[author.name for author in result.authors],
                    published_date=result.published.strftime("%Y-%m-%d"),
                    pdf_url=result.pdf_url,
                    entry_id=result.entry_id
                )
                papers.append(paper)
                
            logger.info(f"Успешно найдено статей: {len(papers)}")
            return papers

        except Exception as e:
            logger.error(f"Ошибка при запросе к ArXiv API: {e}")
            return []
