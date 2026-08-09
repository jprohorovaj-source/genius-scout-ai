import arxiv
from loguru import logger

def enhance_query_for_programming(user_query: str) -> str:
    """
    Автоматически расширяет пользовательский запрос контекстом программирования и ML,
    чтобы исключить омонимы (например, вулкан Этна вместо библиотеки ETNA).
    """
    query_lower = user_query.lower()
    
    # Базовые маркеры IT/ML, по которым определяем, есть ли уже технический контекст
    tech_markers = [
        'python', 'ml', 'ai', 'learning', 'code', 'programming', 
        'library', 'model', 'neural', 'data', 'algorithm', 'software',
        'framework', 'time series', 'forecast', 'nlp', 'cv'
    ]
    
    # Если в запросе нет явных технических терминов, дополняем его поисковым условием для ArXiv
    if not any(term in query_lower for term in tech_markers):
        # Оборачиваем исходный запрос и добавляем жесткую привязку к IT/программированию
        enhanced = f"({user_query}) AND (python OR programming OR software OR algorithm OR 'machine learning' OR 'deep learning' OR library OR framework OR code)"
        logger.info(f"Запрос автоматически расширен контекстом программирования: {enhanced}")
        return enhanced
    
    return user_query


def collect_papers(query: str, max_results: int = 3):
    """
    Собирает научные публикации из ArXiv API с учетом автоматической фильтрации 
    и сужения области поиска исключительно до программирования и ML.
    """
    # Применяем защитный механизм от нерелевантных тематик (геология, физика и т.д.)
    processed_query = enhance_query_for_programming(query)
    
    logger.info(библиотеке `arxiv` отправляется запрос: {processed_query})
    
    try:
        # Настройка клиента ArXiv
        client = arxiv.Client()
        
        search = arxiv.Search(
            query=processed_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            paper = {
                "title": result.title,
                "summary": result.summary,
                "authors": [author.name for author in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id
            }
            papers.append(paper)
            
        logger.info(f"Успешно найдено статей: {len(papers)}")
        return papers

    except Exception as e:
        logger.error(f"Ошибка при запросе к ArXiv API: {e}")
        return []
