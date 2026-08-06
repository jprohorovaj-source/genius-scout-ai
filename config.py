"""
==========================================================
GeniusScout AI v10.5
Global Production Configuration
==========================================================
"""

ARXIV_MAX_RESULTS = 15      # Первичный пул документов для расширенной выборки
ARXIV_TOP_RESULTS = 3       # Количество препринтов для финального глубокого анализа
ARXIV_SORT = "relevance"    # Критерий сортировки поискового движка ArXiv

ARXIV_DEFAULT_CATEGORY = [
    "cs.LG",   # Machine Learning
    "cs.AI",   # Artificial Intelligence
    "stat.ML"  # Machine Learning (Statistical)
]

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SEMANTIC_THRESHOLD = -20.0  # Мягкий порог для корректной работы Cross-Encoder с отрицательными скорами

TOP_K = 5
PROJECTION = "umap"         # Метод снижения размерности для диагностического модуля (UMAP / PCA)
THEME = "presentation"
TRANSLATE_TARGET = "en"     # Целевой язык для отправки запросов в ArXiv API
LOG_LEVEL = "INFO"
