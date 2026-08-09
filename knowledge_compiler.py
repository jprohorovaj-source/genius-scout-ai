%%writefile knowledge_compiler.py

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

from loguru import logger

from collector import ResearchPaper


# ============================================================
# Структуры данных
# ============================================================

@dataclass
class CompiledKnowledge:
    """
    Структурированное представление знаний,
    извлечённых из научной публикации.
    """

    paper_id: str
    title: str

    datasets: List[str]
    frameworks: List[str]
    algorithms: List[str]
    metrics: List[str]
    technical_terms: List[str]

    keyword_frequency: Dict[str, int]


# ============================================================
# Knowledge Compiler
# ============================================================

class KnowledgeCompiler:
    """
    Knowledge Extraction Layer GeniusScout AI.

    Назначение:

    1. Получить публикации, прошедшие Semantic Gate.
    2. Извлечь технические сущности.
    3. Определить используемые:
       - datasets;
       - frameworks;
       - algorithms;
       - evaluation metrics.
    4. Построить частотный профиль терминов.
    5. Вернуть структурированные данные,
       пригодные для дальнейшего анализа и FAISS.
    """

    # --------------------------------------------------------
    # Технические словари
    # --------------------------------------------------------

    DATASETS = {
        "mnist",
        "fashion-mnist",
        "cifar-10",
        "cifar-100",
        "imagenet",
        "coco",
        "lfw",
        "celeba",
        "glue",
        "superglue",
        "squad",
        "imdb",
        "yelp",
        "wikitext",
        "openwebtext",
        "common crawl",
        "pascal voc",
        "cityscapes",
        "ade20k",
        "ucf101",
        "kinetics",
        "mmlu",
        "gsm8k",
        "human eval",
        "hellaswag",
    }

    FRAMEWORKS = {
        "pytorch",
        "tensorflow",
        "keras",
        "jax",
        "scikit-learn",
        "sklearn",
        "hugging face",
        "transformers",
        "sentence-transformers",
        "xgboost",
        "lightgbm",
        "catboost",
        "opencv",
        "pandas",
        "numpy",
        "faiss",
        "onnx",
        "cuda",
        "gradio",
    }

    ALGORITHMS = {
        "random forest",
        "decision tree",
        "decision trees",
        "gradient boosting",
        "xgboost",
        "lightgbm",
        "catboost",
        "logistic regression",
        "linear regression",
        "support vector machine",
        "svm",
        "knn",
        "k-nearest neighbors",
        "naive bayes",
        "k-means",
        "dbscan",
        "pca",
        "transformer",
        "self-attention",
        "multi-head attention",
        "cnn",
        "convolutional neural network",
        "resnet",
        "efficientnet",
        "yolo",
        "faster r-cnn",
        "mask r-cnn",
        "gan",
        "vae",
        "lstm",
        "gru",
        "bert",
        "gpt",
        "diffusion model",
    }

    METRICS = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1-score",
        "roc-auc",
        "auc",
        "log loss",
        "cross entropy",
        "mse",
        "rmse",
        "mae",
        "mape",
        "smape",
        "r2",
        "bleu",
        "rouge",
        "perplexity",
        "map",
        "mean average precision",
        "iou",
        "dice",
        "psnr",
        "ssim",
    }

    # Частотные слова, которые не несут
    # самостоятельной технической информации.
    STOPWORDS = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "using",
        "used",
        "use",
        "based",
        "approach",
        "method",
        "model",
        "models",
        "data",
        "dataset",
        "results",
        "paper",
        "study",
        "proposed",
        "new",
        "also",
        "can",
        "may",
        "our",
        "we",
        "they",
        "their",
        "into",
        "than",
        "more",
        "less",
        "via",
        "such",
        "which",
        "have",
        "has",
        "been",
        "are",
        "was",
        "were",
        "is",
        "in",
        "on",
        "of",
        "to",
        "a",
        "an",
        "as",
        "by",
        "or",
        "at",
        "it",
        "its",
    }

    def __init__(self):
        logger.info(
            "KnowledgeCompiler успешно инициализирован."
        )

    # ========================================================
    # Нормализация
    # ========================================================

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Нормализует текст для поиска технических сущностей.
        """

        text = text.lower()

        text = text.replace("–", "-")
        text = text.replace("—", "-")

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ========================================================
    # Поиск сущностей
    # ========================================================

    def _extract_entities(
        self,
        text: str,
        vocabulary: set
    ) -> List[str]:
        """
        Извлекает технические сущности из текста.

        Используется boundary-aware поиск,
        чтобы избежать ложных совпадений.
        """

        normalized = self._normalize(text)

        found = []

        for term in vocabulary:

            escaped = re.escape(
                term.lower()
            )

            pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"

            if re.search(
                pattern,
                normalized
            ):
                found.append(term)

        return sorted(
            set(found)
        )

    # ========================================================
    # Частотный анализ
    # ========================================================

    def _extract_frequency(
        self,
        text: str,
        top_k: int = 20
    ) -> Dict[str, int]:
        """
        Строит простой частотный профиль
        технических терминов.

        Это намеренно лёгкий слой:
        он не заменяет semantic retrieval,
        а дополняет его.
        """

        normalized = self._normalize(text)

        words = re.findall(
            r"\b[a-z][a-z0-9-]{2,}\b",
            normalized
        )

        frequency: Dict[str, int] = {}

        for word in words:

            if word in self.STOPWORDS:
                continue

            frequency[word] = (
                frequency.get(word, 0) + 1
            )

        sorted_frequency = sorted(
            frequency.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return dict(
            sorted_frequency[:top_k]
        )

    # ========================================================
    # Компиляция одной статьи
    # ========================================================

    def compile_paper(
        self,
        paper: ResearchPaper
    ) -> CompiledKnowledge:
        """
        Извлекает структурированные знания
        из одной научной публикации.
        """

        full_text = (
            f"{paper.title}. "
            f"{paper.summary}"
        )

        datasets = self._extract_entities(
            full_text,
            self.DATASETS
        )

        frameworks = self._extract_entities(
            full_text,
            self.FRAMEWORKS
        )

        algorithms = self._extract_entities(
            full_text,
            self.ALGORITHMS
        )

        metrics = self._extract_entities(
            full_text,
            self.METRICS
        )

        frequency = self._extract_frequency(
            full_text
        )

        # Объединяем наиболее важные технические
        # сущности в единый список.
        technical_terms = sorted(
            set(
                datasets
                + frameworks
                + algorithms
                + metrics
            )
        )

        compiled = CompiledKnowledge(
            paper_id=paper.id,
            title=paper.title,
            datasets=datasets,
            frameworks=frameworks,
            algorithms=algorithms,
            metrics=metrics,
            technical_terms=technical_terms,
            keyword_frequency=frequency,
        )

        logger.info(
            f"[KnowledgeCompiler] "
            f"{paper.title[:80]}"
        )

        logger.info(
            f"  datasets={datasets}"
        )

        logger.info(
            f"  frameworks={frameworks}"
        )

        logger.info(
            f"  algorithms={algorithms}"
        )

        logger.info(
            f"  metrics={metrics}"
        )

        return compiled

    # ========================================================
    # Компиляция коллекции
    # ========================================================

    def compile(
        self,
        papers: List[ResearchPaper]
    ) -> List[CompiledKnowledge]:
        """
        Компилирует знания из коллекции публикаций.
        """

        if not papers:

            logger.warning(
                "KnowledgeCompiler получил "
                "пустой список публикаций."
            )

            return []

        logger.info(
            f"KnowledgeCompiler: обработка "
            f"{len(papers)} публикаций..."
        )

        compiled_results = []

        for paper in papers:

            try:

                result = self.compile_paper(
                    paper
                )

                compiled_results.append(
                    result
                )

            except Exception as e:

                logger.error(
                    f"Ошибка компиляции "
                    f"{paper.title}: {e}"
                )

        logger.info(
            f"KnowledgeCompiler завершил работу: "
            f"{len(compiled_results)} документов."
        )

        return compiled_results

    # ========================================================
    # JSON-ready output
    # ========================================================

    def to_dict(
        self,
        compiled: List[CompiledKnowledge]
    ) -> List[Dict[str, Any]]:
        """
        Преобразует результаты в JSON-compatible структуру.
        """

        return [
            asdict(item)
            for item in compiled
        ]


print(
    "✅ knowledge_compiler.py "
    "успешно создан."
)
