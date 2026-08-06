#  GeniusScout AI v10.5
## Autonomous Multi-Agent RAG & Academic Knowledge Compiler

<p align="center">
Enterprise-grade autonomous AI research assistant for real-time scientific intelligence extraction
</p>

---

#  Author / Автор

**Yuliya Prokhorova (Юлия Прохорова)**

 Contact: j.prohorova.j@gmail.com

🔗 GitHub Repository:

https://github.com/jprohorovaj-source/genius-scout-ai

 Qualification:

Skillbox Certified Machine Learning Junior  
Certificate № SKB0493892  
Issued: 15.06.2026

---

#  Project Overview

## English Description

**GeniusScout AI** is an autonomous **RAG (Retrieval-Augmented Generation) platform** designed for automated scientific knowledge discovery, semantic verification, and structured research analysis.

The system creates a complete AI-powered research pipeline:

- intelligent academic query generation;
- real-time arXiv paper collection;
- semantic relevance filtering;
- automatic knowledge extraction;
- structured Markdown reporting.

The project is engineered using:

- Clean Architecture principles;
- SOLID design principles;
- modular AI-agent architecture;
- deterministic processing pipelines;
- Docker deployment.

---

#  Business Value

Modern researchers and ML engineers spend significant time on:

- searching scientific publications;
- analyzing hundreds of papers;
- identifying used datasets and frameworks;
- comparing approaches;
- tracking new research directions.

GeniusScout AI transforms this workflow into an autonomous intelligence pipeline.

---

#  Core Technical Features


## 1. Real-Time Scientific Intelligence

The platform directly connects to arXiv sources.

Benefits:

- no dependency on static knowledge cutoff;
- access to newly published research;
- continuous academic intelligence updates.

---

## 2. Semantic Noise Filtering

Implemented adaptive semantic gate:

~~~text
gate.py
~~~

Powered by:

~~~text
cross-encoder/ms-marco-MiniLM-L-6-v2
~~~

Responsibilities:

- semantic relevance scoring;
- duplicate research elimination;
- domain ambiguity reduction;
- mathematical threshold filtering.

---

## 3. Autonomous Knowledge Compiler

Module:

~~~text
knowledge_compiler.py
~~~

Automatically extracts:

- machine learning frameworks;
- datasets;
- evaluation metrics;
- algorithms;
- research terminology.

The system uses statistical extraction methods instead of fragile rule-based parsing.

---

## 4. Fault-Tolerant Architecture

Implemented:

- fallback routing;
- isolated components;
- external API resilience;
- dependency separation.

Goal:

Reliable execution under unstable external conditions.

---

#  System Architecture

~~~text

                USER QUERY

                    |
                    v

        +--------------------------+
        | AcademicQueryBuilder     |
        | arXiv DSL Compiler       |
        +------------+-------------+

                     |

                     v

        +--------------------------+
        | arXiv Upstream API       |
        | Real-time Papers Source  |
        +------------+-------------+

                     |

                     v

        +--------------------------+
        | Semantic Gate (ARG)      |
        | Cross Encoder Filtering  |
        +------------+-------------+

                     |

                     v

        +--------------------------+
        | Knowledge Extractor      |
        | Statistical Compiler     |
        +------------+-------------+

                     |

                     v

        +--------------------------+
        | MarkdownRenderer + UI    |
        | Gradio Presentation      |
        +--------------------------+

~~~

---

#  Repository Structure


~~~text
genius-scout-ai/

│
├── agent_core.py
│   └── Multi-agent orchestration layer
│
├── app.py
│   └── Gradio web interface
│
├── collector.py
│   └── Academic query builder
│
├── config.py
│   └── Central configuration
│
├── gate.py
│   └── Semantic relevance filtering
│
├── knowledge_compiler.py
│   └── Knowledge extraction engine
│
├── vector_store.py
│   └── FAISS vector database manager
│
├── taxonomy.json
│   └── Domain taxonomy configuration
│
├── Dockerfile
│   └── Production containerization
│
├── requirements.txt
│   └── Python dependencies
│
└── .gitignore

~~~

---

#  Technology Stack


## Artificial Intelligence

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Cross-Encoder Models
- NLP Processing
- Knowledge Extraction


## Machine Learning

- Sentence Transformers
- FAISS Vector Search
- Statistical Feature Engineering


## Backend

- Python
- Modular Architecture
- Agent-based Design


## Deployment

- Docker
- Gradio
- Production packaging

---

#  Quick Start


## Clone Repository


~~~bash
git clone https://github.com/jprohorovaj-source/genius-scout-ai.git

cd genius-scout-ai
~~~


## Install Dependencies


~~~bash
pip install -r requirements.txt
~~~


## Run Application


~~~bash
python app.py
~~~


Application:

~~~text
http://127.0.0.1:7860
~~~

---

#  Docker Deployment


Build image:


~~~bash
docker build -t genius-scout-ai:v10.5 .
~~~


Run container:


~~~bash
docker run -d \
-p 7860:7860 \
--name genius-scout-app \
genius-scout-ai:v10.5
~~~

---

# 🇷🇺 Русская версия


# Описание проекта


**GeniusScout AI** — автономная интеллектуальная RAG-платформа для анализа научных исследований.


Система автоматически:


- формирует академические запросы;
- получает актуальные публикации arXiv;
- выполняет семантическую фильтрацию;
- извлекает знания;
- формирует структурированные отчёты.


---

# Инженерная ценность проекта


## Работа с актуальными знаниями

Прямая интеграция с arXiv позволяет анализировать новые исследования без ожидания обновления модели.


---

## Семантический фильтр качества


Модуль:

~~~text
gate.py
~~~


использует:


~~~text
cross-encoder/ms-marco-MiniLM-L-6-v2
~~~


для:

- оценки релевантности;
- удаления информационного шума;
- повышения качества поиска.


---

## Компилятор знаний


Модуль:

~~~text
knowledge_compiler.py
~~~


извлекает:


- библиотеки;
- датасеты;
- модели;
- метрики;
- алгоритмы.


---

# 📈 Project Vision


GeniusScout AI demonstrates practical implementation of modern AI Engineering:


- Autonomous Agents
- RAG Pipelines
- Semantic Intelligence
- Scientific Data Mining
- Production Deployment


---

#  Future Development


Planned improvements:


- multi-agent collaboration;
- local LLM support;
- hybrid vector and graph retrieval;
- automatic research comparison;
- knowledge graph generation;
- cloud deployment.


---

<p align="center">
Built with Python • AI Engineering • RAG • Machine Learning
</p>
