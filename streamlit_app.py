import streamlit as st
import asyncio
import sys
import os

# Импортируем нашего агента из существующего ядра
from agent_core import GeniusScoutAgent

# Настройка страницы
st.set_page_config(
    page_title="GeniusScout AI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 GeniusScout AI v10.5")
st.markdown("### Автономная AI-система поиска, семантического анализа и компиляции научных знаний из ArXiv")

# Кэшируем инициализацию агента, чтобы модель и векторная база не загружались заново при каждом клике
@st.cache_resource
def get_agent():
    return GeniusScoutAgent()

try:
    with st.spinner("Инициализация ядра агента и загрузка компонентов..."):
        agent = get_agent()
except Exception as e:
    st.error(f"Ошибка при инициализации агента: {e}")

# Поле ввода поискового запроса
query = st.text_input(
    "Введите исследовательский запрос или тему:",
    placeholder="Например: Vision Transformer for RAG optimization"
)

if st.button("Запустить скаут-анализ", type="primary"):
    if query:
        with st.spinner("Выполняется поиск в ArXiv, семантическая фильтрация через Semantic Gate и генерация отчета..."):
            try:
                # Запускаем асинхронный метод run_audit из нашего ядра
                report = asyncio.run(agent.run_audit(query))
                
                st.success("Исследование успешно завершено!")
                st.markdown("---")
                
                # Выводим реальный сгенерированный отчет агента
                st.markdown(report)
                
            except Exception as e:
                st.error(f"Произошла ошибка при выполнении пайплайна: {e}")
    else:
        st.warning("Пожалуйста, введите текст запроса.")

# Боковая панель с информацией о проекте
with st.sidebar:
    st.markdown("### О системе")
    st.info(
        "**GeniusScout AI** — это автономный агент для поиска и глубокого семантического анализа научных публикаций.\n\n"
        "• **Двухэтапный RAG:** Bi-Encoder + Cross-Encoder\n"
        "• **Векторная база:** FAISS\n"
        "• **Источник:** ArXiv API"
    )
    st.markdown("---")
    st.markdown("**Автор:** Юлия Прохорова")
    st.markdown("[GitHub репозиторий](https://github.com/jprohorovaj-source/genius-scout-ai)")
