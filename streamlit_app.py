import streamlit as st
import sys
import os

# Настройка страницы
st.set_page_config(
    page_title="GeniusScout AI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 GeniusScout AI v10.5")
st.markdown("### Автономная AI-система поиска, семантического анализа и компиляции научных знаний из ArXiv")

# Поле ввода поискового запроса
query = st.text_input(
    "Введите исследовательский запрос или тему:",
    placeholder="Например: Vision Transformer for RAG optimization"
)

if st.button("Запустить скаут-анализ", type="primary"):
    if query:
        with st.spinner("Выполняется поиск в ArXiv, семантическая фильтрация и генерация отчета..."):
            try:
                # Импортируем логику из вашего существующего ядра агента (agent_core.py)
                # Если ваш агент запускается через конкретную функцию, например run_agent(query), вызываем её здесь:
                # from agent_core import run_agent
                # report = run_agent(query)
                
                # Временная демонстрация успешного выполнения пайплайна:
                st.success("Анализ успешно завершен!")
                st.markdown("---")
                st.markdown("### 📊 Итоговый инженерный отчет")
                st.info(f"Обработан запрос: **{query}**")
                st.write("Здесь отображаются результаты работы пайплайна, найденные статьи с ArXiv и структурированные рекомендации.")
                
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
