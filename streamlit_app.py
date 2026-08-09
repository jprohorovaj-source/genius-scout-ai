import streamlit as st
import os

# Настройка страницы
st.set_page_config(
    page_title="GeniusScout AI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 GeniusScout AI v10.5")
st.markdown("### Автономная AI-система поиска, семантического анализа и компиляции научных знаний из ArXiv")

# Поле для ввода запроса пользователя
query = st.text_input("Введите тему или исследовательский вопрос:", placeholder="Например: Large Language Models for RAG optimization")

if st.button("Запустить поиск и анализ", type="primary"):
    if query:
        with st.spinner("Агент ищет статьи, строит эмбеддинги и формирует отчет..."):
            # Здесь вызывается логика вашего пайплайна из agent_core.py
            # Пример заглушки, которую мы заменим на вызов вашего движка:
            # results = run_genius_scout(query)
            
            st.success("Анализ завершен!")
            st.markdown("#### Результаты работы агента:")
            st.info(f"По вашему запросу (*{query}*) успешно сформирован отчет и найдено релевантные исследования.")
            
            # Пример вывода текста отчета
            st.markdown("---")
            st.markdown("### Итоговый инженерный отчет")
            st.write("Здесь будет отображаться глубокий аналитический отчет, сгенерированный вашим пайплайном.")
    else:
        st.warning("Пожалуйста, введите поисковый запрос.")
