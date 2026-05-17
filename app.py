"""
Streamlit interface for document search, graph analysis and risk assessment.
For Master's thesis - cognitive interface for project documentation.
"""

import streamlit as st
import pickle
import json
from pathlib import Path
from collections import Counter
import chromadb

# Page config
st.set_page_config(
    page_title="Инженерный проект: Анализ документов",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Система поддержки принятия решений")
st.subheader("Управление инженерными проектами с применением LLM")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🔍 Информация о проекте")
    st.info(
        """
        **Документы проекта:**
        - 📐 Проектная документация (proect.pdf)
        - 📝 Пояснительная записка (proect_pz.pdf)
        - 💰 Сметная документация (smeta.pdf)
        
        **Модель:** all-MiniLM-L6-v2
        """
    )
    
    # Load and show statistics
    data_file = Path("data/chunks_and_embeddings_clean.pkl")
    if data_file.exists():
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        chunks = data['chunks']
        
        # Count by source
        sources = [c['source'] for c in chunks]
        source_counts = Counter(sources)
        
        total_chunks = len(chunks)
        st.metric("📄 Всего чанков", total_chunks)
        st.caption(f"📐 Проектная: {source_counts.get('proect.pdf', 0)}")
        st.caption(f"📝 Пояснительная: {source_counts.get('proect_pz.pdf', 0)}")
        st.caption(f"💰 Сметная: {source_counts.get('smeta.pdf', 0)}")
    
    st.markdown("---")
    st.caption("🎓 ВКР | Методы и средства управления инженерными проектами с применением LLM")

# ============================================================
# LOAD FUNCTIONS
# ============================================================
@st.cache_resource
def load_vector_store():
    """Load ChromaDB vector store."""
    chroma_dir = Path("data/chroma_db_clean")
    if chroma_dir.exists():
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("project_documents_clean")
        return collection
    return None

@st.cache_resource
def load_chunks():
    """Load chunks from pickle file."""
    data_file = Path("data/chunks_and_embeddings_clean.pkl")
    if data_file.exists():
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return data['chunks']
    return None

@st.cache_resource
def load_graph():
    """Load semantic links graph."""
    links_file = Path("data/semantic_links.json")
    if links_file.exists():
        with open(links_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

@st.cache_resource
def load_risk_report():
    """Load risk report."""
    report_file = Path("data/risk_report.json")
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Load all data
collection = load_vector_store()
chunks = load_chunks()
graph_data = load_graph()
risk_report = load_risk_report()

# Document name mapping
DOC_NAMES = {
    'proect.pdf': '📐 Проектная документация',
    'proect_pz.pdf': '📝 Пояснительная записка',
    'smeta.pdf': '💰 Сметная документация'
}

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔍 Семантический поиск", "🔗 Граф связей", "⚠️ Риски и несоответствия"])

# ============================================================
# TAB 1: SEMANTIC SEARCH
# ============================================================
with tab1:
    st.header("🔍 Семантический поиск по документам")
    
    st.markdown("""
    💡 **Примеры запросов:**
    - "технические требования АСУ ТП"
    - "электромагнитная совместимость"
    - "модернизация системы автоматизации"
    - "смета затрат оборудование"
    """)
    
    if collection:
        # Search mode selection
        st.subheader("🎯 Выберите режим поиска")
        
        search_mode = st.radio(
            "Область поиска:",
            [
                "📊 Все документы",
                "📐 Только проектная документация",
                "📝 Только пояснительная записка",
                "💰 Только сметная документация",
                "🎯 Проектная + Пояснительная (без сметы)"
            ],
            horizontal=True
        )
        
        # Configure filter
        if search_mode == "📐 Только проектная документация":
            where_filter = {"source": "proect.pdf"}
            st.info("🔍 Поиск только в **проектной документации**")
        elif search_mode == "📝 Только пояснительная записка":
            where_filter = {"source": "proect_pz.pdf"}
            st.info("🔍 Поиск только в **пояснительной записке**")
        elif search_mode == "💰 Только сметная документация":
            where_filter = {"source": "smeta.pdf"}
            st.info("🔍 Поиск только в **сметной документации**")
        elif search_mode == "🎯 Проектная + Пояснительная (без сметы)":
            where_filter = {"source": {"$in": ["proect.pdf", "proect_pz.pdf"]}}
            st.info("🔍 Поиск в **проектной документации и пояснительной записке**")
        else:
            where_filter = None
            st.info("🔍 Поиск по **всем документам**")
        
        st.markdown("---")
        
        query = st.text_input("Введите запрос:", placeholder="Например: технические требования")
        
        if query:
            with st.spinner("🔎 Поиск..."):
                try:
                    if where_filter:
                        results = collection.query(
                            query_texts=[query],
                            n_results=10,
                            where=where_filter
                        )
                    else:
                        results = collection.query(
                            query_texts=[query],
                            n_results=10
                        )
                    
                    if results['documents'][0]:
                        st.subheader(f"📋 Результаты поиска ({len(results['documents'][0])}):")
                        
                        for i, (doc, metadata, distance) in enumerate(zip(
                            results['documents'][0],
                            results['metadatas'][0],
                            results['distances'][0]
                        ), 1):
                            similarity = 1 / (1 + distance)
                            
                            if similarity > 0.7:
                                color = "🟢"
                                label = "Высокая релевантность"
                            elif similarity > 0.5:
                                color = "🟡"
                                label = "Средняя релевантность"
                            else:
                                color = "🔴"
                                label = "Низкая релевантность"
                            
                            doc_name = metadata['source']
                            doc_display = DOC_NAMES.get(doc_name, doc_name)
                            
                            with st.expander(f"{color} {i}. {doc_display} — {label} ({similarity:.1%})"):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**📄 Источник:** {doc_name}")
                                    st.markdown(f"**🔢 Чанк №:** {metadata['chunk_id']}")
                                    st.markdown("**📝 Содержание:**")
                                    st.text(doc[:600] + "..." if len(doc) > 600 else doc)
                                with col2:
                                    st.metric("Релевантность", f"{similarity:.1%}")
                    else:
                        st.info("😕 Ничего не найдено. Попробуйте изменить запрос.")
                        
                except Exception as e:
                    st.error(f"Ошибка при поиске: {e}")
    else:
        st.warning("⚠️ Векторное хранилище не найдено. Запустите create_vector_store.py")

# ============================================================
# TAB 2: GRAPH ANALYSIS
# ============================================================
with tab2:
    st.header("🔗 Анализ семантических связей между документами")
    
    if graph_data:
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        params = graph_data.get('parameters', {})
        
        # Statistics
        st.subheader("📊 Статистика семантических связей")
        
        link_types = {}
        for edge in edges:
            lt = edge['link_type']
            link_types[lt] = link_types.get(lt, 0) + 1
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего узлов (чанков)", len(nodes))
        with col2:
            st.metric("Всего связей", len(edges))
        with col3:
            st.metric("Противоречий", link_types.get('contradicts', 0))
        with col4:
            st.metric("Подтверждений", link_types.get('supports', 0))
        
        if link_types:
            st.caption("Распределение по типам связей:")
            cols = st.columns(len(link_types))
            for i, (lt, count) in enumerate(link_types.items()):
                with cols[i]:
                    st.metric(lt, count)
        
        st.markdown("---")
        
        # Parameters
        with st.expander("⚙️ Параметры выявления связей"):
            st.caption(f"K (кандидатов на чанк): {params.get('K', '—')}")
            st.caption(f"Порог сходства: {params.get('similarity_threshold', '—')}")
            st.caption(f"Всего кандидатов: {params.get('total_candidates', '—')}")
        
        # View links by document
        st.subheader("🔍 Просмотр связей по документу")
        
        selected_doc = st.selectbox(
            "Выберите документ:",
            options=list(DOC_NAMES.keys()),
            format_func=lambda x: DOC_NAMES[x]
        )
        
        if selected_doc:
            doc_edges = []
            for edge in edges:
                if edge['source_doc'] == selected_doc or edge['target_doc'] == selected_doc:
                    doc_edges.append(edge)
            
            if doc_edges:
                st.success(f"Найдено {len(doc_edges)} связей")
                
                for link_type in ['contradicts', 'supports', 'elaborates', 'references', 'triggers']:
                    type_edges = [e for e in doc_edges if e['link_type'] == link_type]
                    if type_edges:
                        with st.expander(f"{link_type.upper()} ({len(type_edges)})"):
                            for edge in type_edges[:5]:
                                if edge['source_doc'] == selected_doc:
                                    other_doc = edge['target_doc']
                                    other_text = edge['target_text_preview']
                                    direction = "→"
                                else:
                                    other_doc = edge['source_doc']
                                    other_text = edge['source_text_preview']
                                    direction = "←"
                                
                                st.markdown(f"**{DOC_NAMES.get(other_doc, other_doc)}** {direction}")
                                st.caption(f"Сходство: {edge['similarity']:.1%}")
                                st.text(other_text[:200] + "..." if len(other_text) > 200 else other_text)
                                st.markdown("---")
            else:
                st.info("Связей для выбранного документа не найдено")
    
    else:
        st.warning("⚠️ Граф связей не найден. Запустите:")
        st.code("python extract_semantic_links.py", language="bash")
        st.info("Этот модуль анализирует семантические связи между чанками документов.")
    
    # Quick keyword search (additional feature)
    st.markdown("---")
    st.subheader("🔍 Быстрый поиск по ключевым словам")
    
    if chunks:
        col1, col2 = st.columns([3, 1])
        with col1:
            keyword = st.text_input("Введите ключевое слово:", key="keyword_tab2", placeholder="например: электромагнитная")
        with col2:
            filter_smeta = st.checkbox("Исключить смету", value=True, key="filter_tab2")
        
        if keyword:
            matching_chunks = []
            for chunk in chunks:
                if keyword.lower() in chunk['content'].lower():
                    if filter_smeta and chunk['source'] == 'smeta.pdf':
                        continue
                    matching_chunks.append(chunk)
            
            if matching_chunks:
                st.success(f"✅ Найдено {len(matching_chunks)} чанков")
                for chunk in matching_chunks[:10]:
                    doc_display = DOC_NAMES.get(chunk['source'], chunk['source'])
                    with st.expander(f"📄 {doc_display} (чанк {chunk['chunk_id']})"):
                        st.text(chunk['content'][:400] + "..." if len(chunk['content']) > 400 else chunk['content'])
            else:
                st.info(f"😕 Чанков, содержащих '{keyword}', не найдено")

# ============================================================
# TAB 3: RISK ANALYSIS
# ============================================================
with tab3:
    st.header("⚠️ Анализ рисков и несоответствий")
    
    if risk_report:
        # Display report summary
        summary = risk_report.get('summary', {})
        generated_at = risk_report.get('generated_at', '—')
        
        st.success(f"📋 Отчет загружен (сгенерирован: {generated_at[:19] if generated_at != '—' else '—'})")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего связей", summary.get('total_links', 0))
        with col2:
            st.metric("Несоответствий", summary.get('total_contradictions', 0))
        with col3:
            st.metric("Выявлено рисков", summary.get('total_risks', 0))
        with col4:
            high_risks = summary.get('high_severity_risks', 0)
            st.metric("Рисков высокой серьезности", high_risks, delta="⚠️" if high_risks > 0 else None)
        
        # Risk by type
        risk_by_type = summary.get('risk_by_type', {})
        if risk_by_type:
            st.subheader("📊 Распределение рисков по типам")
            cols = st.columns(len(risk_by_type))
            for i, (risk_type, count) in enumerate(risk_by_type.items()):
                with cols[i]:
                    st.metric(risk_type, count)
        
        st.markdown("---")
        
        # Detailed risks
        st.subheader("⚠️ Детализация выявленных рисков")
        
        risks = risk_report.get('risks', [])
        if risks:
            for risk in risks:
                severity_icon = "🔴" if risk['severity'] == 'high' else "🟡" if risk['severity'] == 'medium' else "🟢"
                with st.expander(f"{severity_icon} Риск {risk['contradiction_id']} (серьезность: {risk['severity']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Документ 1:** {DOC_NAMES.get(risk['source_doc'], risk['source_doc'])}")
                        st.text(risk['source_text'][:300] + "..." if len(risk['source_text']) > 300 else risk['source_text'])
                    with col2:
                        st.markdown(f"**Документ 2:** {DOC_NAMES.get(risk['target_doc'], risk['target_doc'])}")
                        st.text(risk['target_text'][:300] + "..." if len(risk['target_text']) > 300 else risk['target_text'])
                    
                    st.markdown(f"**Типы рисков:** {', '.join(risk['risk_types'])}")
                    if risk.get('extracted_numbers'):
                        st.markdown(f"**Числовые значения:** {risk['extracted_numbers']}")
                    st.markdown(f"**Семантическое сходство:** {risk['similarity']:.1%}")
        else:
            st.info("✅ Несоответствий не выявлено")
        
        # Recommendations
        recommendations = risk_report.get('recommendations', [])
        if recommendations:
            st.subheader("💡 Рекомендации")
            for rec in recommendations:
                priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                st.info(f"{priority_icon} {rec['text']}")
        
        # Refresh button
        if st.button("🔄 Обновить отчет", key="refresh_report"):
            st.cache_data.clear()
            st.rerun()
    
    else:
        st.warning("⚠️ Отчет по рискам не найден. Запустите:")
        st.code("python extract_semantic_links.py && python analyze_risks.py", language="bash")
        st.info("""
        Эти модули:
        1. Выявляют семантические связи между документами
        2. Анализируют несоответствия (contradicts)
        3. Классифицируют риски (финансовые, сроковые, юридические)
        4. Формируют отчет в `data/risk_report.json`
        """)
    
    # Quick risk analysis (legacy feature)
    st.markdown("---")
    st.subheader("🔍 Быстрый анализ по ключевым словам")
    
    if chunks:
        col1, col2 = st.columns(2)
        
        with col1:
            risk_scope = st.radio(
                "Анализировать документы:",
                ["Все документы", "Без сметы", "Только проектная", "Только ПЗ"],
                key="risk_scope_tab3"
            )
        
        with col2:
            if st.button("🔍 Выполнить быстрый анализ", key="quick_analysis"):
                with st.spinner("Анализ документов..."):
                    risks_found = []
                    
                    risk_keywords = {
                        '🔴 Технические риски': ['несоответствие', 'отказ', 'сбой', 'ошибка', 'дефект', 'авария'],
                        '🟡 Сметные риски': ['удорожание', 'перерасход', 'смета', 'стоимость', 'затраты', 'бюджет'],
                        '🟢 Сроковые риски': ['задержка', 'срок', 'график', 'отставание', 'перенос']
                    }
                    
                    filtered_chunks = []
                    scope_map = {
                        "Все документы": None,
                        "Без сметы": lambda c: c['source'] != 'smeta.pdf',
                        "Только проектная": lambda c: c['source'] == 'proect.pdf',
                        "Только ПЗ": lambda c: c['source'] == 'proect_pz.pdf'
                    }
                    
                    filter_func = scope_map.get(risk_scope)
                    for chunk in chunks:
                        if filter_func is None or filter_func(chunk):
                            filtered_chunks.append(chunk)
                    
                    for chunk in filtered_chunks:
                        content_lower = chunk['content'].lower()
                        for risk_type, keywords in risk_keywords.items():
                            for keyword in keywords:
                                if keyword in content_lower:
                                    sentences = chunk['content'].split('.')
                                    for sentence in sentences:
                                        if keyword in sentence.lower():
                                            risks_found.append({
                                                'type': risk_type,
                                                'keyword': keyword,
                                                'text': sentence.strip()[:350],
                                                'source': chunk['source'],
                                                'chunk_id': chunk['chunk_id']
                                            })
                                            break
                    
                    # Remove duplicates
                    unique_risks = []
                    seen = set()
                    for risk in risks_found:
                        key = f"{risk['source']}_{risk['keyword']}_{risk['text'][:50]}"
                        if key not in seen:
                            seen.add(key)
                            unique_risks.append(risk)
                    
                    st.session_state['quick_risks'] = unique_risks
                    st.session_state['quick_risk_analyzed'] = True
                    st.rerun()
        
        if 'quick_risk_analyzed' in st.session_state and st.session_state['quick_risk_analyzed']:
            quick_risks = st.session_state.get('quick_risks', [])
            
            if quick_risks:
                st.success(f"✅ Найдено {len(quick_risks)} потенциальных рисков")
                
                # Group by type
                risk_types = {}
                for risk in quick_risks:
                    if risk['type'] not in risk_types:
                        risk_types[risk['type']] = []
                    risk_types[risk['type']].append(risk)
                
                for risk_type, type_risks in risk_types.items():
                    with st.expander(f"{risk_type} ({len(type_risks)})"):
                        for risk in type_risks[:10]:
                            doc_display = DOC_NAMES.get(risk['source'], risk['source'])
                            st.markdown(f"**{risk['keyword']}** - {doc_display} (чанк {risk['chunk_id']})")
                            st.caption(risk['text'][:200])
                            st.markdown("---")
            else:
                st.info("✅ Рисков не обнаружено")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("🎓 Разработано в рамках ВКР | Методы и средства управления инженерными проектами с применением LLM")