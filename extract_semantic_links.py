"""
Extract semantic links between document chunks using LLM (emulation mode).
For Master's thesis - demonstrates the method with rule-based emulation.
"""

import pickle
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from numpy.linalg import norm

print("="*70)
print("🔗 ВЫЯВЛЕНИЕ СЕМАНТИЧЕСКИХ СВЯЗЕЙ МЕЖДУ ДОКУМЕНТАМИ")
print("="*70)
print("Режим: эмуляция LLM (на основе эвристических правил)")
print()

# 1. Загрузка данных
print("📂 Шаг 1: Загрузка чанков и эмбеддингов...")

data_file = Path("data/chunks_and_embeddings_clean.pkl")
if not data_file.exists():
    print(f"❌ Файл {data_file} не найден!")
    print("   Сначала запустите: python create_embeddings_sbert_fixed.py")
    exit(1)

with open(data_file, 'rb') as f:
    data = pickle.load(f)

chunks = data['chunks']
embeddings = data['embeddings']
print(f"✓ Загружено {len(chunks)} чанков")
print(f"✓ Размерность эмбеддингов: {embeddings.shape[1]}")

# 2. Функция эмуляции LLM (на основе ключевых слов)
def emulate_llm_link_type(text_a, text_b, source_a, source_b):
    """
    Эмуляция определения типа связи.
    В реальной системе здесь был бы вызов LLM API.
    """
    text_a_lower = text_a.lower()
    text_b_lower = text_b.lower()
    
    # Паттерны для разных типов связей
    # 1. Противоречие (contradicts)
    contradict_patterns = [
        (r'не соответствует', r'не соответствует'),
        (r'противоречит', r'противоречит'),
        (r'отличается от', r'отличается от'),
        (r'расхождение', r'расхождение'),
        (r'(\d+)\s*(м3|м2|шт|т|км)', r'(\d+)\s*(м3|м2|шт|т|км)')  # разные цифры
    ]
    
    for pattern in contradict_patterns:
        # Ищем числовые несоответствия
        if isinstance(pattern, tuple):
            nums_a = re.findall(pattern[0], text_a_lower)
            nums_b = re.findall(pattern[1], text_b_lower)
            if nums_a and nums_b and nums_a != nums_b:
                return "contradicts"
        else:
            if pattern in text_a_lower and pattern in text_b_lower:
                return "contradicts"
    
    # 2. Подтверждение (supports)
    support_patterns = ['соответствует', 'согласно', 'в соответствии', 'утверждаю', 'подтверждаю']
    for pattern in support_patterns:
        if pattern in text_a_lower and pattern in text_b_lower:
            return "supports"
    
    # 3. Детализация (elaborates)
    if len(text_b) > len(text_a) * 1.5 and source_a != source_b:
        if any(word in text_b_lower for word in ['подробно', 'включает', 'состоит', 'разделе']):
            return "elaborates"
    
    # 4. Ссылка (references)
    ref_patterns = [r'см\.\s+раздел', r'согласно\s+п\.', r'пункт\s+\d+', r'раздел\s+\d+']
    for pattern in ref_patterns:
        if re.search(pattern, text_a_lower) or re.search(pattern, text_b_lower):
            return "references"
    
    # 5. По умолчанию — нет связи
    return "none"

# 3. Поиск кандидатов на связь
print("\n🔍 Шаг 2: Поиск кандидатов на семантическую связь...")

K = 5  # количество кандидатов на чанк
SIMILARITY_THRESHOLD = 0.5

# Нормализуем эмбеддинги для косинусного сходства
embeddings_norm = embeddings / norm(embeddings, axis=1, keepdims=True)

# Для каждого чанка ищем топ-K похожих из других документов
candidates = defaultdict(list)

for i in range(len(chunks)):
    similarities = embeddings_norm[i] @ embeddings_norm.T
    
    # Исключаем сам себя
    similarities[i] = -1
    
    # Получаем топ-K индексов
    top_indices = np.argsort(similarities)[-K:][::-1]
    
    for j in top_indices:
        sim = similarities[j]
        if sim >= SIMILARITY_THRESHOLD:
            # Исключаем пары из одного документа
            if chunks[i]['source'] != chunks[j]['source']:
                candidates[i].append({
                    'target_idx': j,
                    'similarity': float(sim)
                })

total_candidates = sum(len(v) for v in candidates.values())
print(f"✓ Найдено {total_candidates} пар-кандидатов")
print(f"  (при K={K}, порог={SIMILARITY_THRESHOLD})")

# 4. LLM-анализ (эмуляция)
print("\n🤖 Шаг 3: Определение типов связей (эмуляция LLM)...")

links = []

for source_idx, targets in candidates.items():
    source_chunk = chunks[source_idx]
    
    for target in targets:
        target_idx = target['target_idx']
        target_chunk = chunks[target_idx]
        
        # Эмуляция LLM
        link_type = emulate_llm_link_type(
            source_chunk['content'],
            target_chunk['content'],
            source_chunk['source'],
            target_chunk['source']
        )
        
        if link_type != "none":
            links.append({
                'source_id': source_chunk['id'],
                'source_doc': source_chunk['source'],
                'source_chunk_id': source_chunk['chunk_id'],
                'source_text_preview': source_chunk['content'][:200],
                'target_id': target_chunk['id'],
                'target_doc': target_chunk['source'],
                'target_chunk_id': target_chunk['chunk_id'],
                'target_text_preview': target_chunk['content'][:200],
                'link_type': link_type,
                'similarity': target['similarity']
            })

print(f"✓ Выявлено {len(links)} семантических связей")

# 5. Статистика по типам связей
print("\n📊 Статистика по типам связей:")
link_stats = defaultdict(int)
for link in links:
    link_stats[link['link_type']] += 1

for link_type, count in sorted(link_stats.items(), key=lambda x: -x[1]):
    print(f"   {link_type}: {count}")

# 6. Сохранение графа связей
print("\n💾 Шаг 4: Сохранение графа связей...")

graph_data = {
    'nodes': [
        {
            'id': chunk['id'],
            'source': chunk['source'],
            'chunk_id': chunk['chunk_id'],
            'text_preview': chunk['content'][:300]
        }
        for chunk in chunks
    ],
    'edges': links,
    'parameters': {
        'K': K,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'total_chunks': len(chunks),
        'total_candidates': total_candidates,
        'total_links': len(links)
    }
}

output_file = Path("data/semantic_links.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, ensure_ascii=False, indent=2)

print(f"✓ Сохранено в {output_file}")
print(f"   Размер файла: {output_file.stat().st_size / 1024:.1f} KB")

# 7. Вывод примеров связей
print("\n📝 Примеры выявленных связей:")

# Группируем по типам и показываем по одному примеру
shown_types = set()
for link in links:
    if link['link_type'] not in shown_types:
        shown_types.add(link['link_type'])
        
        # Красивое имя документа
        def doc_name(source):
            if 'proect.pdf' in source:
                return "Договор"
            elif 'proect_pz.pdf' in source:
                return "Проектная документация"
            else:
                return "Смета"
        
        print(f"\n--- {link['link_type'].upper()} ---")
        print(f"  {doc_name(link['source_doc'])} (чанк {link['source_chunk_id']})")
        print(f"    → {doc_name(link['target_doc'])} (чанк {link['target_chunk_id']})")
        print(f"  Сходство: {link['similarity']:.1%}")
        print(f"  Текст источника: {link['source_text_preview'][:100]}...")

print("\n" + "="*70)
print("✅ ГРАФ СЕМАНТИЧЕСКИХ СВЯЗЕЙ ПОСТРОЕН!")
print("="*70)
print("\n📌 Следующий шаг:")
print("   python analyze_risks.py")