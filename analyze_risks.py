"""
Analyze project risks and inconsistencies using semantic links.
For Master's thesis - demonstrates context analysis method.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

print("="*70)
print("⚠️ КОНТЕКСТНЫЙ АНАЛИЗ РИСКОВ И НЕСООТВЕТСТВИЙ")
print("="*70)

# 1. Загрузка графа связей
print("\n📂 Шаг 1: Загрузка графа семантических связей...")

links_file = Path("data/semantic_links.json")
if not links_file.exists():
    print(f"❌ Файл {links_file} не найден!")
    print("   Сначала запустите: python extract_semantic_links.py")
    exit(1)

with open(links_file, 'r', encoding='utf-8') as f:
    graph = json.load(f)

nodes = graph['nodes']
edges = graph['edges']
params = graph['parameters']

print(f"✓ Загружено {len(nodes)} узлов (чанков)")
print(f"✓ Загружено {len(edges)} связей")
print(f"  K={params['K']}, порог={params['similarity_threshold']}")

# 2. Выявление несоответствий (contradicts)
print("\n🔍 Шаг 2: Выявление несоответствий...")

contradictions = []
for edge in edges:
    if edge['link_type'] == 'contradicts':
        contradictions.append(edge)

print(f"✓ Выявлено {len(contradictions)} несоответствий")

# 3. Классификация рисков
print("\n⚠️ Шаг 3: Классификация рисков...")

# Ключевые слова для разных типов рисков
risk_keywords = {
    'financial': [
        'стоимость', 'цена', 'оплата', 'завышение', 'убыток', 'штраф',
        'пеня', 'неустойка', 'бюджет', 'смета', 'расценка', 'объем'
    ],
    'schedule': [
        'срок', 'задержка', 'просрочка', 'график', 'этап', 'срыв',
        'перенос', 'отставание', 'продление'
    ],
    'quality': [
        'качество', 'дефект', 'брак', 'не соответствует', 'недостаток',
        'отклонение', 'несоответствие'
    ],
    'legal': [
        'договор', 'соглашение', 'нарушение', 'претензия', 'иск',
        'суд', 'санкция', 'контролирующий орган'
    ]
}

risks = []

for contradiction in contradictions:
    text_source = contradiction['source_text_preview']
    text_target = contradiction['target_text_preview']
    combined_text = (text_source + ' ' + text_target).lower()
    
    # Определяем тип риска
    risk_types = []
    for risk_type, keywords in risk_keywords.items():
        for keyword in keywords:
            if keyword in combined_text:
                risk_types.append(risk_type)
                break
    
    if not risk_types:
        risk_types = ['operational']  # общий операционный риск
    
    # Оценка серьезности (на основе эвристик)
    severity = 'medium'
    high_risk_keywords = ['завышение', 'штраф', 'суд', 'претензия', 'срыв']
    for keyword in high_risk_keywords:
        if keyword in combined_text:
            severity = 'high'
            break
    
    # Извлечение числовых значений (если есть)
    numbers = re.findall(r'(\d+(?:[.,]\d+)?)\s*(м3|м2|шт|т|км|руб|тыс|млн)', combined_text)
    
    risks.append({
        'contradiction_id': f"R_{len(risks)+1}",
        'source_doc': contradiction['source_doc'],
        'target_doc': contradiction['target_doc'],
        'source_text': text_source[:200],
        'target_text': text_target[:200],
        'risk_types': risk_types,
        'severity': severity,
        'extracted_numbers': numbers[:3] if numbers else [],
        'similarity': contradiction['similarity']
    })

print(f"✓ Классифицировано {len(risks)} рисков")

# Статистика по типам рисков
risk_stats = defaultdict(int)
severity_stats = defaultdict(int)

for risk in risks:
    for rt in risk['risk_types']:
        risk_stats[rt] += 1
    severity_stats[risk['severity']] += 1

print("\n📊 Распределение по типам рисков:")
for risk_type, count in sorted(risk_stats.items(), key=lambda x: -x[1]):
    print(f"   {risk_type}: {count}")

print("\n📊 Распределение по серьезности:")
for severity, count in sorted(severity_stats.items()):
    print(f"   {severity}: {count}")

# 4. Формирование отчета
print("\n📄 Шаг 4: Формирование отчета...")

report = {
    'generated_at': datetime.now().isoformat(),
    'summary': {
        'total_links': len(edges),
        'total_contradictions': len(contradictions),
        'total_risks': len(risks),
        'high_severity_risks': severity_stats.get('high', 0),
        'risk_by_type': dict(risk_stats)
    },
    'risks': risks,
    'recommendations': []
}

# Генерация рекомендаций
if severity_stats.get('high', 0) > 0:
    report['recommendations'].append({
        'priority': 'high',
        'text': 'Выявлены несоответствия высокой серьезности. Рекомендуется провести детальную проверку документов до подписания актов приемки.'
    })

if risk_stats.get('financial', 0) > 0:
    report['recommendations'].append({
        'priority': 'medium',
        'text': 'Обнаружены потенциальные финансовые риски. Проверьте соответствие объемов работ и расценок утвержденной смете.'
    })

if risk_stats.get('legal', 0) > 0:
    report['recommendations'].append({
        'priority': 'medium',
        'text': 'Выявлены возможные договорные риски. Убедитесь в наличии всех необходимых дополнительных соглашений.'
    })

if len(risks) == 0:
    report['recommendations'].append({
        'priority': 'low',
        'text': 'Значимых несоответствий не обнаружено. Рекомендуется продолжить мониторинг при поступлении новых документов.'
    })

# Сохранение отчета
output_file = Path("data/risk_report.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"✓ Отчет сохранен в {output_file}")
print(f"   Размер файла: {output_file.stat().st_size / 1024:.1f} KB")

# 5. Вывод отчета на экран
print("\n" + "="*70)
print("📋 ОТЧЕТ ПО РИСКАМ И НЕСООТВЕТСТВИЯМ")
print("="*70)

print(f"\n📊 Сводка:")
print(f"   Всего связей: {report['summary']['total_links']}")
print(f"   Несоответствий: {report['summary']['total_contradictions']}")
print(f"   Выявлено рисков: {report['summary']['total_risks']}")
print(f"   Рисков высокой серьезности: {report['summary']['high_severity_risks']}")

print("\n⚠️ Детализация рисков:")
for i, risk in enumerate(risks[:5], 1):  # показываем первые 5
    print(f"\n   Риск {i} (серьезность: {risk['severity']})")
    print(f"     Документы: {risk['source_doc']} ↔ {risk['target_doc']}")
    print(f"     Типы: {', '.join(risk['risk_types'])}")
    if risk['extracted_numbers']:
        print(f"     Числовые значения: {risk['extracted_numbers']}")
    print(f"     Фрагмент: {risk['source_text'][:80]}...")

print("\n💡 Рекомендации:")
for rec in report['recommendations']:
    print(f"   [{rec['priority'].upper()}] {rec['text']}")

print("\n" + "="*70)
print("✅ АНАЛИЗ РИСКОВ ЗАВЕРШЕН!")
print("="*70)
print("\n📌 Результаты сохранены в:")
print("   - data/semantic_links.json (граф связей)")
print("   - data/risk_report.json (отчет по рискам)")