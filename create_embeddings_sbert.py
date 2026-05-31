"""Create embeddings with Sentence-BERT """
import pickle
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🤖 СОЗДАНИЕ ЭМБЕДДИНГОВ (FIXED VERSION)")
print("="*70)

# Force import numpy first
try:
    import numpy as np
    print(f"✓ NumPy версия: {np.__version__}")
except ImportError:
    print("❌ NumPy не установлен!")
    print("   Выполните: pip install numpy==1.24.3")
    exit(1)

# Load chunks
print("\n📂 Шаг 1: Загрузка чанков...")
data_file = Path("data/chunks_clean.pkl")

if not data_file.exists():
    print(f"❌ Файл {data_file} не найден!")
    print("   Сначала запустите: python reprocess_fixed.py")
    exit(1)

with open(data_file, 'rb') as f:
    data = pickle.load(f)

chunks = data['chunks']
texts = [c['content'] for c in chunks]
print(f"✓ Загружено {len(chunks)} чанков")

# Load Sentence-BERT
print("\n🤖 Шаг 2: Загрузка модели Sentence-BERT...")
print("   Модель: all-MiniLM-L6-v2")

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f"✓ Модель загружена")
    print(f"   Размерность: {model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    exit(1)

# Create embeddings with explicit numpy conversion
print(f"\n🔢 Шаг 3: Создание эмбеддингов для {len(texts)} чанков...")
print("   Это может занять 5-10 минут...")

start_time = time.time()

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True  
)

# Убеждаемся, что это numpy array
if not isinstance(embeddings, np.ndarray):
    embeddings = np.array(embeddings)

encode_time = time.time() - start_time

print(f"\n✓ Эмбеддинги созданы за {encode_time:.1f} сек")
print(f"   Формат: {embeddings.shape}")
print(f"   Тип: {type(embeddings)}")
print(f"   Объем памяти: {embeddings.nbytes / 1024 / 1024:.2f} MB")

# Save
print("\n💾 Шаг 4: Сохранение эмбеддингов...")
output_file = Path("data/chunks_and_embeddings_clean.pkl")
with open(output_file, 'wb') as f:
    pickle.dump({
        'chunks': chunks,
        'embeddings': embeddings,
        'model_name': 'all-MiniLM-L6-v2',
        'method': 'sentence-bert',
        'embedding_dim': embeddings.shape[1]
    }, f)

file_size = output_file.stat().st_size / 1024 / 1024
print(f"✓ Сохранено в {output_file}")
print(f"   Размер файла: {file_size:.2f} MB")

# Quick test
print("\n🔍 Шаг 5: Быстрый тест поиска...")
test_query = "технические требования"
query_emb = model.encode([test_query], convert_to_numpy=True)

# Cosine similarity
from numpy.linalg import norm
similarities = (query_emb @ embeddings.T) / (norm(query_emb) * norm(embeddings, axis=1))
similarities = similarities[0]

top_indices = similarities.argsort()[-3:][::-1]
print(f"\nРезультаты по запросу '{test_query}':")
for idx in top_indices:
    chunk = chunks[idx]
    score = similarities[idx]
    doc_name = '📐 Проектная' if chunk['source'] == 'proect.pdf' else '📝 Пояснительная' if chunk['source'] == 'proect_pz.pdf' else '💰 Сметная'
    print(f"   {doc_name} (релевантность: {score:.1%})")

print("\n" + "="*70)
print("✅ ЭМБЕДДИНГИ УСПЕШНО СОЗДАНЫ!")
print("="*70)
print("\n📌 Следующий шаг:")
print("   python create_vector_store.py")
