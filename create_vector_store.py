"""Create vector store from Sentence-BERT embeddings."""
import pickle
import chromadb
from chromadb.config import Settings
from pathlib import Path

print("="*60)
print("🗄️ Создание векторного хранилища (Sentence-BERT)")
print("="*60)

# Load data
print("\n📂 Загрузка данных...")
data_file = Path("data/chunks_and_embeddings_clean.pkl")

if not data_file.exists():
    print(f"❌ Файл {data_file} не найден!")
    print("   Сначала запустите: python create_embeddings_sbert.py")
    exit(1)

with open(data_file, 'rb') as f:
    data = pickle.load(f)

chunks = data['chunks']
embeddings = data['embeddings']
method = data.get('method', 'unknown')

print(f"✓ Загружено {len(chunks)} чанков")
print(f"✓ Метод: {method}")
print(f"✓ Размерность эмбеддингов: {embeddings.shape[1]}")

# Create vector store
print("\n🗄️ Создание векторного хранилища...")

# Удаляем старую папку если есть
chroma_dir = Path("data/chroma_db_clean")
if chroma_dir.exists():
    import shutil
    shutil.rmtree(chroma_dir)
    print("  Удалена старая коллекция")

chroma_dir.mkdir(exist_ok=True)

client = chromadb.PersistentClient(
    path=str(chroma_dir),
    settings=Settings(anonymized_telemetry=False)
)

# Create new collection
collection = client.create_collection(
    name="project_documents_clean",
    metadata={"description": "Project documents with Sentence-BERT embeddings"}
)
print("  Создана новая коллекция")

# Add documents in batches
batch_size = 100
print(f"\n📝 Добавление {len(chunks)} документов...")

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    batch_emb = embeddings[i:i+batch_size]
    
    collection.add(
        ids=[c['id'] for c in batch],
        documents=[c['content'] for c in batch],
        metadatas=[{'source': c['source'], 'chunk_id': c['chunk_id']} for c in batch],
        embeddings=[e.tolist() for e in batch_emb]
    )
    
    print(f"  {min(i+batch_size, len(chunks))}/{len(chunks)}")

print("\n✅ Готово!")
print(f"  Коллекция: {collection.name}")
print(f"  Документов: {collection.count()}")

# Test search
print("\n🔍 Тестирование поиска...")
test_queries = [
    "технические требования",
    "электромагнитная совместимость",
    "модернизация"
]

for query in test_queries:
    print(f"\n--- Запрос: '{query}' ---")
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 / (1 + distance)
        
        # Красивое имя документа
        doc_name = metadata['source']
        if doc_name == 'proect.pdf':
            doc_display = "📐 Проектная"
        elif doc_name == 'proect_pz.pdf':
            doc_display = "📝 Пояснительная"
        else:
            doc_display = "💰 Сметная"
        
        print(f"   {i}. {doc_display} (релевантность: {similarity:.1%})")
        print(f"      {doc[:100]}...")

print("\n" + "="*60)
print("✅ Векторное хранилище готово!")
print("="*60)
print("\n📌 Следующий шаг:")
print("   streamlit run app.py")