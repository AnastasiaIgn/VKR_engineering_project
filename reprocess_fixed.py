"""Reprocess documents with improved chunking."""
import pickle
import re
from pathlib import Path
from pypdf import PdfReader
from docx import Document

print("="*60)
print("🔄 Переобработка документов (исправленная версия)")
print("="*60)

# 1. Загрузка документов 
print("\n📂 Загрузка документов...")

docs_dir = Path("docs")
documents = []

for file_path in docs_dir.iterdir():
    if file_path.suffix.lower() == '.pdf':
        print(f"  Чтение: {file_path.name}")
        
        # Загружаем PDF
        text_parts = []
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        content = '\n'.join(text_parts)
        
        # Мягкая очистка 
        content = re.sub(r'https?://[^\s]+', '', content)
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', content)
        
        # Удаляем строки-мусор (короткие строки с сайтами)
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Пропускаем явный мусор
            if 'http://' in line or 'https://' in line:
                continue
            if 'www.' in line:
                continue
            if '@' in line and '.' in line:
                continue
            # Пропускаем слишком короткие строки (менее 10 символов)
            if len(line) < 10:
                continue
            # Пропускаем строки только с цифрами
            if re.match(r'^\d+$', line):
                continue
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
        
        documents.append({
            'source': file_path.name,
            'content': content,
            'char_count': len(content)
        })
        
        print(f"    Размер: {len(content):,} символов")

print(f"\n✓ Загружено {len(documents)} документов")

# 2. Создание чанков
print("\n✂️ Создание чанков...")

def chunk_text(text, source, chunk_size=1500, chunk_overlap=300):
    """Разбиение текста на чанки по абзацам."""
    # Разбиваем по абзацам (двойной перенос строки)
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    chunk_id = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Если абзац слишком длинный, разбиваем его на части
        if len(para) > chunk_size:
            # Разбиваем длинный абзац на предложения
            sentences = re.split(r'([.!?])\s+', para)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for i in range(0, len(sentences), 2):
                sentence_part = sentences[i]
                if i+1 < len(sentences):
                    sentence_part += sentences[i+1]
                
                if len(current_chunk) + len(sentence_part) > chunk_size and current_chunk:
                    chunks.append({
                        'id': f"{source}_{chunk_id}",
                        'content': current_chunk,
                        'source': source,
                        'chunk_id': chunk_id
                    })
                    chunk_id += 1
                    # Перекрытие
                    words = current_chunk.split()
                    overlap_words = words[-50:] if len(words) > 50 else words
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence_part
                else:
                    if current_chunk:
                        current_chunk += ' ' + sentence_part
                    else:
                        current_chunk = sentence_part
        else:
            # Обычный абзац
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    'id': f"{source}_{chunk_id}",
                    'content': current_chunk,
                    'source': source,
                    'chunk_id': chunk_id
                })
                chunk_id += 1
                # Перекрытие
                words = current_chunk.split()
                overlap_words = words[-50:] if len(words) > 50 else words
                current_chunk = ' '.join(overlap_words) + '\n\n' + para
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para
    
    # Добавляем последний чанк
    if current_chunk:
        chunks.append({
            'id': f"{source}_{chunk_id}",
            'content': current_chunk,
            'source': source,
            'chunk_id': chunk_id
        })
    
    return chunks

all_chunks = []
for doc in documents:
    chunks = chunk_text(doc['content'], doc['source'])
    all_chunks.extend(chunks)
    print(f"  - {doc['source']}: {len(chunks)} чанков")

print(f"\n✓ Всего чанков: {len(all_chunks)}")

# 3. Сохранение данных (СОЗДАЕМ ПАПКУ ЕСЛИ НЕТ)
print("\n💾 Сохранение данных...")

# СОЗДАЕМ ПАПКУ data ЕСЛИ ЕЁ НЕТ
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

data_file = data_dir / "chunks_clean.pkl"
with open(data_file, 'wb') as f:
    pickle.dump({'chunks': all_chunks}, f)
print(f"✓ Сохранено в {data_file}")

# 4. Показать примеры
print("\n📝 Примеры чанков:")
for i, chunk in enumerate(all_chunks[:3]):
    print(f"\n--- Чанк {i+1} (из {chunk['source']}) ---")
    print(chunk['content'][:300])
    print("...")

print("\n" + "="*60)
print("✅ Переобработка завершена!")
print(f"   Всего чанков: {len(all_chunks)}")
print("="*60)

# 5. Теперь создаем эмбеддинги
print("\n🤖 Теперь создаем эмбеддинги...")
print("Запустите команду:")
print('python create_embeddings_sbert.py')
