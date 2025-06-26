import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned2"
CHUNKED_DIR = BASE_DIR / "data" / "chunks"
CHUNKED_DIR.mkdir(parents=True, exist_ok=True)

# Configurar el splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,         # caracteres (aprox. 500 tokens)
    chunk_overlap=150,       # solapamiento para mantener contexto
    separators=["\n\n", "\n", ".", " "]  # fragmenta primero por párrafos
)

for filename in os.listdir(CLEANED_DIR):
    if filename.endswith(".txt"):
        input_path = os.path.join(CLEANED_DIR, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Fragmentar en chunks
        chunks = text_splitter.split_text(full_text)

        # Guardar chunks en un archivo (opcionalmente puedes guardar en JSONL)
        base = os.path.splitext(filename)[0]
        output_path = os.path.join(CHUNKED_DIR, f"{base}_chunks.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"[CHUNK {i+1}]\n{chunk}\n\n")

        print(f"[OK] {len(chunks)} chunks generados para {filename}")
