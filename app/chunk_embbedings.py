from sentence_transformers import SentenceTransformer
import faiss
import os
import json
import numpy as np
from pathlib import Path

# === Cargar modelo embedding ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Ligero y preciso

# === Configuración ===
BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKED_DIR = BASE_DIR / "data" / "chunks"
INDEX_DIR = BASE_DIR / "embeddings"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# === Preparar los datos ===
documents = []
metadatas = []
for filename in os.listdir(CHUNKED_DIR):
    if filename.endswith(".txt"):
        
        filepath = os.path.join(CHUNKED_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().split("[CHUNK")  # asumiendo formato del script anterior
            for chunk in content:
                text = chunk.strip()
                if len(text) > 30:  # filtrar ruido
                    documents.append(text)
                    metadatas.append({"source": filename})

# === Vectorizar con el modelo ===
embeddings = model.encode(documents, show_progress_bar=True)

# === Crear índice FAISS ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # índice simple y eficiente
index.add(np.array(embeddings).astype("float32"))

# === Guardar índice y metadatos ===
faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))

# Guardar los metadatos asociados
with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadatas, f, indent=2)

print(f"✅ Indexado {len(documents)} chunks con éxito.")
