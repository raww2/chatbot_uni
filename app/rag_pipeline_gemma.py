import os
import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM  # usa la clase correcta desde langchain-ollama

# ------------------ CONFIGURACIÓN ------------------

# Directorio donde están los archivos .txt con chunks (uno por documento)
CHUNKS_DIR = Path("../data/chunks")
# Directorio donde se guardan índice y metadata
INDEX_DIR = Path("../embeddings")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
# Rutas para archivos persistentes
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
# Modelos
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = "gemma:2b"

# ------------------ FUNCIONES DE PIPELINE ------------------

def load_all_chunks():
    """Recorre CHUNKS_DIR y devuelve lista de textos y metadatos."""
    texts, metadatas = [], []
    for file in CHUNKS_DIR.glob("*.txt"):
        content = file.read_text(encoding="utf-8")
        # Asumimos que cada chunk está separado por etiqueta [CHUNK X]
        parts = [p.strip() for p in content.split("[CHUNK") if p.strip()]
        for i, part in enumerate(parts):
            # Part elimina encabezado si existe número
            chunk = part.split(']', 1)[-1].strip()
            if len(chunk) < 30:
                continue
            texts.append(chunk)
            metadatas.append({
                "source": file.name,
                "chunk_id": i
            })
    return texts, metadatas


def build_or_load_index():
    """Construye o carga índice FAISS y metadatos."""
    if INDEX_PATH.exists() and METADATA_PATH.exists():
        print("Cargando índice FAISS y metadata existentes...")
        index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        texts, _ = load_all_chunks()  # solo para longitud
        return index, metadata, texts

    print("Construyendo nuevo índice FAISS...")
    texts, metadata = load_all_chunks()
    # Generar embeddings (CPU)
    embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Crear índice L2
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    # Guardar
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    return index, metadata, texts


def retrieve_top_k(query: str, k: int = 3):
    """Dado un string de consulta, devuelve los k chunks más similares."""
    query_vec = EMBEDDING_MODEL.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec.astype('float32'), k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": texts[idx],
            "metadata": metadata[idx],
            "distance": float(dist)
        })
    return results

# ------------------ MAIN ------------------

# if __name__ == "__main__":
#     # 1. Construir o cargar índice y metadata
#     index, metadata, texts = build_or_load_index()

#     # 2. Configurar LLM Ollama
#     print("Cargando LLM en Ollama...")
#     llm = OllamaLLM(model=LLM_MODEL_NAME)  # instancia LLM usando OllamaLLM

#     print("Sistema RAG listo. Escribe 'salir' para terminar.")
#     while True:
#         query = input("\n🗨️ Pregunta: ")
#         if query.lower() in ('salir', 'exit', 'quit'):
#             break
#         # 3. Recuperar contexto
#         top_chunks = retrieve_top_k(query, k=3)
#         context = "\n---\n".join([c['text'] for c in top_chunks])
#         prompt = f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"  
#         # 4. Generar respuesta
#         # Generar respuesta usando el método generate
#         gen = llm.generate([prompt])
#         response = gen.generations[0][0].text
#         print("\n🤖 Respuesta:\n", response)

#         # 5. Mostrar fuentes
#         print("\n📄 Fuentes consultadas:")
#         for c in top_chunks:
#             print(f"- {c['metadata']['source']} (chunk {c['metadata']['chunk_id']}, dist={c['distance']:.2f})")
