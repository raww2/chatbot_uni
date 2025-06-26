from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

# ---------------- Configuration ----------------

# Directorio donde están los archivos .txt con chunks (uno por documento)
CHUNKS_DIR = Path("../data/chunks")
# Directorio donde se guardan índice y metadata
INDEX_DIR = Path("../embeddings")

INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemma:2b"

# ---------------- Data Models ----------------
class QueryRequest(BaseModel):
    query: str = Field(..., example="¿Como solicito una ampliación de creditos?")
    top_k: int = Field(3, ge=1, le=10, description="Número de chunks a recuperar")

class SourceDoc(BaseModel):
    source: str
    chunk_id: int
    distance: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]

# ---------------- App Initialization ----------------
app = FastAPI(title="RAG API", version="1.0")

# Global variables to hold models and data
index: faiss.Index = None
metadata: List[Dict[str, Any]] = []
texts: List[str] = []
embedder: SentenceTransformer = None
llm: OllamaLLM = None

@app.on_event("startup")
def load_resources():
    global index, metadata, texts, embedder, llm
    # Load or build FAISS index and metadata
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("Índice o metadata no encontrados. Genera primero el índice FAISS.")
    # Load index
    index = faiss.read_index(str(INDEX_PATH))
    # Load metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    # Load texts list for retrieval use (only distances and indices)
    # We assume texts loaded same order as metadata
    # For context building we need raw texts, not stored
    # So load all chunks from CHUNKS_DIR
    texts = []
    for file in CHUNKS_DIR.glob("*.txt"):
        content = file.read_text(encoding="utf-8")
        parts = [p.strip() for p in content.split("[CHUNK") if p.strip()]
        for part in parts:
            texts.append(part.split(']', 1)[-1].strip())
    # Initialize embedder
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Initialize Ollama LLM
    llm = OllamaLLM(model=LLM_MODEL_NAME)

# ---------------- Utility ----------------
def retrieve_top_k(query: str, k: int):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_vec.astype('float32'), k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        md = metadata[idx]
        results.append(SourceDoc(
            source=md.get("source", ""),
            chunk_id=md.get("chunk_id", -1),
            distance=float(dist)
        ))
    return results

# ---------------- API Endpoints ----------------
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    # Validate non-empty
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")
    # Retrieve top-k
    sources = retrieve_top_k(request.query, request.top_k)
    # Build context
    context_texts = []
    for src in sources:
        idx = src.chunk_id  # metadata idx corresponds to text index
        context_texts.append(texts[idx])
    context = "\n---\n".join(context_texts)
    prompt = f"Contexto:\n{context}\n\nPregunta: {request.query}\nRespuesta:"  
    # Generate answer
    try:
        gen = llm.generate([prompt])
        answer = gen.generations[0][0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en LLM: {str(e)}")
    return QueryResponse(answer=answer, sources=sources)

# ---------------- Health Check ----------------
@app.get("/health")
def health_check():
    return {"status": "ok"}