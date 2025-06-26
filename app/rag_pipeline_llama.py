import os
import json
import faiss
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
import time
from typing import List, Dict, Tuple

# ------------------ CONFIGURACIÓN ------------------

# Directorio donde están los archivos .txt con chunks avanzados
CHUNKS_DIR = Path("../data/chunks_advanced")
# Directorio donde se guardan índice y metadata
INDEX_DIR = Path("../embeddings")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
# Rutas para archivos persistentes - NUEVOS NOMBRES
INDEX_PATH = INDEX_DIR / "faiss_new.index"
METADATA_PATH = INDEX_DIR / "metadata_new.json"
# Modelos
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = "llama3.2:3b"  # Cambiado de gemma:2b a llama3.2:3b
SYNONYM_MODEL_NAME = "gemma:2b"  # Modelo ligero para sinónimos

# Instrucciones para el modelo principal
SYSTEM_INSTRUCTIONS = """Eres un administrativo universitario especializado en atender consultas de estudiantes. 
Tu rol es brindar información precisa, clara y útil con la calidez y profesionalismo que caracteriza al personal administrativo de una universidad.

Directrices importantes:
- Utiliza solo el Documento como fuente de información para responder la pregunta, pero si los documentos no estan relacionados a la pregunta responde que no tienes información necesaria
- Analiza si la pregunta tiene que ver con aspectos universitarios si no es asi entonces responde con "La pregunta esta fuera de contexto" y no menciones que no encontraste información relevante.
- No menciones actividades que hagas como segun los documentos...
- No des un resumen final, es decir no digas en resumen...
- No hagas preguntas para indagar mas en el tema, si no tienes el contexto necesario entonces responde que se comunique con las autoridades correspondientes
- Solo responde en base a los documentos no busques interactuar mas
- Si son mensajes cortos como gracias y cosas asi solo responde que estamos para ayudarla y no tomes en cuenta los documentos
- No respondas en tercera persona, responde en primera persona como nosotros (universidad)
- Analiza TODA la información proporcionada en el contexto antes de responder
- Integra la información de todos los documentos para dar UNA respuesta coherente y completa
- Responde de manera amigable, profesional y orientada al estudiante
- Si encuentras información relevante en CUALQUIER parte del contexto, úsala para construir una respuesta integral
- Solo si NINGÚN documento contiene información relevante para una respuesta coherente, responde: "Lo siento, tu consulta no es aceptada. Te recomiendo acercarte a la oficina correspondiente para obtener información más detallada y actualizada."
- Utiliza un tono cálido pero profesional
- Si hay procesos o requisitos, explícalos paso a paso de manera clara
- Sintetiza la información de múltiples fuentes en una respuesta unificada
- Evita repetir la misma información si aparece en varios documentos
- Siempre prioriza la precisión y completitud de la información"""

# ------------------ FUNCIONES DE PIPELINE ------------------

def load_all_chunks():
    """Recorre CHUNKS_DIR y devuelve lista de textos y metadatos con contexto expandido."""
    texts, metadatas = [], []
    document_chunks = {}  # Para almacenar chunks por documento
    
    for file in CHUNKS_DIR.glob("*.txt"):
        content = file.read_text(encoding="utf-8")
        # Asumimos que cada chunk está separado por etiqueta [CHUNK X]
        parts = [p.strip() for p in content.split("[CHUNK") if p.strip()]
        doc_chunks = []
        
        for i, part in enumerate(parts):
            # Extraer número de chunk y contenido
            chunk_match = re.match(r'(\d+)\](.*)', part, re.DOTALL)
            if chunk_match:
                chunk_num = int(chunk_match.group(1))
                chunk_content = chunk_match.group(2).strip()
            else:
                chunk_content = part.split(']', 1)[-1].strip() if ']' in part else part.strip()
                chunk_num = i
            
            if len(chunk_content) < 30:
                continue
                
            doc_chunks.append({
                'content': chunk_content,
                'chunk_num': chunk_num,
                'index': len(texts)
            })
            
            texts.append(chunk_content)
            metadatas.append({
                "source": file.name,
                "chunk_id": chunk_num,
                "document_index": len(texts) - 1
            })
        
        document_chunks[file.name] = doc_chunks
    
    return texts, metadatas, document_chunks

def generate_synonyms(text: str, synonym_llm) -> List[str]:
    """Genera sinónimos para las palabras clave de la consulta."""
    # Extraer palabras clave (eliminar stopwords básicas)
    stopwords = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'las', 'los', 'una', 'como', 'pero', 'sus', 'le', 'ya', 'o', 'porque', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'donde', 'quien', 'desde', 'todos', 'durante', 'todo', 'puede', 'más'}
    
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if len(w) > 3 and w not in stopwords]
    
    if not keywords:
        return [text]
    
    # Generar sinónimos de forma optimizada
    keywords_str = ", ".join(keywords[:5])  # Limitar a 5 palabras clave
    
    synonym_prompt = f"""Para las siguientes palabras clave relacionadas con trámites universitarios, genera únicamente sinónimos comunes separados por comas (máximo 3 sinónimos por palabra):

Palabras: {keywords_str}

Responde solo con sinónimos en este formato: palabra1: sinónimo1, sinónimo2; palabra2: sinónimo1, sinónimo2"""

    try:
        response = synonym_llm.generate([synonym_prompt])
        synonym_text = response.generations[0][0].text.strip()
        
        # Extraer sinónimos y crear consultas expandidas
        expanded_terms = [text]  # Consulta original
        for line in synonym_text.split(';'):
            if ':' in line:
                synonyms = line.split(':')[1].strip()
                expanded_terms.extend([s.strip() for s in synonyms.split(',') if s.strip()])
        
        return expanded_terms[:10]  # Limitar para optimizar búsqueda
    except Exception as e:
        print(f"Error generando sinónimos: {e}")
        return [text]

def get_expanded_context(chunk_idx: int, metadata: List[Dict], texts: List[str], document_chunks: Dict) -> str:
    """Obtiene el contexto expandido con lógica mejorada para chunks extremos."""
    current_meta = metadata[chunk_idx]
    source = current_meta['source']
    current_chunk_id = current_meta['chunk_id']
    
    if source not in document_chunks:
        return texts[chunk_idx]
    
    doc_chunks = document_chunks[source]
    expanded_content = []
    
    # Buscar posición del chunk actual
    current_pos = -1
    for i, chunk_info in enumerate(doc_chunks):
        if chunk_info['chunk_num'] == current_chunk_id:
            current_pos = i
            break
    
    if current_pos == -1:
        return texts[chunk_idx]
    
    total_chunks = len(doc_chunks)
    
    # Lógica mejorada para chunks extremos
    if current_pos == 0:  # Primer chunk: tomar los dos siguientes
        chunks_to_include = [current_pos]
        if total_chunks > 1:
            chunks_to_include.append(current_pos + 1)
        if total_chunks > 2:
            chunks_to_include.append(current_pos + 2)
    elif current_pos == total_chunks - 1:  # Último chunk: tomar los dos anteriores
        chunks_to_include = []
        if total_chunks > 2:
            chunks_to_include.append(current_pos - 2)
        if total_chunks > 1:
            chunks_to_include.append(current_pos - 1)
        chunks_to_include.append(current_pos)
    else:  # Chunk intermedio: anterior, actual, posterior
        chunks_to_include = [current_pos - 1, current_pos, current_pos + 1]
    
    # Construir contenido expandido
    for pos in chunks_to_include:
        if 0 <= pos < total_chunks:
            chunk_info = doc_chunks[pos]
            chunk_text = texts[chunk_info['index']]
            
            if pos < current_pos:
                expanded_content.append(f"[Contexto anterior]\n{chunk_text}")
            elif pos == current_pos:
                expanded_content.append(f"[Información principal]\n{chunk_text}")
            else:
                expanded_content.append(f"[Contexto posterior]\n{chunk_text}")
    
    return "\n\n".join(expanded_content) if expanded_content else texts[chunk_idx]

def build_or_load_index():
    """Construye o carga índice FAISS y metadatos."""
    if INDEX_PATH.exists() and METADATA_PATH.exists():
        print("Cargando índice FAISS y metadata existentes...")
        index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        texts, _, document_chunks = load_all_chunks()
        return index, metadata, texts, document_chunks

    print("Construyendo nuevo índice FAISS...")
    texts, metadata, document_chunks = load_all_chunks()
    
    # Generar embeddings (CPU) - Optimizado
    print(f"Generando embeddings para {len(texts)} chunks...")
    embeddings = EMBEDDING_MODEL.encode(
        texts, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        batch_size=32  # Optimización para velocidad
    )
    
    # Crear índice L2 optimizado
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    
    # Guardar
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Índice construido con {index.ntotal} vectores de dimensión {dim}")
    return index, metadata, texts, document_chunks

def retrieve_top_k_with_synonyms(query: str, synonym_llm,k: int = 5, data: list = None) -> List[Dict]:
    if data:
        index = data[0]
        metadata = data[1]
        texts = data[2]
        document_chunks = data[3]


    """Búsqueda mejorada con sinónimos y contexto expandido."""
    start_time = time.time()
    
    # Generar sinónimos de forma optimizada
    print("🔍 Generando sinónimos...")
    expanded_queries = generate_synonyms(query, synonym_llm)
    
    # Realizar múltiples búsquedas y combinar resultados
    all_results = {}
    
    for expanded_query in expanded_queries:
        query_vec = EMBEDDING_MODEL.encode([expanded_query], convert_to_numpy=True)
        distances, indices = index.search(query_vec.astype('float32'), k * 3)  # Aumentamos el factor de búsqueda
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx not in all_results or dist < all_results[idx]['distance']:
                all_results[idx] = {
                    "index": idx,
                    "distance": float(dist),
                    "query_used": expanded_query
                }
    
    # Seleccionar los mejores k resultados únicos
    sorted_results = sorted(all_results.values(), key=lambda x: x['distance'])[:k]
    
    # Obtener contexto expandido para cada resultado
    final_results = []
    processed_sources = set()  # Para evitar duplicar información del mismo documento
    
    for result in sorted_results:
        idx = result['index']
        source = metadata[idx]['source']
        
        # Evitar chunks muy similares del mismo documento
        source_key = f"{source}_{metadata[idx]['chunk_id']}"
        if source_key in processed_sources:
            continue
            
        expanded_text = get_expanded_context(idx, metadata, texts, document_chunks)
        processed_sources.add(source_key)
        
        final_results.append({
            "text": expanded_text,
            "original_text": texts[idx],
            "metadata": metadata[idx],
            "distance": result['distance'],
            "query_used": result['query_used']
        })
    
    search_time = time.time() - start_time
    print(f"⚡ Búsqueda completada en {search_time:.2f}s")
    
    return final_results

# ------------------ MAIN ------------------

if __name__ == "__main__":
    print("🚀 Iniciando Sistema RAG Avanzado...")
    
    # 1. Construir o cargar índice y metadata
    index, metadata, texts, document_chunks = build_or_load_index()

    # 2. Configurar LLMs
    print("🤖 Cargando modelos LLM...")
    llm = OllamaLLM(model=LLM_MODEL_NAME)
    synonym_llm = OllamaLLM(model=SYNONYM_MODEL_NAME)

    print("\n✅ Sistema RAG Universitario listo. Escribe 'salir' para terminar.")
    print("📚 Funcionalidades: Búsqueda con sinónimos, contexto expandido, asistente administrativo")
    
    while True:
        query = input("\n🎓 Consulta del estudiante: ")
        if query.lower() in ('salir', 'exit', 'quit'):
            print("👋 ¡Hasta luego! Que tengas un buen día.")
            break
            
        start_time = time.time()
        
        # 3. Recuperar contexto con sinónimos y expansión
        top_chunks = retrieve_top_k_with_synonyms(query, synonym_llm, k=3)
        
        if not top_chunks:
            print("\n🤖 Respuesta:\nLo siento, no encontré información específica sobre tu consulta en la base de datos. Te recomiendo acercarte a la oficina correspondiente para obtener información más detallada y actualizada.")
            continue
        
        context = "\n\n---DOCUMENTO---\n\n".join([c['text'] for c in top_chunks])
        
        # 4. Construir prompt con instrucciones del sistema
        full_prompt = f"""{SYSTEM_INSTRUCTIONS}

CONTEXTO DISPONIBLE:
{context}

CONSULTA DEL ESTUDIANTE: {query}

RESPUESTA:"""
        
        # 5. Generar respuesta
        print("💭 Generando respuesta...")
        response = llm.generate([full_prompt])
        answer = response.generations[0][0].text.strip()
        
        total_time = time.time() - start_time
        
        print(f"\n🤖 Respuesta del Asistente Administrativo:\n{answer}")
        
        # 6. Mostrar fuentes consultadas
        print(f"\n📄 Fuentes consultadas ({len(top_chunks)} documentos):")
        for i, c in enumerate(top_chunks, 1):
            print(f"{i}. {c['metadata']['source']} (chunk {c['metadata']['chunk_id']}) - Similitud: {c['distance']:.3f}")
            if c['query_used'] != query:
                print(f"   🔍 Encontrado con: '{c['query_used']}'")
        
        print(f"\n⏱️ Tiempo total de respuesta: {total_time:.2f}s")