import os
import requests
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

class ChunkValidator:
    def __init__(self, model_name="llama3.2:3b", ollama_url="http://localhost:11434"):
        """
        Inicializa el validador de chunks
        
        Args:
            model_name: Nombre del modelo en Ollama (recomendado: llama3.2:3b)
            ollama_url: URL del servidor Ollama
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/generate"
        
        # Verificar que Ollama esté corriendo y el modelo disponible
        self._check_ollama_connection()
        self._ensure_model_available()
    
    def _check_ollama_connection(self):
        """Verifica que Ollama esté corriendo"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama no está respondiendo correctamente")
            print("[✓] Conexión con Ollama establecida")
        except Exception as e:
            raise Exception(f"Error conectando con Ollama: {e}. ¿Está Ollama corriendo?")
    
    def _ensure_model_available(self):
        """Verifica que el modelo esté disponible, si no lo descarga"""
        try:
            # Verificar modelos disponibles
            response = requests.get(f"{self.ollama_url}/api/tags")
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model_name not in model_names:
                print(f"[!] Modelo {self.model_name} no encontrado. Descargando...")
                self._download_model()
            else:
                print(f"[✓] Modelo {self.model_name} disponible")
        except Exception as e:
            raise Exception(f"Error verificando modelo: {e}")
    
    def _download_model(self):
        """Descarga el modelo si no está disponible"""
        print(f"[!] Modelo {self.model_name} no encontrado.")
        print(f"[!] Por favor, instálalo manualmente con:")
        print(f"    ollama pull {self.model_name}")
        print(f"[!] Luego ejecuta nuevamente este script.")
        
        user_input = input("\n¿Quieres que intente descargarlo automáticamente? (y/n): ").lower().strip()
        
        if user_input != 'y':
            raise Exception("Descarga cancelada. Instala el modelo manualmente.")
        
        try:
            print(f"[⬇] Descargando modelo {self.model_name}...")
            print("[⬇] Esto puede tomar 5-15 minutos dependiendo de tu conexión...")
            
            pull_data = {"name": self.model_name}
            response = requests.post(
                f"{self.ollama_url}/api/pull", 
                json=pull_data, 
                stream=True,
                timeout=1800  # 30 minutos timeout
            )
            
            if response.status_code == 200:
                last_status = ""
                progress_count = 0
                
                for line in response.iter_lines():
                    if line:
                        try:
                            status_data = json.loads(line.decode('utf-8'))
                            
                            if 'status' in status_data:
                                current_status = status_data['status']
                                
                                # Solo mostrar cambios de estado importantes
                                if current_status != last_status:
                                    print(f"[⬇] {current_status}")
                                    last_status = current_status
                                    progress_count = 0
                                else:
                                    progress_count += 1
                                    # Mostrar progreso cada 50 iteraciones del mismo estado
                                    if progress_count % 50 == 0:
                                        print(f"[⬇] {current_status} (continuando...)")
                                
                                # Si el status indica completado
                                if 'success' in current_status.lower() or 'complete' in current_status.lower():
                                    break
                                    
                        except json.JSONDecodeError:
                            continue
                            
                print(f"[✓] Modelo {self.model_name} descargado exitosamente")
            else:
                raise Exception(f"Error descargando modelo: {response.status_code}")
                
        except requests.exceptions.Timeout:
            raise Exception("Timeout descargando modelo. Intenta descargarlo manualmente con: ollama pull llama3.2:3b")
        except Exception as e:
            raise Exception(f"Error descargando modelo: {e}")
    
    def validate_and_clean_chunk(self, chunk_text):
        """
        Valida y limpia un chunk usando el LLM
        
        Args:
            chunk_text: Texto del chunk a validar
            
        Returns:
            tuple: (is_valid, cleaned_text)
        """
        
        # Prompt más estricto y directo
        system_prompt = """Eres un procesador de texto especializado. Tu única tarea es evaluar y limpiar fragmentos de texto extraídos de PDFs.

REGLAS ESTRICTAS:

1. EVALUACIÓN:
   - VÁLIDO: Texto con información coherente y útil (mínimo 15 palabras con sentido)
   - INVÁLIDO: Ruido, caracteres aleatorios, texto incomprensible

2. RESPUESTA OBLIGATORIA:
   - Si es VÁLIDO: Responde solo "VALID:" seguido del texto corregido en la misma línea
   - Si es INVÁLIDO: Responde solo "INVALID"

3. LIMPIEZA (solo texto válido):
   - Corrige errores de OCR
   - Elimina caracteres extraños
   - Mejora puntuación
   - Mantén el significado original
   - NO agregues explicaciones, comentarios o notas

EJEMPLOS:
Input: "Hola.. mundo !! como   estas???"
Output: VALID: Hola mundo, ¿cómo estás?

Input: "xdf##$%&dfdf888"
Output: INVALID

IMPORTANTE: Responde ÚNICAMENTE con "VALID:" + texto limpio O "INVALID". No agregues nada más."""

        user_prompt = f"Procesa este texto:\n\n{chunk_text}"
        
        try:
            # Preparar datos para la API de Ollama
            data = {
                "model": self.model_name,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Muy baja para respuestas consistentes
                    "top_p": 0.8,
                    "max_tokens": 1500,
                    "stop": ["\n\n", "Input:", "Output:", "Ejemplo:"]  # Detener en palabras clave
                }
            }
            
            # Hacer la solicitud al LLM
            response = requests.post(self.api_url, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '').strip()
                
                # Limpiar la respuesta de posibles prefijos no deseados
                llm_response = self._clean_llm_response(llm_response)
                
                # Procesar la respuesta del LLM
                if llm_response.upper().startswith('INVALID'):
                    return False, None
                elif llm_response.upper().startswith('VALID:'):
                    # Extraer solo el texto después de "VALID:"
                    cleaned_text = llm_response[6:].strip()  # Remover "VALID:"
                    
                    # Validación adicional del texto limpio
                    if len(cleaned_text.split()) < 5:  # Menos de 5 palabras
                        return False, None
                    
                    return True, cleaned_text
                else:
                    # Si no sigue el formato, intentar extraer texto útil
                    if len(llm_response.split()) >= 10:  # Al menos 10 palabras
                        # Limpiar posibles artefactos de la respuesta
                        cleaned = self._extract_useful_text(llm_response)
                        if cleaned:
                            return True, cleaned
                    
                    return False, None
            else:
                print(f"[!] Error en API de Ollama: {response.status_code}")
                return False, None  # En caso de error, descartar chunk
                
        except Exception as e:
            print(f"[!] Error validando chunk: {e}")
            return False, None  # En caso de error, descartar chunk
    
    def _clean_llm_response(self, response):
        """Limpia la respuesta del LLM de artefactos comunes"""
        # Remover líneas que contienen palabras clave de explicación
        lines = response.split('\n')
        cleaned_lines = []
        
        skip_keywords = [
            'después de evaluar', 'puedo decir', 'he determinado', 
            'correcciones realizadas', 'nota:', 'importante:', 
            'sin embargo', 'a continuación', 'te presento',
            'ejemplo:', 'input:', 'output:'
        ]
        
        for line in lines:
            line = line.strip()
            if line and not any(keyword in line.lower() for keyword in skip_keywords):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_useful_text(self, response):
        """Extrae texto útil de una respuesta que no sigue el formato esperado"""
        # Dividir en párrafos y buscar el más sustancial
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        
        best_paragraph = ""
        max_words = 0
        
        for paragraph in paragraphs:
            # Saltar párrafos que parecen explicaciones
            if any(word in paragraph.lower() for word in ['evaluar', 'corrección', 'mejora', 'ejemplo']):
                continue
                
            word_count = len(paragraph.split())
            if word_count > max_words and word_count >= 10:
                max_words = word_count
                best_paragraph = paragraph
        
        return best_paragraph if max_words >= 10 else None

def process_files_with_validation():
    """Procesa archivos de texto con validación de chunks"""
    
    # Configurar rutas
    BASE_DIR = Path(__file__).resolve().parent.parent
    CLEANED_DIR = BASE_DIR / "data" / "cleaned_advanced"
    CHUNKED_DIR = BASE_DIR / "data" / "chunks_advanced"
    
    # Crear directorio de chunks si no existe
    CHUNKED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verificar que existe el directorio de archivos limpios
    if not CLEANED_DIR.exists():
        print(f"[!] Error: El directorio {CLEANED_DIR} no existe")
        print("    Asegúrate de tener archivos .txt en la carpeta 'cleaned_advanced'")
        return
    
    # Inicializar validador
    print("[🤖] Inicializando validador de chunks...")
    try:
        validator = ChunkValidator(model_name="llama3.2:3b")
    except Exception as e:
        print(f"[!] Error inicializando validador: {e}")
        return
    
    # Configurar el splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,         # caracteres (aprox. 500 tokens)
        chunk_overlap=150,       # solapamiento para mantener contexto
        separators=["\n\n", "\n", ".", " "]  # fragmenta primero por párrafos
    )
    
    # Procesar cada archivo
    txt_files = [f for f in os.listdir(CLEANED_DIR) if f.endswith(".txt")]
    
    if not txt_files:
        print(f"[!] No se encontraron archivos .txt en {CLEANED_DIR}")
        return
    
    print(f"[📂] Encontrados {len(txt_files)} archivos para procesar")
    
    total_chunks_original = 0
    total_chunks_valid = 0
    total_files_processed = 0
    
    for filename in txt_files:
        print(f"\n[📄] Procesando: {filename}")
        
        input_path = CLEANED_DIR / filename
        
        try:
            # Leer archivo
            with open(input_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            
            if not full_text.strip():
                print(f"[!] Archivo vacío: {filename}")
                continue
            
            # Fragmentar en chunks
            chunks = text_splitter.split_text(full_text)
            total_chunks_original += len(chunks)
            
            print(f"[🔄] Validando {len(chunks)} chunks...")
            
            # Validar y limpiar cada chunk
            valid_chunks = []
            
            for i, chunk in enumerate(chunks, 1):
                print(f"    Procesando chunk {i}/{len(chunks)}...", end=" ")
                
                is_valid, cleaned_chunk = validator.validate_and_clean_chunk(chunk)
                
                if is_valid and cleaned_chunk and cleaned_chunk.strip():
                    # Verificación adicional de calidad
                    if len(cleaned_chunk.split()) >= 8:  # Al menos 8 palabras
                        valid_chunks.append(cleaned_chunk.strip())
                        print("✓ VÁLIDO")
                    else:
                        print("✗ MUY CORTO")
                else:
                    print("✗ INVÁLIDO")
                
                # Pausa para no sobrecargar
                time.sleep(0.2)
            
            total_chunks_valid += len(valid_chunks)
            
            # Guardar chunks válidos
            if valid_chunks:
                base = os.path.splitext(filename)[0]
                output_path = CHUNKED_DIR / f"{base}_chunks.txt"
                
                with open(output_path, "w", encoding="utf-8") as f:
                    for i, chunk in enumerate(valid_chunks, 1):
                        f.write(f"[CHUNK {i}]\n{chunk}\n\n")
                
                print(f"[✅] {len(valid_chunks)} chunks válidos guardados para {filename}")
                total_files_processed += 1
            else:
                print(f"[⚠️] No se encontraron chunks válidos en {filename}")
                
        except Exception as e:
            print(f"[!] Error procesando {filename}: {e}")
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"[📊] RESUMEN DEL PROCESAMIENTO:")
    print(f"    • Archivos procesados: {total_files_processed}")
    print(f"    • Chunks originales: {total_chunks_original}")
    print(f"    • Chunks válidos: {total_chunks_valid}")
    print(f"    • Chunks descartados: {total_chunks_original - total_chunks_valid}")
    print(f"    • Tasa de validez: {(total_chunks_valid/total_chunks_original*100):.1f}%")
    print(f"[📁] Chunks guardados en: {CHUNKED_DIR}")

# if __name__ == "__main__":
#     print("🚀 Sistema RAG - Validador de Chunks con LLM")
#     print("=" * 50)
    
#     # Verificar requisitos
#     print("[1/4] Verificando requisitos...")
    
#     try:
#         import requests
#         print("    ✓ Requests disponible")
#     except ImportError:
#         print("    ✗ Error: pip install requests")
#         exit(1)
        
#     try:
#         from langchain.text_splitter import RecursiveCharacterTextSplitter
#         print("    ✓ LangChain disponible")
#     except ImportError:
#         print("    ✗ Error: pip install langchain")
#         exit(1)
    
#     print("\n[2/4] Verificando Ollama...")
#     try:
#         response = requests.get("http://localhost:11434/api/tags", timeout=5)
#         if response.status_code == 200:
#             print("    ✓ Ollama está corriendo")
            
#             # Verificar si el modelo está disponible
#             models = response.json().get('models', [])
#             model_names = [model['name'] for model in models]
            
#             if 'llama3.2:3b' in model_names:
#                 print("    ✓ Modelo llama3.2:3b disponible")
#             else:
#                 print("    ⚠️ Modelo llama3.2:3b NO encontrado")
#                 print("    💡 Instálalo con: ollama pull llama3.2:3b")
                
#                 user_choice = input("\n¿Continuar de todas formas? El script intentará descargarlo (y/n): ").lower().strip()
#                 if user_choice != 'y':
#                     print("Operación cancelada. Instala el modelo primero.")
#                     exit(1)
#         else:
#             print("    ✗ Ollama no responde correctamente")
#             exit(1)
#     except requests.exceptions.ConnectionError:
#         print("    ✗ No se puede conectar a Ollama")
#         print("    💡 ¿Está Ollama corriendo? Ejecuta: ollama serve")
#         exit(1)
#     except Exception as e:
#         print(f"    ✗ Error verificando Ollama: {e}")
#         exit(1)
    
#     print("\n[3/4] Iniciando procesamiento...")
#     process_files_with_validation()
    
#     print("\n[4/4] ¡Proceso completado! 🎉")