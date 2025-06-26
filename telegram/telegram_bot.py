import telebot
import sys

sys.path.append("/home/luismeza/Documentos/Tecno_Emer/Proyecto_RAG/rag-estudiantes/app")
from rag_pipeline_llama import build_or_load_index, retrieve_top_k_with_synonyms
from rag_pipeline_llama import SYSTEM_INSTRUCTIONS

from langchain_ollama import OllamaLLM
import time

TOKEN = "YOUR_TOKEN"
LLM_MODEL_NAME = "llama3.2:3b"  # Cambiado de gemma:2b a llama3.2:3b
SYNONYM_MODEL_NAME = "gemma:2b"  # Modelo ligero para sinónimos
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Holaaaa")


@bot.message_handler(commands=['help'])
def send_help(message):
    chat_id = message.chat.id
    print(f"Chat ID: {message.chat.id}")
    print(f"Chat Type: {message.chat.type}")
    print(f"User ID: {message.from_user.id}")
    
    # Método 1: Acción simple antes del procesamiento
    result = bot.send_chat_action(chat_id, 'typing')
    print(f"✅ Typing enviado, resultado: {result}")
    bot.reply_to(message, "Mensaje de ayuda")


@bot.message_handler(func = lambda m: True)
def send_message(message):
    chat_id = message.chat.id

    
    # Método 1: Acción simple antes del procesamiento
    result = bot.send_chat_action(chat_id, 'typing')
    sent_message = bot.send_message(chat_id, "Escribiendo...")
    
    
    
    query = message.text
    start_time = time.time()
        
    # 3. Recuperar contexto con sinónimos y expansión
    top_chunks = retrieve_top_k_with_synonyms(query, synonym_llm, k=3, data = [index, metadata, texts, document_chunks])
        
    if not top_chunks:
        answer= "\n🤖 Respuesta:\nLo sentimos, no se encontró información específica sobre tu consulta en la base de datos o no esta relacionada a aspectos universitarios. Te recomiendo acercarte al áre de infromes ubicada en el pabellon L primer piso para obtener información más detallada y actualizada."
        #bot.reply_to(message, answer)
        bot.edit_message_text(
                chat_id=chat_id,
                message_id=sent_message.message_id,
                text=answer
            )
        return
        
        
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
    #bot.reply_to(message, answer)
    bot.edit_message_text(
        chat_id=chat_id,
        message_id=sent_message.message_id,
        text=answer
    )
    
            
    # 6. Mostrar fuentes consultadas
    print(f"\n📄 Fuentes consultadas ({len(top_chunks)} documentos):")
    for i, c in enumerate(top_chunks, 1):
        print(f"{i}. {c['metadata']['source']} (chunk {c['metadata']['chunk_id']}) - Similitud: {c['distance']:.3f}")
        if c['query_used'] != query:
            print(f"   🔍 Encontrado con: '{c['query_used']}'")
            
    print(f"\n⏱️ Tiempo total de respuesta: {total_time:.2f}s") 


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
    
    bot.polling(non_stop=True)
        