import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.callbacks.manager import get_openai_callback
from langchain.schema import Document  
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import os
import re
from bs4 import BeautifulSoup
import fitz  # PyMuPDF para procesar PDFs
import pandas as pd  # Para procesar archivos Excel
import openai  # Para utilizar la API de OpenAI


# Cargar variables de entorno
load_dotenv()
SALUDOS= "Hola en que puedo asistirte, ¡Saludos! ¿Cómo puedo ayudarte hoy?, ¡Bienvenido/a! ¿Cómo puedo asistirte?, ¡Qué gusto verte por aquí! ¿Cómo puedo ayudarte hoy? "
NOMBRE_DE_LA_EMPRESA = "Corporación Write"
NOMBRE_AGENTE = "Kliofer"

prompt_inicial = f"""
Soy {NOMBRE_AGENTE}, parte del equipo de {NOMBRE_DE_LA_EMPRESA}, un asistente inteligente diseñado para responder exclusivamente preguntas basadas en tu base de conocimiento. Tu conocimiento está limitado a la información contenida en estos documentos, y tu objetivo principal es ayudar a resolver consultas relacionadas con ellos de manera eficiente y amigable. 
Estas interactuando con personas que trabajan en esta empresa.


1. Inicia la conversacion presentandote.
Tu propósito es servir a todas las consultas que se hagan sobre lo que está en la base de conocimiento de {NOMBRE_DE_LA_EMPRESA} que básicamente es la información de tu base de conocimiento, proporcionándoles respuestas precisas y útiles. No puedes ofrecer información sobre temas fuera de tu base de conocimiento.
Si se te realiza una pregunta fuera del contexto de los archivos, por favor responde con amabilidad, explicando que no tienes información sobre ese tema.
En caso de que no encuentres suficiente información en tu base de conocimiento para responder a una consulta, explica de manera clara y amable las razones de tu limitación.
Recuerda siempre ser cordial, servicial y profesional, priorizando la satisfacción del usuario y ayudando a resolver sus dudas dandole informacion sobre los archivos proporcionados.
Gracias por tu colaboración.
Basándote en el historial de la conversación. Responde preguntas que esten en el historial de conversacion.
Si las respuestas que se solicitan preguntale si buscabas algun tema especifico para que le muestres solo la que corresponde si es el caso de que hay muchas opciones.
"""

# Inicializar memoria en session_state si no existe
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Inicializar variable response en session_state para evitar error
if 'response' not in st.session_state:
    st.session_state.response = None  # Inicializamos como None

def conexion_a_mongo():
    """
    Conecta a MongoDB y devuelve la colección de chats.
    """
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")  # Usar variable de entorno o conexión local
    client = MongoClient(MONGO_URI)
    db = client["db-historial-chats"]  # Nombre de la base de datos
    collection = db["coleccion-histochats"]  # Nombre de la colección donde guardaremos los chats
    return collection

def guardar_chat_en_mongo(user, message, response):
    """
    Guarda chats en MongoDB pero sin resuperarción.
    """
    collection = conexion_a_mongo()
    chat_data = {
        "user": user,
        "message": message,
        "response": response,
        "timestamp": datetime.now()
    }
    collection.insert_one(chat_data)  # Inserta el documento en la colección

def extraer_titulos(text):
    """
    Identifica títulos en el texto basándose en patrones como:
    - Líneas en mayúsculas
    - Texto con caracteres especiales (indicando encabezados)
    """
    titles = []
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r'^[A-Z\s]+$', line) and len(line) > 5:
            titles.append(line)
    return titles

def procesar_texto_con_gerarquía(text):
    """
    Procesa el texto y lo divide en chunks jerárquicos,
    asegurando que los títulos agrupan los contenidos correctos.
    """
    titles = extraer_titulos(text)
    
    paragraphs = text.split("\n\n")
    processed_paragraphs = []
    current_section = "General"  # Título predeterminado si no se encuentra uno
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        
        # Si encontramos un título, lo usamos como la nueva sección activa
        if paragraph in titles:
            current_section = paragraph  # Almacena el título como sección
        else:
            # Cada chunk se asocia a la sección actual
            processed_paragraphs.append((current_section, paragraph))
    
    return processed_paragraphs


def process_text_with_hierarchy(text):
    """
    Crea una base de conocimiento jerárquica con ChromaDB.
    """
    processed_data = procesar_texto_con_gerarquía(text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    documents = []
    
    for section, content in processed_data:
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"section": section}))
    
    knowledge_base = Chroma.from_documents(documents, embeddings) if documents else None
    return knowledge_base

def search_with_hierarchy(query, knowledge_base):
    """
    Realiza búsqueda jerárquica:
    1. Busca en títulos primero.
    2. Luego busca en los detalles dentro de las secciones relevantes.
    """
    if not knowledge_base:
        return "No hay información almacenada."
    
    # Primera fase: Buscar en títulos
    title_docs = knowledge_base.similarity_search(query, k=3)
    
    relevant_sections = set(doc.metadata["section"] for doc in title_docs)
    
    # Segunda fase: Buscar dentro de esas secciones
    all_results = []
    for section in relevant_sections:
        section_results = knowledge_base.similarity_search(query, k=2, filter={"section": section})
        all_results.extend(section_results)
    
    return all_results

def print_hierarchical_chunks(knowledge_base):
    """
    Imprime los chunks organizados por su jerarquía de títulos en ChromaDB.
    """
    if not knowledge_base:
        print("⚠️ No hay información en la base de datos.")
        return
    
    # Obtener todos los documentos almacenados
    docs = knowledge_base._collection.get(include=["documents", "metadatas"])

    # Crear un diccionario para organizar los chunks por secciones
    sections = {}
    for doc, meta in zip(docs["documents"], docs["metadatas"]):
        section = meta.get("section", "Sin sección")
        if section not in sections:
            sections[section] = []
        sections[section].append(doc)

    # Imprimir la jerarquía
    for section, chunks in sections.items():
        print(f"\n📂 **Sección: {section}**")
        for i, chunk in enumerate(chunks):
            print(f"  📜 Chunk {i+1}: {chunk[:200]}...")  # Solo mostramos 200 caracteres por chunk


# Función principal
def main():
    st.sidebar.markdown("**Autores:**.  \n- *Nicolas Liberio*  \n- *Mateo Rivadeneira*")
    st.sidebar.image('logo.jpg', width=250)
    st.markdown('<h1 style="color:  #FFD700;">InfoBot </h1>', unsafe_allow_html=True)
    
    # 🔽 Inicializamos la base de conocimiento en session_state para evitar errores 🔽
    if "knowledgeBase" not in st.session_state:
        st.session_state["knowledgeBase"] = None
    
    uploaded_files = st.file_uploader("Sube archivos (PDF, CSV, HTML, XML)", type=["pdf", "csv", "xlsx", "html", "xml"], accept_multiple_files=True)
    text = ""

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            if file_type in ["text/html", "application/xml"]:
                soup = BeautifulSoup(uploaded_file, 'html.parser' if file_type == "text/html" else 'xml')
                text += soup.get_text()
            elif file_type == "application/pdf":
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text += "".join([page.get_text("text") for page in doc])
            elif file_type == "text/csv":
                df = pd.read_csv(uploaded_file)
                text += df.to_string(index=False) + "\n"
            


    # 🔹 Crear Tabs para separar Chat, Historial/Costos y Chunks
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📜 Historial & Costos", "🔍 Ver Chunks"])

    with tab1:
        st.markdown("## 💬 Chat con InfoBot")
        query = st.text_input('Escribe tu pregunta...')
        cancel_button = st.button('Cancelar')

        if cancel_button:
            st.stop()

        if query:
            knowledgeBase = st.session_state.get("knowledgeBase", None)
            if knowledgeBase:
                docs = knowledgeBase.similarity_search(query)
                context = "\n".join([doc.page_content for doc in docs]) if docs else "No hay información relevante."
                
                # Obtener historial de la conversación desde la memoria
                history_messages = st.session_state.memory.load_memory_variables({}).get("history", [])
                messages = [{"role": "system", "content": prompt_inicial}]
                
                # Agregar historial de conversación al mensaje
                for message in history_messages:
                    messages.append({"role": "user", "content": message.content})  
                    messages.append({"role": "assistant", "content": message.content})  

                # Agregar la nueva consulta
                messages.append({"role": "user", "content": query})

                # Agregar el contexto relevante si existe
                if context:
                    messages.append({"role": "system", "content": context})
            else:
                messages = [{"role": "system", "content": prompt_inicial}, {"role": "user", "content": query}]
            
            with get_openai_callback() as obtienec:
                start_time = datetime.now()
                st.session_state.response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=messages,
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
                end_time = datetime.now()
                
                answer = st.session_state.response['choices'][0]['message']['content'] if st.session_state.response.get('choices') else "Lo siento, no pude obtener una respuesta."
                
                # Guardar el contexto de la nueva conversación
                st.session_state.memory.save_context({"input": query}, {"output": answer})
                
                guardar_chat_en_mongo("nliberio", query, answer)
                
                st.write(answer)

    with tab2:
        st.markdown("## 📜 Historial de Conversación y Costos")
        st.write("### 🔹 Historial de Conversación")
        st.write(st.session_state.memory.buffer)

        st.write("### 💰 Costos y Tokens")


        if st.session_state.response:
            response = st.session_state.response  
            prompt_tokens = response['usage'].get('prompt_tokens', 0)
            completion_tokens = response['usage'].get('completion_tokens', 0)
            total_tokens = response['usage'].get('total_tokens', 0)

            costo_total = ((prompt_tokens * 0.00001) + (completion_tokens * 0.00003))

            st.write(f"🔹 Tokens de entrada: {prompt_tokens}")
            st.write(f"🔹 Tokens de salida: {completion_tokens}")
            st.write(f"🔹 Total tokens: {total_tokens}")
            st.write(f"💵 **Costo Total:** ${costo_total:.4f}")
            st.write(f"🕒 **Tiempo de proceso:** {end_time - start_time}")
            st.write(f"📅 **Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("❌ No hay datos disponibles. Haz una consulta en el chat primero.")
        if text:
                st.session_state.knowledgeBase = process_text_with_hierarchy(text)
        if "knowledgeBase" in st.session_state and st.session_state.knowledgeBase:
         st.write("📂 **Verificando e imprimiendo jerarquía de chunks en la base de conocimiento...**")
        print_hierarchical_chunks(st.session_state.knowledgeBase)  # 🔍 Aquí verificamos la jerarquía


    with tab3:
        st.markdown("## 🔍 Inspección de Chunks en Chroma")

        if "knowledgeBase" in st.session_state and st.session_state.knowledgeBase:
            num_chunks = st.session_state.knowledgeBase._collection.count()
            st.write(f"📂 **Total de chunks en la base de datos:** {num_chunks}")

            docs = st.session_state.knowledgeBase._collection.get(include=["documents"], limit=3)
            st.write("🔎 **Visualiza los primeros 3 chunks:**")
            for i, doc in enumerate(docs["documents"]):
                st.write(f"📜 **Chunk {i+1}:**")
                st.text(doc)

            # 🟢 Nueva sección para evaluar similitud con una consulta
            st.markdown("## 🔎 Comparar Similitud de Embeddings")
            query_sim = st.text_input("🔍 Escribe una pregunta para comparar con los embeddings:")

            if query_sim:
                with st.spinner("Buscando en Chroma..."):
                    results = st.session_state.knowledgeBase.similarity_search_with_score(query_sim, k=3)

                    if results:
                        st.write("📊 **Resultados de Similitud:**")
                        
                        # Calcular la suma total de similitud para normalizar el porcentaje
                        suma_total_scores = sum(score for _, score in results)

                        for i, (doc, score) in enumerate(results):
                            porcentaje_importancia = (score / suma_total_scores) * 100  # Convertimos la similitud en porcentaje

                            with st.expander(f"🔹 Resultado {i+1} (Importancia: {porcentaje_importancia:.2f}%)"):
                                st.write(f"📜 **Chunk:**")
                                st.text(doc.page_content)
                                st.write(f"🔹 **Similitud:** {score:.4f} ({porcentaje_importancia:.2f}% de importancia)")
                    else:
                        st.warning("⚠️ No se encontraron resultados relevantes para esta consulta.")
        else:
            st.warning("⚠️ No hay chunks almacenados. Sube un archivo primero.")


if __name__ == "__main__":
    main()
