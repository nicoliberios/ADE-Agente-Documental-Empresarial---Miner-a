import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.callbacks.manager import get_openai_callback
from datetime import datetime
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import fitz  # PyMuPDF para procesar PDFs
import pandas as pd  # Para procesar archivos Excel
import openai  # Para utilizar la API de OpenAI


# Cargar las variables de entorno
load_dotenv()
SALUDOS= "Hola en que puedo asistirte, ¡Saludos! ¿Cómo puedo ayudarte hoy?, ¡Bienvenido/a! ¿Cómo puedo asistirte?, ¡Qué gusto verte por aquí! ¿Cómo puedo ayudarte hoy? "
NOMBRE_DE_LA_EMPRESA = "Corporación Write"
NOMBRE_AGENTE = "Kliofer"

prompt_inicial = f"""
Eres {NOMBRE_AGENTE}, parte del equipo de {NOMBRE_DE_LA_EMPRESA}, un asistente inteligente diseñado para responder exclusivamente preguntas basadas en tu base de conocimiento. Tu conocimiento está limitado a la información contenida en estos documentos, y tu objetivo principal es ayudar a resolver consultas relacionadas con ellos de manera eficiente y amigable. 
Estas interactuando con personas que trabajan en esta empresa.


1. Inicia la conversacion presentandote.
Tu propósito es servir a todas las consultas que se hagan sobre lo que está en la base de conocimiento de {NOMBRE_DE_LA_EMPRESA} que básicamente es la información de tu base de conocimiento, proporcionándoles respuestas precisas y útiles. No puedes ofrecer información sobre temas fuera de tu base de conocimiento.
Si se te realiza una pregunta fuera del contexto de los archivos, por favor responde con amabilidad, explicando que no tienes información sobre ese tema.
En caso de que no encuentres suficiente información en tu base de conocimiento para responder a una consulta, explica de manera clara y amable las razones de tu limitación.
Recuerda siempre ser cordial, servicial y profesional, priorizando la satisfacción del usuario y ayudando a resolver sus dudas dentro de los límites de los archivos proporcionados.
Gracias por tu colaboración.
Basándote en el historial de la conversación. Responde preguntas que esten en el historial de conversacion.
"""


# Inicializar memoria en session_state si no existe
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Verificar si ya se ha agregado el prompt inicial
if 'prompt_inicial_added' not in st.session_state:
    st.session_state.prompt_inicial_added = False

# Encargada de dividir el texto en chunks y crear la knowledge
def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=300, length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = Chroma.from_texts(chunks, embeddings) if chunks else None
    return knowledge_base

# Función principal
def main():
    st.sidebar.markdown("**Autores:**.  \n- *Nicolas Liberio*  \n- *Mateo Rivadeneira*")
    st.sidebar.image('logo.jpg', width=250)
    st.markdown('<h1 style="color:  #FFD700;">InfoBot </h1>', unsafe_allow_html=True)
    
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
    
    #Procesamiento del Texto (División en Chunks)
    if text:
        st.session_state.knowledgeBase = process_text(text)
    
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
                messages.append({"role": "user", "content": message.content})  # Acceder a `content` directamente
                messages.append({"role": "assistant", "content": message.content})  # Acceder a `content` directamente

            # Agregar la nueva consulta
            messages.append({"role": "user", "content": query})

            # Agregar el contexto relevante si existe
            if context:
                messages.append({"role": "system", "content": context})
        else:
            messages = [{"role": "system", "content": prompt_inicial}, {"role": "user", "content": query}]
        
        with get_openai_callback() as obtienec:
            start_time = datetime.now()
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=messages,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
            end_time = datetime.now()
            
            
                # Obtener los tokens de entrada y salida
            prompt_tokens = response['usage'].get('prompt_tokens', 0)
            completion_tokens = response['usage'].get('completion_tokens', 0)
            total_tokens = response['usage'].get('total_tokens', 0)
                    
            answer = response['choices'][0]['message']['content'] if response.get('choices') else "Lo siento, no pude obtener una respuesta."
            
            # Guardar el contexto de la nueva conversación
            st.session_state.memory.save_context({"input": query}, {"output": answer})
            
            
            st.write(answer)
            st.write("Historial de conversación:", st.session_state.memory.buffer)
            
            # Mostrar los tokens
            st.write(f"Tokens de entrada (prompt): {prompt_tokens}")
            st.write(f"Tokens de salida (completación): {completion_tokens}")
            st.write(f"Total de tokens usados: {total_tokens}")
                    
            costo_total = ((prompt_tokens*0.00001)+(completion_tokens*0.00003))
            st.write(f"Costo Total: ${costo_total:.4f}")
            st.write(f"Tiempo de proceso: {end_time - start_time}")
            st.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.write("Bienvenido, por favor sube archivos para comenzar.")

if __name__ == "__main__":
    main()
