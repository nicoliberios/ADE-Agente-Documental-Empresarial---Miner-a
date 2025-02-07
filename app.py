from langchain_community.vectorstores.chroma import Chroma
# from langchain.vectorstores.chroma import Chroma
from bs4 import BeautifulSoup
from datetime import datetime
from langchain_community.llms import OpenAI
# Importa la función load_dotenv del módulo dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
# Importa el módulo os para interactuar con el sistema operativo
import os
# Importa el módulo langchain
import langchain
# Importa la biblioteca Streamlit para crear aplicaciones web interactivas
import streamlit as st
# Importa el CharacterTextSplitter del módulo langchain.text_splitter para dividir texto en caracteres
from langchain.text_splitter import CharacterTextSplitter
# Importa OpenAIEmbeddings del módulo langchain.embeddings.openai para generar incrustaciones de texto utilizando OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# Importa load_qa_chain del módulo langchain.chains.question_answering para cargar cadenas de preguntas y respuestas
from langchain.chains.question_answering import load_qa_chain
# Importa OpenAI del módulo langchain.llms para interactuar con el modelo de lenguaje de OpenAI
from langchain.llms import OpenAI
# Importa get_openai_callback del módulo langchain.callbacks para obtener realimentación de OpenAI
from langchain.callbacks import get_openai_callback


# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Función para procesar el texto extraído de un archivo HTML
def process_text(text):
    # Divide el texto en trozos usando langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    knowledge_base = Chroma.from_texts(chunks, embeddings) if chunks else None

    return knowledge_base


# Función principal de la aplicación
def main():
    # Agrega una foto de perfil en la barra lateral
    st.sidebar.markdown("**Autores:**.  \n- *Nicolas Liberio*  \n- *Mateo Rivadeneira*")
    st.sidebar.write("")
    st.sidebar.image('logo.jpg', width=250)
    st.markdown('<h1 style="color:  #FFD700;">ADE (Agente Documental Empresarial)</h1>', unsafe_allow_html=True)  # Establece el título de la aplicación

    html = st.file_uploader("HTML", type="html")
    xml = st.file_uploader("xml", type="xml")
    text = ""

    # Iterar sobre cada par (file, parser)
    for file, parser in [(html, 'html.parser'), (xml, 'xml')]:
        # Verificar si el archivo no es None
        if file is not None:
            # Crear el objeto BeautifulSoup y obtener el texto
            soup = BeautifulSoup(file, parser)
            text += soup.get_text()

    if text:
        # Crea un objeto de base de conocimientos a partir del texto del HTML
        knowledgeBase = process_text(text)


        # Caja de entrada de texto para que el usuario escriba su pregunta
        query = st.text_input('Escribe tu pregunta para los HTML...')

        # Botón para cancelar la pregunta
        cancel_button = st.button('Cancelar')

        if cancel_button:
            st.stop()  # Detiene la ejecución de la aplicación

        if query and knowledgeBase:
            # Realiza una búsqueda de similitud en la base de conocimientos
            docs = knowledgeBase.similarity_search(query)

            # Inicializa un modelo de lenguaje de OpenAI y ajustamos sus parámetros
            model = "gpt-3.5-turbo-instruct"  # Acepta 4096 tokens
            temperature = 0
            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

            # Carga la cadena de preguntas y respuestas
            chain = load_qa_chain(llm, chain_type="map_reduce")

            # Obtiene la realimentación de OpenAI para el procesamiento de la cadena
            with get_openai_callback() as obtienec:
                start_time = datetime.now()
                response = chain.invoke(input={"question": query, "input_documents": docs})
                end_time = datetime.now()
                total_tokens = obtienec.total_tokens  # Total de tokens utilizados
                time = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")  # excluyo parte de los microsegundos con .strftime("%Y-%m-%d %H:%M:%S")
                duracion_proceso = end_time - start_time
                st.write(response["output_text"])  # Muestra el texto de salida de la cadena de preguntas y respuestas en la aplicación
                st.write(f"Total de tokens usados:   {total_tokens}")
                st.write(f"Tiempo de proceso:        {duracion_proceso}")
                st.write(f"Fecha:                    {time}")

# Punto de entrada para la ejecución del programa
if __name__ == "__main__":
    main()