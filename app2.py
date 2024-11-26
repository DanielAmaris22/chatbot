import streamlit as st
import pandas as pd
from openai import OpenAI
import json
from utils import get_context_from_query, generar_respuesta_gpt4, calcular_similitud 

with open('credentials.json') as file:
    config_env = json.load(file)
api_key = config_env["openai_key"]

client = OpenAI(api_key=api_key)

# Cargar archivo de preguntas y respuestas
questions_answers_file = "preguntas_respuestas.xlsx"  
df_questions = pd.read_excel(questions_answers_file)

df_questions = df_questions.head(8)

df_vector_store = pd.read_pickle('df_vector_store.pkl')

def main_page():
    
    st.set_page_config(page_title="Chatbot Interactivo", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "model" not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"

    with st.sidebar:
        st.image('usta.png', use_column_width="always")
        st.header("Chat personalizado :robot_face:")
        st.subheader("Configuración del modelo :level_slider:")

        # Permitir al usuario elegir entre diferentes modelos de OpenAI.
        model_name = st.radio("**Elije un modelo**:", ("GPT-3.5", "GPT-4"))
        if model_name == "GPT-3.5":
            st.session_state.model = "gpt-3.5-turbo"
        elif model_name == "GPT-4":
            st.session_state.model = "gpt-4"
        
        # Permitir al usuario ajustar el nivel de creatividad (temperatura) de las respuestas generadas.
        st.session_state.temperature = st.slider("**Nivel de creatividad de respuesta**  \n  [Poco creativo ►►► Muy creativo]",
                                                 min_value=0.0,
                                                 max_value=1.0,
                                                 step=0.1,
                                                 value=0.0)
    
    if prompt := st.selectbox("¿Cuál es tu consulta?", df_questions['Pregunta']):

        # Mostrar la consulta del usuario en el chat.
        with st.chat_message("user"):
            st.markdown(prompt)

        # Obtener la respuesta propuesta desde el archivo de preguntas y respuestas
        respuesta_propuesta = df_questions[df_questions['Pregunta'] == prompt]["Respuesta"].values[0]
        
        # Obtener el contexto relevante para la pregunta
        context_list = get_context_from_query(query=prompt, vector_store=df_vector_store, n_chunks=5)
        
        if context_list:  
            # Generar la respuesta del chatbot
            respuesta_chatbot = generar_respuesta_gpt4(client, prompt, context_list)
            
            # Mostrar las respuestas
            with st.chat_message("assistant"):
                st.markdown(f"**Respuesta del Chatbot:** {respuesta_chatbot}")
                st.markdown(f"**Respuesta Propuesta:** {respuesta_propuesta}")
            
            # Botón para calcular similitud
            if st.button("Calcular Similitud"):
                similitud = calcular_similitud(respuesta_chatbot, respuesta_propuesta)
                st.write(f"**Similitud:** {similitud:.2f}")

        else:
            st.write("No se encontró contexto relevante en el documento para generar la respuesta.")

# Punto de entrada de la aplicación Streamlit
if __name__ == "__main__":
    main_page()
