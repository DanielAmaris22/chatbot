# utils.py
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json

# Cargar el documento PDF
def load_pdf():
    loader = PyPDFLoader("llm_doc.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(documents)
    data = [{'Chunks': doc.page_content, 'Metadata': doc.metadata} for doc in doc_splits]
    return pd.DataFrame(data)

# Obtener embedding de un texto
def text_embedding(text=[]):
    # Cargar la clave API
    file_name = open('credentials.json')
    config_env = json.load(file_name)
    api_key = config_env["openai_key"]
    client = OpenAI(api_key=api_key)
    
    embeddings = client.embeddings.create(model="text-embedding-ada-002", input=text,encoding_format="float")
    return embeddings.data[0].embedding

# Calcular embeddings para el Vector Store
def create_embeddings(df_vector_store):
    df_vector_store["Embedding"] = df_vector_store["Chunks"].apply(lambda x: text_embedding([x]))
    df_vector_store["Embedding"] = df_vector_store["Embedding"].apply(np.array)
    return df_vector_store

# Función para obtener el contexto relevante basado en la similitud
def get_context_from_query(query, vector_store, n_chunks=5):
    query_embedding = np.array(text_embedding([query]))  # Generar el embedding para la pregunta
    vector_store['CosineSimilarity'] = vector_store['Embedding'].apply(
        lambda x: cosine_similarity([x], [query_embedding])[0][0]
    )
    top_matches = vector_store.sort_values(by='CosineSimilarity', ascending=False).head(n_chunks)
    context_list = top_matches['Chunks'].tolist()  # Extraer los textos más relevantes
    return context_list

# Generar respuesta utilizando GPT-4
def generar_respuesta_gpt4(client, query, context_list):
    # Crear el prompt para el modelo GPT-4 con el contexto y la pregunta
    custom_prompt = f"""
    Eres una Inteligencia Artificial avanzada.
    Utiliza los siguientes fragmentos de texto para responder la pregunta de manera precisa:

    CONTEXTO:
    {str(context_list)}

    PREGUNTA:
    {query}

    Responde de forma breve y clara.
    """
    
    # Llamar a GPT-4 para obtener la respuesta
    completion = client.chat.completions.create(
        model="gpt-4",
        temperature=0.0,
        messages=[{"role": "system", "content": custom_prompt}, {"role": "user", "content": query}]
    )
    
    return completion.choices[0].message.content

# Calcular similitud entre respuestas generadas y respuestas propuestas
def calcular_similitud(respuesta_generada, respuesta_propuesta):
    emb_chat = np.array(text_embedding([respuesta_generada])).reshape(1, -1)  # Embedding de la respuesta generada
    emb_propuesta = np.array(text_embedding([respuesta_propuesta])).reshape(1, -1)  # Embedding de la respuesta propuesta
    similitud = cosine_similarity(emb_propuesta, emb_chat)[0][0]
    return similitud
