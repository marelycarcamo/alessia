"""
Archivo:
    bootstrap.py

Módulo:
    Inicialización de aplicación.

Objetivo:
    Centralizar la creación de servicios necesarios para la aplicación.

Descripción:
    Este módulo contiene las funciones encargadas de inicializar
    las dependencias principales de la aplicación.

    Permite mantener la capa de presentación desacoplada de la
    configuración interna del pipeline RAG.

Proyecto:
    ALESSIA
    Asistente basado en RAG para la gestión del riesgo de desastres.
"""


import streamlit as st

from src.services.agent_service import AgentService


@st.cache_resource(show_spinner=False)
def load_agent_service() -> AgentService:
    """
    Inicializa y reutiliza el servicio principal del agente.

    El servicio se almacena como recurso de Streamlit para evitar
    recrear innecesariamente los modelos de embeddings, reranking
    y la conexión persistente con ChromaDB en cada rerun de la aplicación.

    La creación del recurso desactiva el indicador de Streamlit, que
    muestra el nombre de esta función y otra información interna que no
    aporta nada a quien utiliza la aplicación.

    Returns
    -------
    AgentService
        Servicio configurado para ejecutar consultas RAG.
    """

    return AgentService.create()