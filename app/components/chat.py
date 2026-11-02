"""
Archivo:
    chat.py

Módulo:
    components

Objetivo:
    Gestionar la interfaz conversacional de la aplicación Streamlit.

Descripción:
    Este módulo contiene los componentes visuales relacionados con la
    interacción entre el usuario y el asistente.

    Su responsabilidad es administrar el historial de conversación,
    capturar preguntas del usuario y mostrar las respuestas generadas
    por AgentService.

    Este componente no contiene lógica del pipeline RAG. La recuperación
    de información, reranking y generación de respuestas son delegados
    al servicio de aplicación.

Proyecto:
    ALESSIA
    Asistente basado en RAG para la gestión del riesgo de desastres
    en la comuna de Valdivia.
"""

"""
Componente encargado de renderizar la interfaz conversacional
de ALESSIA.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from app.config import (
    ALESSIA_AVATAR_PATH,
    USER_AVATAR_PATH,
)


WELCOME_MESSAGE = (
    "¡Hola, vecino! 👋\n\n"
    "Soy **ALESSIA**, tu asistente especializada en "
    "gestión del riesgo de desastres para la comuna de Valdivia.\n\n"
    "Puedo ayudarte a consultar información oficial sobre "
    "planes comunales, amenazas, emergencias, zonas de riesgo "
    "y medidas de preparación.\n\n"
    "¿En qué puedo ayudarte hoy?"
)


def _build_sources_markdown(
    sources: list[dict[str, Any]],
) -> str:
    """
    Construye el detalle de fuentes consultadas para una respuesta.

    Parameters
    ----------
    sources : list[dict[str, Any]]
        Fragmentos utilizados durante la generación de la respuesta.

    Returns
    -------
    str
        Detalle de fuentes en formato Markdown.
    """

    lines = []

    for source in sources:

        origin = str(
            source.get("source", "")
        )

        title = str(
            source.get("title", "")
        )

        # Se muestra únicamente el nombre del archivo para no exponer
        # rutas internas del servidor donde se ejecuta la aplicación.
        label = (
            Path(origin).name
            or title
            or "Documento oficial"
        )

        index = source.get(
            "chunk_index"
        )

        probability = source.get(
            "rerank_probability"
        )

        detail = f"- **{label}**"

        if index is not None:
            detail += f" · fragmento {index}"

        # Se muestra la probabilidad y no el puntaje crudo del modelo.
        # Un valor como "relevancia -1.32" parece un error del sistema:
        # es un logit sin escala interpretable. La probabilidad sí se
        # entiende y además es la que decide el umbral de contexto.
        if probability is not None:
            detail += f" · pertinencia {probability:.0%}"

        lines.append(detail)

    return "\n".join(lines)


def render_chat(agent_service):
    """
    Renderiza la interfaz conversacional y procesa
    las consultas del usuario.

    Parameters
    ----------
    agent_service :
        Servicio encargado de ejecutar el pipeline RAG.

    Returns
    -------
    None
    """

    # Inicializar historial de conversación.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ]

    # Mostrar historial existente.
    for message in st.session_state.messages:

        avatar = (
            str(ALESSIA_AVATAR_PATH)
            if message["role"] == "assistant"
            else str(USER_AVATAR_PATH)
        )

        with st.chat_message(
            message["role"],
            avatar=avatar,
        ):
            st.markdown(
                message["content"]
            )

            # Las respuestas conservan las fuentes utilizadas.
            sources = message.get(
                "sources"
            )

            if sources:
                st.expander(
                    "📄 Fuentes consultadas"
                ).markdown(
                    _build_sources_markdown(sources)
                )

    # Capturar nueva consulta.
    question = st.chat_input(
        "Escribe tu consulta..."
    )

    if question:

        # Pregunta anterior del usuario, si existe. Se usa solo para
        # reintentar la búsqueda cuando el mensaje actual, por sí solo,
        # no recupera fragmentos utilizables.
        previous_question = next(
            (
                message["content"]
                for message in reversed(st.session_state.messages)
                if message["role"] == "user"
            ),
            None,
        )

        # Guardar mensaje del usuario.
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # Mostrar mensaje del usuario.
        with st.chat_message(
            "user",
            avatar=str(USER_AVATAR_PATH),
        ):
            st.markdown(question)

        # Ejecutar el pipeline RAG.
        with st.spinner(
            "ALESSIA está analizando la información oficial..."
        ):
            result = agent_service.ask(
                question,
                previous_question,
            )

        # Guardar respuesta del asistente junto con sus fuentes.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
            }
        )

        # Mostrar respuesta de ALESSIA.
        with st.chat_message(
            "assistant",
            avatar=str(ALESSIA_AVATAR_PATH),
        ):
            st.markdown(
                result.answer
            )

            # Mantiene la trazabilidad hacia la documentación utilizada.
            if result.sources:
                st.expander(
                    "📄 Fuentes consultadas"
                ).markdown(
                    _build_sources_markdown(result.sources)
                )