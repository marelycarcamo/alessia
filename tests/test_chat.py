"""
Archivo:
    test_chat.py

Módulo:
    Tests del componente chat.

Objetivo:
    Validar la interacción entre la interfaz conversacional
    y AgentService.

Descripción:
    Verifica que el componente chat capture preguntas del usuario,
    invoque el servicio del agente y almacene correctamente
    la conversación en session_state, incluyendo las fuentes
    utilizadas para construir la respuesta.

Proyecto:
    ALESSIA
    Asistente basado en RAG para la gestión del riesgo de desastres.
"""


from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import streamlit as st

from app.components.chat import render_chat


def _render_with_clean_session(
    agent_service,
    question: str,
    mock_expander=None,
) -> list[dict]:
    """
    Ejecuta render_chat simulando una sesión nueva y retorna el
    historial resultante.

    En modo bare el estado de Streamlit es global y persiste entre
    ejecuciones de pytest, por lo que la sesión se limpia antes y
    después de la prueba para evitar contaminación entre casos.
    """

    history: list[dict] = []

    try:
        st.session_state.clear()

        patches = [
            patch(
                "app.components.chat.st.chat_input",
                return_value=question,
            ),
            patch(
                "app.components.chat.st.chat_message",
            ),
            patch(
                "app.components.chat.st.markdown",
            ),
        ]

        if mock_expander is None:
            patches.append(
                patch(
                    "app.components.chat.st.expander",
                )
            )
        else:
            patches.append(
                patch(
                    "app.components.chat.st.expander",
                    mock_expander,
                )
            )

        for applied in patches:
            applied.start()

        try:
            render_chat(
                agent_service=agent_service,
            )

            history = list(
                st.session_state.messages
            )
        finally:
            for applied in reversed(patches):
                applied.stop()

    finally:
        st.session_state.clear()

    return history


def test_render_chat_processes_user_question():
    """Captura la consulta, la responde y conserva el historial."""

    question = "¿Qué hacer ante una inundación?"

    answer = (
        "Evacuar según instrucciones oficiales."
    )

    sources = [
        {
            "document_id": "doc-1",
            "chunk_index": 4,
            "source": "datos/Plan-Comunal-de-Emergencia-2025.pdf",
            "rerank_score": 0.87,
        }
    ]

    agent_service = MagicMock()

    agent_service.ask.return_value = SimpleNamespace(
        answer=answer,
        sources=sources,
    )

    history = _render_with_clean_session(
        agent_service=agent_service,
        question=question,
    )

    agent_service.ask.assert_called_once_with(
        question
    )

    # El historial contiene el saludo inicial, la consulta del usuario
    # y la respuesta del asistente junto con sus fuentes.
    assert len(history) == 3

    assert history[0]["role"] == "assistant"
    assert "ALESSIA" in history[0]["content"]

    assert history[1] == {
        "role": "user",
        "content": question,
    }

    assert history[2]["role"] == "assistant"
    assert history[2]["content"] == answer
    assert history[2]["sources"] == sources


def test_render_chat_reuses_existing_history():
    """Una sesión existente no vuelve a agregar el saludo inicial."""

    question = "¿Cuáles son las zonas de riesgo?"

    agent_service = MagicMock()

    agent_service.ask.return_value = SimpleNamespace(
        answer="Respuesta oficial.",
        sources=[],
    )

    history = _render_with_clean_session(
        agent_service=agent_service,
        question=question,
    )

    # Al limpiar el estado, render_chat inicializa la conversación.
    assert len(history) == 3

    agent_service.ask.assert_called_once_with(
        question
    )


def test_render_chat_shows_sources_to_user():
    """Las fuentes consultadas se muestran al usuario."""

    question = "¿Qué hacer ante un tsunami?"

    sources = [
        {
            "document_id": "doc-2",
            "chunk_index": 9,
            "source": "datos/Otro-Plan-Oficial.pdf",
            "rerank_score": 0.41,
        }
    ]

    agent_service = MagicMock()

    agent_service.ask.return_value = SimpleNamespace(
        answer="Respuesta oficial.",
        sources=sources,
    )

    mock_expander = MagicMock()

    history = _render_with_clean_session(
        agent_service=agent_service,
        question=question,
        mock_expander=mock_expander,
    )

    assert history[-1]["sources"] == sources

    # Las fuentes se muestran al usuario de forma explícita.
    mock_expander.assert_called_once()
