"""
Archivo: test_agent_service.py

Módulo:
    tests

Objetivo:
    Validar el comportamiento de AgentService como capa de orquestación
    del pipeline RAG, verificando la correcta comunicación entre los
    módulos de recuperación, reranking y generación de respuestas.

Descripción:
    Este módulo contiene pruebas unitarias para comprobar que AgentService
    coordina correctamente las distintas etapas del pipeline RAG.

    Las pruebas utilizan objetos simulados para evitar depender de
    servicios externos como ChromaDB, modelos de embeddings o proveedores
    LLM, enfocándose en validar la responsabilidad del servicio como
    punto de entrada de la aplicación.

Proyecto:
    ALESSIA
    Asistente basado en RAG para la gestión del riesgo de desastres
    en la comuna de Valdivia.
"""
from unittest.mock import Mock, patch

from src.services.agent_service import AgentService


def test_agent_service_executes_pipeline():
    """AgentService.ask coordina retrieval, reranking, generation
    y expone las fuentes utilizadas."""

    question = "¿Qué hacer ante una inundación?"

    expected_answer = (
        "Mantenerse informado por canales oficiales."
    )

    retrieved_chunk = Mock(
        document_id="doc-1",
        chunk_index=4,
        metadata={
            "source": "data/raw/Plan-Comunal.pdf",
            "page": 12,
            "rerank_score": 0.91,
        },
    )

    ranked_chunk = Mock(
        document_id="doc-1",
        chunk_index=4,
        metadata={
            "source": "data/raw/Plan-Comunal.pdf",
            "page": 12,
            "rerank_score": 0.91,
        },
    )

    with patch.object(
        AgentService, "_retrieve_context", return_value=[retrieved_chunk]
    ) as mock_retrieve, \
    patch.object(
        AgentService, "_rerank_chunks", return_value=[ranked_chunk]
    ) as mock_rerank, \
    patch.object(
        AgentService, "_generate_answer", return_value=expected_answer
    ) as mock_generate:

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        result = service.ask(question)

    assert result.answer == expected_answer

    assert result.sources == [
        {
            "document_id": "doc-1",
            "chunk_index": 4,
            "source": "data/raw/Plan-Comunal.pdf",
            "title": "",
            "page": 12,
            "rerank_score": 0.91,
        }
    ]

    mock_retrieve.assert_called_once_with(question)

    mock_rerank.assert_called_once_with(question, [retrieved_chunk])

    mock_generate.assert_called_once_with(question, [ranked_chunk])


def test_agent_service_avoids_model_without_context():
    """Sin contexto recuperado no se consulta al modelo de lenguaje."""

    with patch.object(
        AgentService, "_retrieve_context", return_value=[]
    ), patch.object(
        AgentService, "_generate_answer"
    ) as mock_generate:

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        result = service.ask("¿Cuál es el plan de evacuación?")

    # Se informa al usuario en lugar de responder sin fundamento.
    assert "No encontré información" in result.answer
    assert result.sources == []

    mock_generate.assert_not_called()


def test_agent_service_informs_provider_errors():
    """Un error del proveedor se comunica al usuario sin propagarse."""

    chunk = Mock(
        document_id="doc-1",
        chunk_index=0,
        metadata={},
    )

    with patch.object(
        AgentService, "_retrieve_context", return_value=[chunk]
    ), patch.object(
        AgentService, "_rerank_chunks", return_value=[chunk]
    ), patch.object(
        AgentService,
        "_generate_answer",
        side_effect=ValueError("No se encontró la API Key de OpenRouter."),
    ):

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        result = service.ask("¿Qué hacer ante una inundación?")

    assert "no pudo generar la respuesta" in result.answer
    assert "API Key" in result.answer
    assert result.sources == []