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

from src.processing.llm_client import LLMProviderError
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
            "rerank_score": 4.60,
            "rerank_probability": 0.99,
        },
    )

    ranked_chunk = Mock(
        document_id="doc-1",
        chunk_index=4,
        metadata={
            "source": "data/raw/Plan-Comunal.pdf",
            "page": 12,
            "rerank_score": 4.60,
            "rerank_probability": 0.99,
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
            "rerank_score": 4.60,
            "rerank_probability": 0.99,
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
        metadata={
            "rerank_probability": 0.95,
        },
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


def test_agent_service_informs_provider_failures():
    """Un fallo del servicio LLM (timeout, error HTTP) no se propaga."""

    chunk = Mock(
        document_id="doc-1",
        chunk_index=0,
        metadata={
            "rerank_probability": 0.95,
        },
    )

    with patch.object(
        AgentService, "_retrieve_context", return_value=[chunk]
    ), patch.object(
        AgentService, "_rerank_chunks", return_value=[chunk]
    ), patch.object(
        AgentService,
        "_generate_answer",
        side_effect=LLMProviderError("NVIDIA no respondió correctamente."),
    ):

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        result = service.ask("¿Qué hacer ante una inundación?")

    assert "no pudo generar la respuesta" in result.answer
    assert result.sources == []


def test_agent_service_rejects_irrelevant_context():
    """Los fragmentos bajo el umbral no llegan al modelo de lenguaje.

    Un fragmento puede ser del tema correcto y aun así no responder la
    consulta. En ese caso la respuesta honesta es no tener información,
    no una respuesta construida sobre material que no la contiene.
    """

    irrelevant_chunk = Mock(
        document_id="doc-1",
        chunk_index=7,
        metadata={
            "rerank_score": -4.98,
            "rerank_probability": 0.0068,
        },
    )

    with patch.object(
        AgentService, "_retrieve_context", return_value=[irrelevant_chunk]
    ), patch.object(
        AgentService, "_rerank_chunks", return_value=[irrelevant_chunk]
    ), patch.object(
        AgentService, "_generate_answer"
    ) as mock_generate:

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        result = service.ask(
            "¿Qué rol cumple SENAPRED en la emergencia?"
        )

    assert "No encontré información" in result.answer
    assert result.sources == []

    mock_generate.assert_not_called()


def test_agent_service_answers_emergency_without_model():
    """Una emergencia en curso no espera al modelo ni al retrieval.

    En una emergencia, la latencia del proveedor (40 a 113 segundos
    medidos) y su disponibilidad no pueden ser el eslabón del que
    depende la respuesta.
    """

    with patch.object(
        AgentService, "_retrieve_context"
    ) as mock_retrieve, patch.object(
        AgentService, "_generate_answer"
    ) as mock_generate:

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        result = service.ask(
            "hay un incendio en mi casa, necesito ayuda ahora"
        )

    assert "Esto es una emergencia" in result.answer
    assert "132" in result.answer
    assert result.sources == []

    mock_retrieve.assert_not_called()
    mock_generate.assert_not_called()


def test_agent_service_treats_hypothetical_as_regular_query():
    """Preguntar por incendios no es tener una emergencia.

    La detección exige una amenaza nombrada y un marcador de urgencia;
    sin el segundo, la consulta sigue el camino normal.
    """

    with patch.object(
        AgentService, "_retrieve_context", return_value=[]
    ) as mock_retrieve:

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        result = service.ask(
            "¿Qué medidas previenen los incendios forestales?"
        )

    assert "no encontré información" in result.answer.lower()

    mock_retrieve.assert_called_once()