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


def test_agent_service_answers_greeting_without_pipeline():
    """Un saludo no se busca en los planes de emergencia.

    Buscarlo produce una negativa que hace parecer que el sistema no
    entendió nada, aunque técnicamente sea correcto.
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

        result = service.ask("hola")

    assert "Soy ALESSIA" in result.answer
    assert result.sources == []

    mock_retrieve.assert_not_called()
    mock_generate.assert_not_called()


def test_agent_service_answers_identity_question():
    """La pregunta por la identidad se responde sin inventar personalidad."""

    with patch.object(
        AgentService, "_retrieve_context"
    ) as mock_retrieve:

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        result = service.ask("¿Eres valdiviana?")

    assert "No soy una persona" in result.answer
    assert result.sources == []

    mock_retrieve.assert_not_called()


def test_agent_service_answers_capabilities_question():
    """Saber qué puede hacer es parte de usar el asistente."""

    with patch.object(
        AgentService, "_retrieve_context"
    ) as mock_retrieve:

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        result = service.ask("¿Qué puedes hacer?")

    assert "planes de emergencia" in result.answer

    mock_retrieve.assert_not_called()


def test_greeting_with_real_question_goes_to_pipeline():
    """Un saludo con una consulta adentro sigue siendo una consulta.

    Es el caso que rompe la detección naive por "contiene hola": quien
    escribe esto necesita información, no un saludo de vuelta.
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
            "hola, ¿qué hago si hay un tsunami?"
        )

    assert "no encontré información" in result.answer.lower()

    mock_retrieve.assert_called_once()


def test_agent_service_retries_with_previous_question():
    """Un seguimiento sin tema propio se reencuentra con el turno anterior.

    Caso medido: "Qué raro, tengo entendido que hay vías de evacuación..."
    después de una pregunta sobre tsunamis. Sin el turno anterior, el mejor
    fragmento quedó en 1,8 % de pertinencia y el sistema negó información
    que el plan sí contiene. Anteponiendo la pregunta previa, tres
    fragmentos superaron el umbral.
    """

    irrelevante = Mock(
        document_id="doc-1",
        chunk_index=1,
        metadata={
            "rerank_probability": 0.018,
        },
    )

    relevante = Mock(
        document_id="doc-1",
        chunk_index=61,
        metadata={
            "rerank_probability": 0.63,
        },
    )

    with patch.object(
        AgentService,
        "_retrieve_context",
        side_effect=[[irrelevante], [relevante]],
    ) as mock_retrieve, patch.object(
        AgentService,
        "_rerank_chunks",
        side_effect=lambda question, chunks: chunks,
    ), patch.object(
        AgentService,
        "_generate_answer",
        return_value="El plan identifica 25 puntos de encuentro.",
    ):

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        result = service.ask(
            "¿y las vías de evacuación?",
            "¿Hay plan de emergencia ante tsunami en la costa?",
        )

    assert "25 puntos de encuentro" in result.answer

    assert mock_retrieve.call_count == 2

    segunda_consulta = mock_retrieve.call_args_list[1].args[0]

    assert "tsunami" in segunda_consulta
    assert "vías de evacuación" in segunda_consulta


def test_agent_service_does_not_retry_when_first_search_works():
    """El reintento solo ocurre cuando la primera búsqueda no sirve.

    Repetir la búsqueda siempre costaría latencia en todas las consultas
    para arreglar una minoría.
    """

    relevante = Mock(
        document_id="doc-1",
        chunk_index=3,
        metadata={
            "rerank_probability": 0.92,
        },
    )

    with patch.object(
        AgentService,
        "_retrieve_context",
        return_value=[relevante],
    ) as mock_retrieve, patch.object(
        AgentService,
        "_rerank_chunks",
        side_effect=lambda question, chunks: chunks,
    ), patch.object(
        AgentService,
        "_generate_answer",
        return_value="La alcaldesa convoca al COGRID.",
    ):

        service = AgentService(
            collection=Mock(),
            embedding_model=Mock(),
            reranker_model=Mock(),
        )

        service.ask(
            "¿Quién convoca al COGRID Comunal?",
            "¿Hay plan de tsunami?",
        )

    assert mock_retrieve.call_count == 1