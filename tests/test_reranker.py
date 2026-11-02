"""
Archivo:
    test_reranker.py

Módulo:
    Tests unitarios - Reranking

Objetivo:
    Validar el comportamiento del módulo reranker.py,
    verificando que los fragmentos documentales sean ordenados
    correctamente según su relevancia calculada por el modelo
    CrossEncoder.

Descripción:
    Estos tests validan que:

    - el módulo reciba objetos Chunk;
    - los puntajes de relevancia sean asociados a cada fragmento;
    - los resultados sean ordenados de mayor a menor relevancia;
    - la cantidad de resultados retornados respete el parámetro top_k.

    El modelo CrossEncoder es simulado para evitar dependencia de
    modelos externos durante las pruebas unitarias.

Proyecto:
    ALESSIA - Asistente de gestión del riesgo de desastres
    para la comuna de Valdivia.
"""


from unittest.mock import Mock

from src.models.chunk import Chunk
from src.processing.reranker import rerank_chunks


def test_rerank_chunks_orders_chunks_by_score():
    """
    Verifica que rerank_chunks ordene los fragmentos
    desde mayor a menor relevancia.
    """

    chunks = [
        Chunk(
            id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            content="Fragmento poco relevante.",
            metadata={},
        ),
        Chunk(
            id="chunk-2",
            document_id="doc-1",
            chunk_index=1,
            content="Fragmento muy relevante.",
            metadata={},
        ),
    ]

    mock_model = Mock()

    mock_model.predict.return_value = [
        0.20,
        0.90,
    ]

    results = rerank_chunks(
        query="¿Qué hacer ante una inundación?",
        chunks=chunks,
        model=mock_model,
        top_k=2,
    )

    assert results[0].id == "chunk-2"
    assert results[1].id == "chunk-1"

    assert results[0].metadata["rerank_score"] == 0.90
    assert results[1].metadata["rerank_score"] == 0.20


def test_rerank_chunks_returns_top_k_results():
    """
    Verifica que rerank_chunks respete la cantidad máxima
    de resultados solicitados.
    """

    chunks = [
        Chunk(
            id=f"chunk-{index}",
            document_id="doc-1",
            chunk_index=index,
            content=f"Contenido {index}",
            metadata={},
        )
        for index in range(5)
    ]

    mock_model = Mock()

    mock_model.predict.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]

    results = rerank_chunks(
        query="Consulta de prueba",
        chunks=chunks,
        model=mock_model,
        top_k=3,
    )

    assert len(results) == 3

    assert results[0].metadata["rerank_score"] == 0.5


def test_rerank_chunks_exposes_probability():
    """Cada fragmento recibe además una probabilidad interpretable.

    El puntaje del CrossEncoder es un logit sin escala fija: "relevancia
    -1.32" no le dice nada a un usuario. La sigmoide del logit sí es una
    probabilidad, y es lo que permite fijar un umbral con significado.
    """

    chunks = [
        Chunk(
            id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            content="Fragmento sin relación.",
            metadata={},
        ),
        Chunk(
            id="chunk-2",
            document_id="doc-1",
            chunk_index=1,
            content="Fragmento que sí responde.",
            metadata={},
        ),
    ]

    mock_model = Mock()

    mock_model.predict.return_value = [
        -4.98,
        4.60,
    ]

    results = rerank_chunks(
        query="Consulta",
        chunks=chunks,
        model=mock_model,
        top_k=2,
    )

    assert results[0].id == "chunk-2"
    assert results[0].metadata["rerank_probability"] > 0.99

    assert results[1].id == "chunk-1"
    assert results[1].metadata["rerank_probability"] < 0.01


def test_rerank_chunks_probability_is_monotonic_with_score():
    """La probabilidad conserva el orden del puntaje original.

    Si no fuera monótona, el umbral cortaría fragmentos en un orden
    distinto al que decidió el modelo.
    """

    chunks = [
        Chunk(
            id=f"chunk-{index}",
            document_id="doc-1",
            chunk_index=index,
            content=f"Contenido {index}",
            metadata={},
        )
        for index in range(4)
    ]

    mock_model = Mock()

    mock_model.predict.return_value = [
        -6.0,
        -1.0,
        0.5,
        3.0,
    ]

    results = rerank_chunks(
        query="Consulta",
        chunks=chunks,
        model=mock_model,
        top_k=4,
    )

    probabilities = [
        chunk.metadata["rerank_probability"]
        for chunk in results
    ]

    assert probabilities == sorted(probabilities, reverse=True)