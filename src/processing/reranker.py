"""
Archivo:
    reranker.py

Módulo:
    Processing - Reranking

Objetivo:
    Reordenar los fragmentos recuperados durante la etapa de Retrieval
    utilizando un modelo CrossEncoder para estimar la relevancia entre
    la consulta del usuario y cada fragmento documental.

Descripción:
    Este módulo recibe objetos Chunk provenientes del Retriever y calcula
    un puntaje de relevancia utilizando un modelo CrossEncoder.

    La función retorna los mismos objetos Chunk ordenados desde el más
    relevante al menos relevante.

    La interacción con ChromaDB queda completamente encapsulada en capas
    anteriores del pipeline.

Proyecto:
    ALESSIA - Asistente de gestión del riesgo de desastres
    para la comuna de Valdivia.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.models.chunk import Chunk

if TYPE_CHECKING:
    # Solo para anotaciones: ver la nota en embedder.py.
    from sentence_transformers import CrossEncoder


def _sigmoid(value: float) -> float:
    """
    Convierte un logit en una probabilidad entre 0 y 1.

    Se evita calcular `exp(-value)` con valores grandes porque desborda:
    en logits muy negativos la exponencial crece sin control. Por eso se
    usa la forma numéricamente estable según el signo.
    """

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))

    exponential = math.exp(value)

    return exponential / (1.0 + exponential)


def rerank_chunks(
    query: str,
    chunks: list[Chunk],
    model: CrossEncoder,
    top_k: int = 3,
) -> list[Chunk]:
    """
    Ordena fragmentos documentales según su relevancia para una consulta.

    Para cada fragmento se calcula un puntaje utilizando un modelo
    CrossEncoder. Posteriormente los fragmentos son ordenados de mayor
    a menor relevancia y se retorna únicamente la cantidad solicitada.

    Args:
        query:
            Consulta realizada por el usuario.

        chunks:
            Lista de objetos Chunk recuperados durante Retrieval.

        model:
            Modelo CrossEncoder utilizado para calcular relevancia.

        top_k:
            Cantidad máxima de fragmentos a retornar.

    Returns:
        Lista de objetos Chunk ordenados por relevancia.
    """

    if not chunks:
        return []

    # Construye pares consulta-documento para el modelo CrossEncoder.
    sentence_pairs = [
        (
            query,
            chunk.content,
        )
        for chunk in chunks
    ]

    # Calcula puntajes de relevancia.
    scores = model.predict(
        sentence_pairs
    )

    # Guarda el score dentro de metadata para mantener trazabilidad.
    #
    # Se almacenan dos valores:
    #
    # - rerank_score: el logit original. Sirve para depurar y comparar
    #   corridas, pero no es interpretable para un usuario ("relevancia
    #   -1.32" parece un error del sistema).
    # - rerank_probability: el mismo valor convertido a probabilidad
    #   entre 0 y 1. El CrossEncoder se entrena con entropía cruzada
    #   binaria, así que la sigmoide del logit es su propia estimación de
    #   "este fragmento responde a la consulta". Es lo que se puede
    #   mostrar en la interfaz y lo que permite fijar un umbral con
    #   significado.
    for chunk, score in zip(
        chunks,
        scores,
    ):
        logit = float(score)

        chunk.metadata["rerank_score"] = logit
        chunk.metadata["rerank_probability"] = _sigmoid(logit)

    # Ordena los fragmentos desde mayor a menor relevancia.
    ranked_chunks = sorted(
        chunks,
        key=lambda chunk: chunk.metadata["rerank_score"],
        reverse=True,
    )

    return ranked_chunks[:top_k]