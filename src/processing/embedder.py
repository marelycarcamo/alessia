"""
Archivo: embedder.py

Módulo:
Procesamiento (Processing)

Objetivo:
Generar representaciones vectoriales (embeddings) para los chunks
utilizando un modelo de Sentence Transformers.

Descripción:
Este módulo transforma el contenido textual de cada Chunk en un
vector numérico que preserva su significado semántico. Estos vectores
serán utilizados posteriormente para construir el índice vectorial
del sistema RAG.

Proyecto: ALESSIA

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.chunk import Chunk

if TYPE_CHECKING:
    # Solo para anotaciones de tipo. Importar sentence_transformers aquí
    # arrastra torch y suma decenas de segundos a cualquier import del
    # proyecto, incluidos los tests que nunca usan un modelo.
    from sentence_transformers import SentenceTransformer


def embed_chunk(
    chunk: Chunk,
    model: SentenceTransformer,
) -> Chunk:

    """
	Genera los embeddings para una colección de objetos Chunk.

	Esta función reutiliza ``embed_chunk`` para mantener la lógica
	de generación de embeddings en un único lugar.

    Args:
        chunk: Fragmento de texto que será transformado en un vector semántico.
        model: Modelo previamente cargado que realizará la generación del embedding.

    Returns:
        El mismo objeto Chunk enriquecido con su representación vectorial.
    """

    # El modelo se recibe como parámetro para evitar acoplar el módulo
    # a una implementación específica y permitir reutilizar la función
    # con distintos modelos de embeddings.
    chunk.embedding = model.encode(chunk.content)

    return chunk


def generate_embeddings(
    chunks: list[Chunk],
    model: SentenceTransformer,
) -> list[Chunk]:
    """
    Genera embeddings para una colección de Chunk.

    Args:
        chunks: Lista de fragmentos que serán transformados en vectores semánticos.
        model: Modelo previamente cargado para generar los embeddings.

    Returns:
        Lista de Chunk enriquecidos con su representación vectorial.
    """

    for chunk in chunks:
        # Se reutiliza embed_chunk para mantener una única
        # implementación de la generación de embeddings.
        embed_chunk(chunk, model)

    return chunks

