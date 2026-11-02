"""
Módulo encargado de la persistencia y recuperación de embeddings
utilizando ChromaDB como vector store.

ALESSIA 1.0 utiliza ChromaDB como capa de almacenamiento vectorial.
La adaptación hacia objetos de dominio (Chunk) se realiza en capas
superiores del pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.models.chunk import Chunk

if TYPE_CHECKING:
    # Solo para anotaciones: ver la nota en embedder.py.
    from chromadb.api.models.Collection import Collection
    from sentence_transformers import SentenceTransformer


def get_or_create_collection(
    collection_name: str,
    vector_store_path: Path,
) -> Collection:
    """
    Obtiene una colección existente de ChromaDB o la crea si no existe.

    Args:
        collection_name:
            Nombre identificador de la colección.

        vector_store_path:
            Ruta donde ChromaDB mantiene la persistencia.

    Returns:
        Colección activa de ChromaDB.
    """

    # Import diferido: chromadb arrastra dependencias pesadas y este
    # módulo se importa desde toda la aplicación, incluidos los tests que
    # nunca tocan un vector store.
    import chromadb

    client = chromadb.PersistentClient(
        path=str(vector_store_path)
    )

    collection = client.get_or_create_collection(
        name=collection_name
    )

    return collection


def index_chunks(
    chunks: list[Chunk],
    collection: Collection,
) -> None:
    """
    Indexa fragmentos documentales dentro del vector store.

    Cada Chunk se almacena junto con:
    - contenido textual;
    - embedding;
    - información mínima de trazabilidad.

    La estructura metadata utiliza diccionarios porque corresponde
    al formato requerido por ChromaDB.
    """

    ids = [
        chunk.id
        for chunk in chunks
    ]

    # Evita duplicar información al ejecutar nuevamente la indexación.
    collection.delete(
        ids=ids
    )

    collection.add(
        ids=ids,

        embeddings=[
            chunk.embedding.tolist()
            for chunk in chunks
        ],

        documents=[
            chunk.content
            for chunk in chunks
        ],

        metadatas=[
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
            for chunk in chunks
        ],
    )


def search_chunks(
    query: str,
    collection: Collection,
    model: SentenceTransformer,
    k: int = 5,
):
    """
    Recupera fragmentos relevantes mediante búsqueda semántica.

    La consulta es transformada en embedding y posteriormente
    utilizada para buscar coincidencias dentro de ChromaDB.

    Args:
        query:
            Pregunta realizada por el usuario.

        collection:
            Colección vectorial donde se realiza la búsqueda.

        model:
            Modelo utilizado para generar embeddings.

        k:
            Cantidad máxima de resultados.

    Returns:
        Resultado nativo entregado por ChromaDB.

    Note:
        La conversión del resultado hacia objetos Chunk pertenece
        a la capa Retriever.
    """

    query_embedding = model.encode(
        query
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=k,
    )

    return results