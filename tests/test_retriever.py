"""
Archivo:
    retriever.py

Módulo:
    Processing - Retrieval

Objetivo:
    Recuperar fragmentos relevantes desde el Vector Store y convertirlos
    en objetos Chunk utilizados por el pipeline RAG.

Descripción:
    Este módulo actúa como capa de adaptación entre ChromaDB y la lógica
    de negocio de ALESSIA.

    La respuesta interna del vector store puede cambiar dependiendo de la
    tecnología utilizada. Por esta razón, el resto del sistema trabaja
    únicamente con objetos Chunk y no depende de estructuras específicas
    entregadas por ChromaDB.

    Esta separación permite mantener trazabilidad de los documentos y
    facilita futuras migraciones del motor vectorial.

Proyecto:
    ALESSIA - Asistente de gestión del riesgo de desastres
    para la comuna de Valdivia.
"""


from typing import Any

from src.models.chunk import Chunk
from src.processing.vector_store import search_chunks


def retrieve_context(
    query: str,
    collection: Any,
    model: Any,
    k: int = 5,
) -> list[Chunk]:
    """
    Recupera los fragmentos más relevantes para una consulta.

    La función realiza la búsqueda semántica mediante el Vector Store
    y transforma los resultados obtenidos en objetos Chunk.

    Args:
        query:
            Consulta realizada por el usuario.

        collection:
            Colección persistente del Vector Store.

        model:
            Modelo utilizado para generar el embedding de la consulta.

        k:
            Cantidad máxima de fragmentos a recuperar.

    Returns:
        Lista de objetos Chunk ordenados según similitud vectorial.
    """

    results = search_chunks(
        query=query,
        collection=collection,
        model=model,
        k=k,
    )

    documents = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]

    chunks = []

    for chunk_id, content, metadata in zip(
        ids,
        documents,
        metadatas,
    ):
        chunk = Chunk(
            id=chunk_id,
            document_id=metadata.get(
                "document_id",
                "",
            ),
            chunk_index=metadata.get(
                "chunk_index",
                0,
            ),
            content=content,
            metadata=metadata,
        )

        chunks.append(chunk)

    return chunks