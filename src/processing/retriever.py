"""
Archivo: retriever.py

Módulo:
Procesamiento (Processing)

Objetivo:
Recuperar los fragmentos de texto más relevantes para una consulta
realizada por el usuario utilizando búsqueda semántica sobre el
índice vectorial.

Descripción:
Este módulo actúa como intermediario entre la consulta del usuario y
el Vector Store. Su responsabilidad es recuperar los Chunk más
relevantes a partir de una búsqueda por similitud, proporcionando el
contexto que será utilizado posteriormente durante la generación de
respuestas del sistema RAG.

Proyecto: ALESSIA

"""


"""
Módulo encargado de transformar los resultados obtenidos desde ChromaDB
en objetos del dominio utilizados por ALESSIA.

La capa Retriever evita que el resto del pipeline dependa directamente
de la estructura interna del vector store.
"""

from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.models.chunk import Chunk
from src.processing.vector_store import search_chunks


def retrieve_context(
    query: str,
    collection: Collection,
    model: SentenceTransformer,
    k: int = 5,
) -> list[Chunk]:
    """
    Recupera los fragmentos más relevantes para una consulta.

    Convierte la respuesta entregada por ChromaDB en objetos Chunk,
    manteniendo una representación consistente dentro del pipeline RAG.

    Args:
        query:
            Consulta realizada por el usuario.

        collection:
            Colección vectorial donde se realiza la búsqueda.

        model:
            Modelo utilizado para generar embeddings.

        k:
            Cantidad máxima de fragmentos recuperados.

    Returns:
        Lista de objetos Chunk recuperados.
    """

    results = search_chunks(
        query=query,
        collection=collection,
        model=model,
        k=k,
    )

    # ChromaDB devuelve listas vacías cuando la colección no contiene
    # fragmentos, por lo que la ausencia de resultados no es un error.
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    ids = (results.get("ids") or [[]])[0] or []

    chunks = []

    for chunk_id, document, metadata in zip(
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
            content=document,
            metadata=metadata,
        )

        chunks.append(chunk)

    return chunks