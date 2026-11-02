"""
Archivo:
    test_vector_store.py

Módulo:
    Tests del almacenamiento vectorial.

Objetivo:
    Validar que los objetos Chunk sean transformados correctamente
    antes de ser almacenados en ChromaDB.

Descripción:
    Verifica que vector_store conserve embeddings, contenido y
    metadata necesaria para mantener la trazabilidad del pipeline RAG.

Proyecto:
    ALESSIA
    Asistente basado en RAG para gestión del riesgo de desastres.
"""


from pathlib import Path
from unittest.mock import MagicMock

import chromadb
import numpy as np

from src.models.chunk import Chunk
from src.processing.vector_store import index_chunks


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_index_chunks():
    """
    Verifica que los chunks sean almacenados correctamente
    incluyendo metadata de trazabilidad.
    """

    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        chunk_index=0,
        content="Contenido de prueba.",
        metadata={
            "page": 1,
        },
        embedding=np.array(
            [
                0.1,
                0.2,
                0.3,
            ]
        ),
    )

    collection = MagicMock()

    index_chunks(
        chunks=[chunk],
        collection=collection,
    )

    collection.add.assert_called_once_with(
        ids=[
            "chunk-1",
        ],
        embeddings=[
            [
                0.1,
                0.2,
                0.3,
            ]
        ],
        documents=[
            "Contenido de prueba.",
        ],
        metadatas=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "chunk_index": 0,
                "page": 1,
            }
        ],
    )


def test_indice_versionado_contiene_fragmentos():
    """El índice que viaja al despliegue no puede estar vacío.

    Streamlit Community Cloud no tiene un paso de compilación donde
    ejecutar ``scripts/build_index.py``: el índice tiene que viajar en el
    repositorio.

    Si viaja vacío, la aplicación arranca sin documentos y responde
    "No encontré información" a cualquier consulta, **sin mostrar ningún
    error**. Es un fallo silencioso y por eso esta prueba existe: impide
    publicar un índice vacío, que es indistinguible de un motor roto.
    """

    client = chromadb.PersistentClient(
        path=str(REPO_ROOT / "data" / "vector_store"),
    )

    collection = client.get_or_create_collection(
        "alessia_collection",
    )

    assert collection.count() > 0