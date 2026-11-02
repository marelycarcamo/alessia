"""
Archivo:
    test_build_index.py

Módulo:
    Tests del script de construcción del índice vectorial.

Objetivo:
    Validar que el pipeline de ingesta cargue los documentos desde
    disco, los fragmente, genere embeddings y los indexe en ChromaDB,
    y que la opción --recreate reconstruya la colección desde cero.

Descripción:
    Los documentos se escriben en disco en formato HTML, que no
    requiere dependencias externas para su lectura. Sólo se reemplaza
    el modelo de embeddings, de modo que las pruebas no descargan
    pesos ni dependen de documentos reales. ChromaDB se ejecuta sobre
    un directorio temporal dentro del proyecto.

Proyecto:
    ALESSIA
    Asistente basado en RAG para la gestión del riesgo de desastres.
"""


from pathlib import Path
from unittest.mock import patch

import chromadb
import numpy as np

from scripts import build_index


BODY_TEXT = (
    "Plan Comunal de Emergencia de Valdivia. "
    "Medidas de evacuación ante tsunami en la zona costera. "
    "Roles del Comité Comunal de Emergencia durante la respuesta. "
) * 10


class FakeEmbeddingModel:
    """Modelo de embeddings simulado, sin descarga de pesos."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, text):
        # Se imita el tipo de retorno de sentence-transformers.
        return np.array([0.1, 0.2, 0.3])


def _write_documents(directory: Path) -> Path:
    """Escribe los documentos oficiales de prueba en disco."""

    source_dir = directory / "planes"
    source_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (source_dir / "plan-comunal.html").write_text(
        f"<html><body><p>{BODY_TEXT}</p></body></html>",
        encoding="utf-8",
    )

    (source_dir / "notas.txt").write_text(
        "Formato no soportado.",
        encoding="utf-8",
    )

    return source_dir


def _run_build(
    directory: Path,
    recreate: bool = False,
) -> int:
    """Ejecuta el script sobre un índice temporal."""

    source_dir = _write_documents(directory)

    arguments = [
        "--input",
        str(source_dir),
        "--store",
        str(directory / "vector_store"),
    ]

    if recreate:
        arguments.append("--recreate")

    with patch.object(
        build_index, "load_embedding_model", FakeEmbeddingModel
    ):
        return build_index.main(arguments)


def _collection(directory: Path):
    """Abre la colección generada durante la prueba."""

    client = chromadb.PersistentClient(
        path=str(directory / "vector_store")
    )

    return client.get_collection(
        build_index.DEFAULT_COLLECTION
    )


def test_build_index_populates_collection(workspace_tmp):
    """El pipeline completo deja los fragmentos indexados."""

    assert _run_build(workspace_tmp) == 0

    collection = _collection(workspace_tmp)

    assert collection.count() > 0

    stored = collection.get(
        include=["documents", "metadatas"]
    )

    # El contenido indexado proviene del documento en disco.
    assert "Plan Comunal de Emergencia" in stored["documents"][0]

    # La metadata conserva la trazabilidad hacia el archivo de origen.
    metadata = stored["metadatas"][0]

    assert metadata["source"].endswith("plan-comunal.html")
    assert metadata["title"] == "plan-comunal"
    assert "document_id" in metadata
    assert "chunk_index" in metadata


def test_build_index_ignores_unsupported_formats(workspace_tmp):
    """Los archivos con formato no soportado no se indexan."""

    assert _run_build(workspace_tmp) == 0

    stored = _collection(workspace_tmp).get(
        include=["metadatas"]
    )

    sources = {
        metadata["source"]
        for metadata in stored["metadatas"]
    }

    assert len(sources) == 1

    assert not any(
        source.endswith(".txt")
        for source in sources
    )


def test_build_index_recreate_resets_collection(workspace_tmp):
    """--recreate reconstruye la colección sin acumular fragmentos."""

    assert _run_build(workspace_tmp) == 0

    first_count = _collection(workspace_tmp).count()

    assert first_count > 0

    assert _run_build(workspace_tmp, recreate=True) == 0

    second_count = _collection(workspace_tmp).count()

    # Sin duplicación: la reconstrucción produce la misma cantidad.
    assert second_count == first_count


def test_build_index_fails_on_missing_input(workspace_tmp):
    """Una ruta inexistente produce un código de salida de error."""

    exit_code = build_index.main(
        [
            "--input",
            str(workspace_tmp / "no-existe"),
            "--store",
            str(workspace_tmp / "vector_store"),
        ]
    )

    assert exit_code == 1


def test_collect_documents_filters_unsupported_files(workspace_tmp):
    """Sólo se consideran los formatos soportados."""

    source_dir = _write_documents(workspace_tmp)

    (source_dir / "anexo.docx").write_bytes(b"x")

    found = build_index.collect_documents(source_dir)

    assert [path.name for path in found] == [
        "anexo.docx",
        "plan-comunal.html",
    ]


def test_default_embedding_model_matches_runtime_settings():
    """El índice y la consulta deben usar el mismo modelo de embeddings.

    Si divergen, la búsqueda compara vectores generados por modelos
    distintos. No falla con un error: devuelve fragmentos sin relación
    con la consulta, que es un modo de fallo mucho más difícil de
    detectar. Por eso el valor se lee desde un único lugar.
    """

    from src import settings

    assert build_index.DEFAULT_EMBEDDING_MODEL == settings.EMBEDDING_MODEL
