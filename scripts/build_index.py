"""
Archivo:
    build_index.py

Módulo:
    scripts

Objetivo:
    Reconstruir el índice vectorial de ALESSIA a partir de los
    documentos oficiales almacenados en disco.

Descripción:
    Ejecuta el pipeline de ingesta completo:

        carga -> limpieza -> chunking -> embeddings -> ChromaDB

    Reproduce las etapas previas a la consulta, de modo que la base de
    conocimiento pueda regenerarse sin depender de un notebook o de un
    índice previamente versionado.

    Nota:
        Los fragmentos reciben identificadores nuevos en cada ejecución.
        Para reconstruir el índice desde cero utiliza ``--recreate``;
        de lo contrario, volver a indexar los mismos documentos
        acumularía fragmentos repetidos.

Uso:
    python scripts/build_index.py
    python scripts/build_index.py --input data/raw/muni_valdivia/planes
    python scripts/build_index.py --recreate

Proyecto:
    ALESSIA - Asistente de gestión del riesgo de desastres.
"""


import argparse
import sys
from pathlib import Path
from typing import Sequence

# Permite ejecutar el script directamente desde la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer  # noqa: E402

from src import settings  # noqa: E402
from src.ingestion.loaders import load_file  # noqa: E402
from src.processing.chunker import create_chunks  # noqa: E402
from src.processing.cleaner import clean_document  # noqa: E402
from src.processing.embedder import generate_embeddings  # noqa: E402
from src.processing.vector_store import index_chunks  # noqa: E402

import chromadb  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "muni_valdivia" / "planes"

DEFAULT_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

DEFAULT_COLLECTION = "alessia_collection"

# El modelo de embeddings debe coincidir con el que usa el pipeline en
# tiempo de consulta (ver src/settings.py). Se toma de allí para que no
# existan dos valores que puedan divergir.
DEFAULT_EMBEDDING_MODEL = settings.EMBEDDING_MODEL

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Define y procesa los argumentos de línea de comandos.

    Returns
    -------
    argparse.Namespace
        Argumentos validados.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Reconstruye el índice vectorial de ALESSIA a partir de "
            "documentos oficiales en disco."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directorio o archivo con los documentos oficiales "
            f"(por defecto: {DEFAULT_INPUT_DIR})."
        ),
    )

    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE_DIR,
        help=(
            "Directorio de persistencia de ChromaDB "
            f"(por defecto: {DEFAULT_STORE_DIR})."
        ),
    )

    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Nombre de la colección (por defecto: {DEFAULT_COLLECTION}).",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Tamaño máximo de cada fragmento en caracteres.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Caracteres compartidos entre fragmentos consecutivos.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Modelo de embeddings a utilizar "
            f"(por defecto: {DEFAULT_EMBEDDING_MODEL})."
        ),
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Elimina la colección antes de indexar. Necesario si el "
            "índice debe reconstruirse desde cero."
        ),
    )

    return parser.parse_args(argv)


def collect_documents(input_path: Path) -> list[Path]:
    """
    Obtiene la lista de documentos soportados a procesar.

    Parameters
    ----------
    input_path : Path
        Archivo o directorio de entrada.

    Returns
    -------
    list[Path]
        Rutas ordenadas de los documentos encontrados.

    Raises
    ------
    FileNotFoundError
        Si la ruta indicada no existe.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe la ruta indicada: {input_path}"
        )

    if input_path.is_file():
        return [input_path]

    documents = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    return documents


def relative_source(document_path: Path) -> str:
    """
    Expresa la ruta de un documento de forma portable.

    Se prefiere la ruta relativa a la raíz del proyecto para no exponer
    rutas absolutas; si el documento está fuera del proyecto se utiliza
    su ruta absoluta.

    Parameters
    ----------
    document_path : Path
        Ruta del documento indexado.

    Returns
    -------
    str
        Ruta utilizable como metadato de trazabilidad.
    """

    try:
        return str(
            document_path.relative_to(PROJECT_ROOT)
        )

    except ValueError:
        return str(document_path)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Ejecuta el proceso completo de construcción del índice.

    Returns
    -------
    int
        Código de salida del proceso.
    """

    args = parse_args(argv)

    try:
        documents = collect_documents(args.input)

    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        return 1

    if not documents:
        print(
            f"[ERROR] No se encontraron documentos soportados en {args.input}"
        )
        return 1

    print(f"[1/5] Documentos encontrados: {len(documents)}")

    model = SentenceTransformer(args.model)

    args.store.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(args.store))

    if args.recreate:
        try:
            client.delete_collection(args.collection)
            print(f"[2/5] Colección '{args.collection}' eliminada.")
        except Exception:
            print(f"[2/5] La colección '{args.collection}' no existía.")

    collection = client.get_or_create_collection(
        name=args.collection
    )

    print(f"[3/5] Colección activa: {args.collection}")

    total_chunks = 0

    for document_path in documents:

        document = load_file(document_path)

        # La metadata permite mantener la trazabilidad hacia el archivo
        # de origen durante la recuperación.
        document.metadata["source"] = relative_source(
            document_path
        )
        document.metadata["title"] = document.title

        cleaned = clean_document(document)

        chunks = create_chunks(
            cleaned,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        embedded = generate_embeddings(
            chunks,
            model,
        )

        index_chunks(
            chunks=embedded,
            collection=collection,
        )

        total_chunks += len(embedded)

        print(
            f"       {document_path.name}: {len(embedded)} fragmentos"
        )

    print(f"[4/5] Fragmentos indexados: {total_chunks}")

    print(f"[5/5] Total en la colección: {collection.count()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
