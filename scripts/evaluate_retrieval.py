"""
Archivo:
    evaluate_retrieval.py

Módulo:
    scripts

Objetivo:
    Medir, de forma repetible, si la etapa de recuperación entrega
    fragmentos que realmente contienen la información consultada.

Descripción:
    Ejecuta un conjunto corto de consultas de referencia contra el índice
    vectorial real y reporta, para cada una, si el fragmento esperado
    llega al top-k **y supera el umbral de relevancia** que aplica el
    pipeline.

    Por qué existe este script: al cambiar modelos de embeddings o de
    reranking sin medir, es imposible distinguir una mejora real de una
    impresión. Hoy la comparación se hace a mano y con suerte.

    Alcance, para no mentirse: esto NO es un set de evaluación completo.
    Comprueba presencia de vocabulario propio de la respuesta esperada,
    no si la respuesta generada es correcta ni fiel. Sirve para detectar
    regresiones de recuperación, que es donde hoy está el problema.

Uso:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --embedding <modelo> --reranker <modelo>

    Si se indican modelos distintos de los configurados, el índice debe
    haber sido construido con ese mismo modelo de embeddings. Un índice
    con vectores de otro modelo devuelve resultados sin sentido.

Proyecto:
    ALESSIA - Asistente de gestión del riesgo de desastres.
"""

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

# Permite ejecutar el script directamente desde la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

from src import settings  # noqa: E402
from src.processing.reranker import rerank_chunks  # noqa: E402
from src.processing.retriever import retrieve_context  # noqa: E402
from src.processing.vector_store import get_or_create_collection  # noqa: E402


DEFAULT_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

DEFAULT_COLLECTION = "alessia_collection"

# Candidatos que se recuperan antes del reranking. Coincide con el valor
# usado por el pipeline en producción: medir con otro número daría una
# lectura que no corresponde al comportamiento real.
DEFAULT_CANDIDATES = 5

# Fragmentos que finalmente se envían al modelo de lenguaje.
DEFAULT_TOP_K = 3


# Cada consulta declara el vocabulario que debería aparecer en la
# respuesta esperada. Se usan fragmentos de palabra y no la palabra
# completa para no depender de plurales ni de tildes.
#
# Estas consultas se eligieron verificando que el contenido exista en el
# corpus, no por lo que un vecino preguntaría. La diferencia importa: una
# consulta que el documento no responde aparece como "FALLA" aunque el
# sistema esté funcionando bien, porque devuelve fragmentos del tema pero
# sin la respuesta.
#
# Limitación conocida: el set no incluye consultas fuera de alcance, así
# que no mide si el sistema inventa relevancia donde no la hay.
PROBES: list[dict[str, object]] = [
    {
        "question": "¿Quién convoca al COGRID Comunal?",
        "keywords": ["alcaldesa", "convocar"],
    },
    {
        "question": "¿Qué es el Comité Comunal de Emergencia?",
        "keywords": ["comite comunal"],
    },
    {
        "question": "¿Qué funciones tiene el Mando Técnico?",
        "keywords": ["mando tecnico"],
    },
    {
        "question": "¿Qué rol cumple SENAPRED en la emergencia?",
        "keywords": ["senapred"],
    },
    {
        "question": "¿Dónde se habilitan los albergues si hay una emergencia?",
        "keywords": ["albergue"],
    },
    {
        "question": "¿Qué amenazas considera el plan comunal?",
        "keywords": ["tsunami", "remocion", "inundacion"],
    },
    {
        # La consulta que originó esta etapa. Antes fallaba porque el plan
        # de incendios no estaba indexado; ahora el corpus sí la responde.
        "question": "¿Qué medidas de seguridad previenen los incendios forestales?",
        "keywords": ["preven"],
    },
]


def normalize(text: str) -> str:
    """
    Pasa el texto a minúsculas y elimina tildes.

    Permite comparar vocabulario sin depender de acentos, que en los PDF
    oficiales aparecen y desaparecen según cómo se extrajo el texto.
    """

    decomposed = unicodedata.normalize(
        "NFKD",
        text.lower(),
    )

    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Define y procesa los argumentos de línea de comandos."""

    parser = argparse.ArgumentParser(
        description=(
            "Mide si la recuperación entrega fragmentos que contienen "
            "la información consultada."
        )
    )

    parser.add_argument(
        "--embedding",
        default=settings.EMBEDDING_MODEL,
        help=(
            "Modelo de embeddings con el que se consulta el índice "
            f"(por defecto: {settings.EMBEDDING_MODEL})."
        ),
    )

    parser.add_argument(
        "--reranker",
        default=settings.RERANKER_MODEL,
        help=(
            "Modelo de reranking a evaluar "
            f"(por defecto: {settings.RERANKER_MODEL})."
        ),
    )

    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE_DIR,
        help=f"Directorio del vector store (por defecto: {DEFAULT_STORE_DIR}).",
    )

    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Nombre de la colección (por defecto: {DEFAULT_COLLECTION}).",
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATES,
        help="Fragmentos recuperados antes del reranking.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Fragmentos que el pipeline enviaría al modelo de lenguaje.",
    )

    return parser.parse_args(argv)


def evaluate_probe(
    probe: dict[str, object],
    collection,
    embedding_model: SentenceTransformer,
    reranker_model: CrossEncoder,
    candidates: int,
    top_k: int,
) -> dict[str, object]:
    """
    Evalúa una consulta de referencia.

    Retorna si el vocabulario esperado apareció en el top-k **y** si ese
    fragmento supera el umbral de relevancia, junto con el detalle de
    cada uno.
    """

    question = str(probe["question"])

    keywords = [
        normalize(str(keyword))
        for keyword in probe["keywords"]  # type: ignore[union-attr]
    ]

    chunks = retrieve_context(
        query=question,
        collection=collection,
        model=embedding_model,
        k=candidates,
    )

    ranked = rerank_chunks(
        query=question,
        chunks=chunks,
        model=reranker_model,
        top_k=top_k,
    )

    details = []

    matched = False

    for chunk in ranked:

        content = normalize(chunk.content)

        probability = float(
            chunk.metadata.get("rerank_probability", 0.0)
        )

        hit = any(
            keyword in content
            for keyword in keywords
        )

        # Un acierto solo cuenta si el fragmento además supera el umbral
        # que aplica el pipeline. Sin esta condición el arnés da por
        # correctas consultas que en producción terminan en una negativa:
        # estaría midiendo recuperación, no lo que recibe el usuario.
        usable = (
            hit
            and probability >= settings.RERANK_MIN_PROBABILITY
        )

        matched = matched or usable

        details.append(
            {
                "score": chunk.metadata.get("rerank_score"),
                "probability": probability,
                "hit": hit,
                "usable": usable,
                "preview": chunk.content[:70].replace("\n", " "),
            }
        )

    best_score = (
        details[0]["score"]
        if details
        else None
    )

    return {
        "question": question,
        "matched": matched,
        "best_score": best_score,
        "details": details,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """
    Ejecuta la medición completa y reporta el resultado.

    Código de salida
    ----------------
    int
        0 si todas las consultas obtienen un fragmento utilizable, 1 si
        alguna falla. Permite usar el script como control de regresión.
    """

    args = parse_args(argv)

    print(f"embeddings : {args.embedding}")
    print(f"reranker   : {args.reranker}")

    collection = get_or_create_collection(
        collection_name=args.collection,
        vector_store_path=args.store,
    )

    print(f"fragmentos : {collection.count()}")
    print()

    embedding_model = SentenceTransformer(args.embedding)

    reranker_model = CrossEncoder(args.reranker)

    passed = 0

    for probe in PROBES:

        result = evaluate_probe(
            probe=probe,
            collection=collection,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            candidates=args.candidates,
            top_k=args.top_k,
        )

        estado = "OK  " if result["matched"] else "FALLA"

        passed += 1 if result["matched"] else 0

        print(f"[{estado}] {result['question']}")

        for detail in result["details"]:  # type: ignore[union-attr]

            # "*" acierta y sirve; "~" contiene el término pero no alcanza
            # el umbral, así que en producción no se usaría.
            if detail["usable"]:
                marca = "*"
            elif detail["hit"]:
                marca = "~"
            else:
                marca = " "

            print(
                f"        {marca} {detail['probability']:6.1%}  "
                f"{detail['preview']}"
            )

        print()

    print(
        f"recuperación utilizable: {passed}/{len(PROBES)} "
        f"(umbral {settings.RERANK_MIN_PROBABILITY:.0%})"
    )

    return 0 if passed == len(PROBES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
