"""
Module: cleaner.py

Purpose
-------
Clean and normalize document text before chunking.

Responsabilidades
----------------
- Remove unnecessary whitespace.
- Normalize line breaks.
- Remove page headers and footers that repeat across pages.
- Return a cleaned Document.

This module DOES NOT:
- Download files.
- Read documents.
- Split text into chunks.
- Generate embeddings.

Proyecto: ALESSIA
"""

import re
from collections import Counter

from src.models.document import Document


# Una línea que aparece al menos cinco veces en un mismo documento no es
# contenido: es cabecera, pie de página o marca de agua. Se exige un
# mínimo alto para no borrar encabezados de sección que se repiten dos
# o tres veces de forma legítima.
HEADER_LINE_MIN_REPETITIONS = 5

# Las cabeceras son cortas. Una línea larga que se repite suele ser un
# párrafo real duplicado por un error de extracción, y borrarlo perdería
# información.
HEADER_LINE_MAX_LENGTH = 120

# "1 de 33", "12/33": numeración de página que no aporta contenido.
PAGE_NUMBER_PATTERN = re.compile(
    r"^\d{1,3}\s*(?:de|/)\s*\d{1,3}$"
)


def _normalize_line(line: str) -> str:
    """
    Normaliza una línea para poder compararla con otras.

    Sin esto, "Página  3" y "pagina 3" se consideran distintas y la
    detección de cabeceras falla justo donde el PDF es irregular.
    """

    return re.sub(
        r"\s+",
        " ",
        line,
    ).strip().lower()


def remove_repeated_lines(
    text: str,
    min_repetitions: int = HEADER_LINE_MIN_REPETITIONS,
    max_length: int = HEADER_LINE_MAX_LENGTH,
) -> str:
    """
    Elimina líneas repetidas que corresponden a cabeceras o pies de página.

    En documentos oficiales, la cabecera institucional aparece en cada
    página. Al fragmentar el texto, esa cabecera queda dentro de decenas
    de fragmentos y compite con el contenido real durante la búsqueda:
    es texto genérico que se parece a cualquier consulta y nunca responde
    ninguna.

    Las viñetas no se eliminan aunque se repitan: en una lista de
    contactos, "• Bomberos" puede aparecer legítimamente varias veces.

    Parámetros
    ----------
    text : str
        Texto del documento.

    min_repetitions : int
        Cantidad mínima de apariciones para considerar una línea ruido.

    max_length : int
        Longitud máxima de una línea para ser considerada cabecera.

    Retorna
    -------
    str
        Texto sin las líneas repetidas.
    """

    lines = text.split("\n")

    candidates = [
        _normalize_line(line)
        for line in lines
        if line.strip()
        and not line.lstrip().startswith("•")
    ]

    counts = Counter(candidates)

    repeated = {
        line
        for line, count in counts.items()
        if count >= min_repetitions
        and len(line) <= max_length
    }

    kept_lines = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            kept_lines.append(line)
            continue

        if line.lstrip().startswith("•"):
            kept_lines.append(line)
            continue

        if PAGE_NUMBER_PATTERN.match(stripped):
            continue

        if _normalize_line(line) in repeated:
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines)


def clean_document(document: Document) -> Document:
    """
    Clean the textual content of a Document.

    Args:
        document (Document): Document to clean.

    Returns:
        Document: New Document with cleaned text.
    """

    text = document.content

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Se ejecuta antes de normalizar espacios: la detección compara
    # líneas completas, y colapsar espacios primero volvería idénticas
    # líneas que en el original no lo eran.
    text = remove_repeated_lines(text)

    # Collapse multiple blank lines into one
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Replace multiple spaces or tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return Document(

        id=document.id,
        title=document.title,
        source=document.source,
        file_type=document.file_type,
        content=text,
        metadata=document.metadata.copy()
    )