"""
Archivo: test_prompt_builder.py

Módulo:
    tests

Objetivo:
    Validar el comportamiento del módulo prompt_builder, verificando que
    la construcción del prompt utilizado por ALESSIA incluya correctamente
    la consulta del usuario y el contexto recuperado desde el pipeline RAG.

Descripción:
    Este módulo contiene pruebas unitarias para asegurar que el generador
    de prompts construya instrucciones consistentes para el modelo de
    lenguaje, manteniendo la conexión entre la pregunta del usuario y la
    información obtenida desde los documentos oficiales.

    Los tests validan el comportamiento esperado del módulo, evitando
    depender de la estructura exacta del texto generado para facilitar
    futuras mejoras en las instrucciones del sistema.

Proyecto:
    ALESSIA
    Asistente basado en RAG para la gestión del riesgo de desastres
    en la comuna de Valdivia.
"""

from src.processing.prompt_builder import build_prompt
from src.models.chunk import Chunk


def test_build_prompt_contains_question():

    question = "¿Qué hacer ante una inundación?"

    prompt = build_prompt(
        question,
        []
    )

    assert question in prompt

def test_build_prompt_contains_context():

    chunk = Chunk(
        id="1",
        document_id="doc1",
        chunk_index=0,
        content="Mantenerse informado por canales oficiales."
    )

    prompt = build_prompt(
        "Pregunta",
        [chunk]
    )

    assert (
        "Mantenerse informado"
        in prompt
    )