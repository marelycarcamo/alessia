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


def test_build_prompt_includes_emergency_contacts():
    """El protocolo de emergencia debe llegar siempre con los teléfonos."""

    prompt = build_prompt("Pregunta", [])

    for telefono in ("132", "133", "131"):
        assert telefono in prompt


def test_build_prompt_forbids_external_knowledge():
    """La restricción de fidelidad documental debe estar en el prompt."""

    prompt = build_prompt("Pregunta", [])

    assert "No uses conocimiento" in prompt
    assert "CONTEXTO RECUPERADO" in prompt


def test_build_prompt_respects_token_budget():
    """Las instrucciones fijas no deben volver a inflarse.

    Cada sección del prompt viaja en todas las llamadas al modelo, así que
    su longitud se paga en tokens, latencia y costo. El presupuesto se fija
    en caracteres porque es una aproximación conservadora de tokens (4
    caracteres por token) y no requiere tokenizar en los tests.

    Si este test falla, la pregunta correcta no es "¿subo el límite?", sino
    "¿esta regla nueva repite algo que ya estaba escrito?".
    """

    prompt = build_prompt("Pregunta", [])

    assert len(prompt) < 3800


def test_prompt_no_permite_negar_el_corpus_completo():
    """El modelo no puede afirmar que la documentación no trata un tema.

    Solo recibe tres fragmentos, no el documento completo. Con la
    instrucción anterior ("si la respuesta no está en la documentación,
    dilo claramente") negó información que el plan de tsunami sí contenía:
    un vecino preguntó por el plan y la respuesta aseguró que no había
    zonas de evacuación ni puntos de encuentro, cuando el documento
    detalla 25 puntos de encuentro y más de 30 vías de evacuación.

    Una negativa falsa es tan peligrosa como una invención, y más difícil
    de detectar porque suena prudente.
    """

    prompt = build_prompt("Pregunta", [])

    assert "fragmentos seleccionados" in prompt
    assert "sin\n   afirmar que la documentación no lo trata" in prompt