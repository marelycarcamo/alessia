"""
Archivo: llm_client.py

Módulo:
Procesamiento (Processing)

Objetivo:
Centralizar la comunicación entre ALESSIA y los distintos
proveedores de modelos de lenguaje (LLMs).

Descripción:
Este módulo encapsula las llamadas a los modelos de lenguaje,
permitiendo cambiar de proveedor sin afectar el resto del
pipeline RAG.

Proyecto: ALESSIA
"""

# ------------------------------------------------------------------
# Versión 1.0
#
# Actualmente ALESSIA utiliza OpenRouter como proveedor principal.
#
# En futuras versiones se podrán incorporar nuevos proveedores
# como Hugging Face u Ollama sin modificar el pipeline RAG.
# ------------------------------------------------------------------


from openrouter import OpenRouter

from src.settings import (
    DEFAULT_PROVIDER,
    LLM_TEMPERATURE,
    OPENROUTER_API_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)


# --------------------------------------------------------------
# Proveedores disponibles
# --------------------------------------------------------------

def generate_response(prompt: str) -> str:
    """
    Genera una respuesta utilizando el proveedor LLM configurado.

    Parámetros
    ----------
    prompt : str
        Prompt construido para el modelo de lenguaje.

    Retorna
    -------
    str
        Respuesta generada por el modelo.
    """

    providers = {
        "openrouter": _generate_openrouter,
    }

    try:
        provider = providers[DEFAULT_PROVIDER]
    except KeyError as error:
        raise ValueError(
            f"Proveedor de LLM no soportado: {DEFAULT_PROVIDER}"
        ) from error

    return provider(prompt)


# --------------------------------------------------------------
# OpenRouter
# --------------------------------------------------------------

def _generate_openrouter(prompt: str) -> str:
    """
    Envía un prompt a OpenRouter utilizando el SDK oficial.

    El modelo, la temperatura y, cuando se configura, la URL base del
    servicio se obtienen desde la configuración del proyecto.
    """

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "No se encontró la API Key de OpenRouter."
        )

    # La URL base sólo se envía cuando está configurada, de modo que el
    # SDK utilice su endpoint oficial por defecto en caso contrario.
    client_options = {
        "api_key": OPENROUTER_API_KEY,
    }

    if OPENROUTER_API_BASE_URL:
        client_options["server_url"] = OPENROUTER_API_BASE_URL

    with OpenRouter(**client_options) as client:

        request_options = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        if LLM_TEMPERATURE is not None:
            request_options["temperature"] = LLM_TEMPERATURE

        response = client.chat.send(
            **request_options
        )

    return response.choices[0].message.content