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

    El proveedor activo se selecciona con la variable de entorno
    DEFAULT_PROVIDER y se resuelve en el diccionario `providers`.
    El resto del pipeline sigue llamando a `generate_response` sin
    saber cuál proveedor está detrás.

Proyecto: ALESSIA
"""

from openai import OpenAI, OpenAIError
from openrouter import OpenRouter

from src.settings import (
    DEFAULT_PROVIDER,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    NVIDIA_API_BASE_URL,
    NVIDIA_API_KEY,
    NVIDIA_MODEL,
    OPENROUTER_API_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)


class LLMProviderError(RuntimeError):
    """
    Error al comunicarse con el proveedor de modelos de lenguaje.

    Se distingue de ValueError, que queda reservado para problemas de
    configuración (por ejemplo, una credencial ausente). Así la capa
    superior puede decidir qué mensaje mostrar al usuario: una cosa es
    "falta configurar la clave" y otra "el proveedor no respondió".

    Hereda de RuntimeError porque no es un error de programación:
    el código está bien escrito y el fallo es externo y transitorio.
    """


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

    Excepciones
    -----------
    ValueError
        Si el proveedor configurado no está soportado o falta su
        credencial.

    LLMProviderError
        Si el proveedor respondió con un error o con contenido vacío.
    """

    providers = {
        "openrouter": _generate_openrouter,
        "nvidia": _generate_nvidia,
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


# --------------------------------------------------------------
# NVIDIA (NIM)
# --------------------------------------------------------------

def _generate_nvidia(prompt: str) -> str:
    """
    Envía un prompt a NVIDIA NIM usando el SDK de OpenAI.

    NVIDIA expone una API compatible con OpenAI, así que se reutiliza
    el mismo SDK apuntando a su URL base. No se usa el SDK de OpenRouter
    aquí: son proveedores distintos con credenciales distintas.

    La llamada lleva tiempo máximo de espera y un presupuesto de salida
    holgado, porque los modelos de razonamiento consumen tokens antes
    de escribir la primera palabra de la respuesta.
    """

    if not NVIDIA_API_KEY:
        raise ValueError(
            "No se encontró la API Key de NVIDIA."
        )

    request_options = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    try:
        with OpenAI(
            base_url=NVIDIA_API_BASE_URL,
            api_key=NVIDIA_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
        ) as client:

            completion = client.chat.completions.create(
                **request_options
            )

    except OpenAIError as error:
        # Cubre timeouts, errores de conexión y respuestas HTTP 4xx/5xx.
        raise LLMProviderError(
            f"NVIDIA no respondió correctamente: {error}"
        ) from error

    content = (
        completion.choices[0].message.content
        or ""
    ).strip()

    finish_reason = completion.choices[0].finish_reason

    # Un modelo de razonamiento puede agotar el presupuesto de salida
    # razonando y devolver content vacío con finish_reason="length".
    # Sin esta guarda, la interfaz mostraría una respuesta en blanco
    # como si el sistema hubiera funcionado.
    if not content:
        raise LLMProviderError(
            "NVIDIA devolvió una respuesta vacía. Si el modelo razona "
            "antes de responder, aumenta LLM_MAX_TOKENS."
        )

    # Peor que una respuesta vacía es una respuesta cortada: parece
    # completa y puede dejar una instrucción de seguridad a medias.
    # El consumo de razonamiento varía entre llamadas, así que truncarse
    # no es un caso teórico.
    if finish_reason == "length":
        raise LLMProviderError(
            "NVIDIA truncó la respuesta por falta de presupuesto de "
            "salida. Aumenta LLM_MAX_TOKENS."
        )

    return content
