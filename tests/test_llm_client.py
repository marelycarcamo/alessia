"""
Archivo:
    test_llm_client.py

Módulo:
    Tests del cliente LLM.

Objetivo:
    Validar que la configuración del proveedor (modelo, temperatura,
    límite de tokens y endpoint) se aplique efectivamente al enviar el
    prompt, y que los fallos de configuración y de servicio se informen
    con excepciones distinguibles.

Descripción:
    Los proveedores reales se reemplazan mediante mocks, de modo que las
    pruebas no realizan llamadas a la red.

    Cada prueba fija explícitamente DEFAULT_PROVIDER: el módulo lee el
    archivo .env al importarse, así que sin este patch el resultado
    dependería de la configuración local de quien ejecute la suite.

Proyecto:
    ALESSIA
    Asistente basado en RAG para la gestión del riesgo de desastres.
"""


from unittest.mock import MagicMock, patch

import pytest

from src.processing import llm_client


def _build_client_mock() -> MagicMock:
    """Construye un cliente simulado que retorna una respuesta válida."""

    client = MagicMock()

    client.chat.send.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="Respuesta simulada."
                )
            )
        ]
    )

    # El cliente se utiliza como context manager.
    client.__enter__.return_value = client
    client.__exit__.return_value = False

    return client


def _build_nvidia_client_mock(
    content: str = "Respuesta simulada.",
) -> MagicMock:
    """Construye un cliente NVIDIA simulado, también como context manager."""

    client = MagicMock()

    client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=content
                )
            )
        ]
    )

    client.__enter__.return_value = client
    client.__exit__.return_value = False

    return client


# --------------------------------------------------------------
# OpenRouter
# --------------------------------------------------------------

def test_generate_response_applies_configuration():
    """El modelo, la temperatura y la URL base se envían al proveedor."""

    client = _build_client_mock()

    with patch.object(
        llm_client, "DEFAULT_PROVIDER", "openrouter"
    ), patch.object(
        llm_client, "OpenRouter", return_value=client
    ) as mock_openrouter, patch.object(
        llm_client, "OPENROUTER_API_KEY", "clave-de-prueba"
    ), patch.object(
        llm_client, "OPENROUTER_API_BASE_URL", "https://gateway.local/v1"
    ), patch.object(
        llm_client, "OPENROUTER_MODEL", "modelo/configurado"
    ), patch.object(
        llm_client, "LLM_TEMPERATURE", 0.2
    ):

        answer = llm_client.generate_response("PROMPT DE PRUEBA")

    assert answer == "Respuesta simulada."

    mock_openrouter.assert_called_once_with(
        api_key="clave-de-prueba",
        server_url="https://gateway.local/v1",
    )

    client.chat.send.assert_called_once_with(
        model="modelo/configurado",
        messages=[
            {
                "role": "user",
                "content": "PROMPT DE PRUEBA",
            }
        ],
        temperature=0.2,
    )


def test_generate_response_omits_base_url_when_not_configured():
    """Sin URL base configurada se usa el endpoint oficial del SDK."""

    client = _build_client_mock()

    with patch.object(
        llm_client, "DEFAULT_PROVIDER", "openrouter"
    ), patch.object(
        llm_client, "OpenRouter", return_value=client
    ) as mock_openrouter, patch.object(
        llm_client, "OPENROUTER_API_KEY", "clave-de-prueba"
    ), patch.object(
        llm_client, "OPENROUTER_API_BASE_URL", None
    ):

        llm_client.generate_response("PROMPT")

    assert "server_url" not in mock_openrouter.call_args.kwargs


def test_generate_response_openrouter_without_api_key_raises():
    """Sin credenciales de OpenRouter se informa un error explícito."""

    with patch.object(
        llm_client, "DEFAULT_PROVIDER", "openrouter"
    ), patch.object(
        llm_client, "OPENROUTER_API_KEY", None
    ):

        with pytest.raises(ValueError) as error:

            llm_client.generate_response("PROMPT")

    assert "API Key" in str(error.value)


# --------------------------------------------------------------
# NVIDIA
# --------------------------------------------------------------

def test_generate_response_nvidia_applies_configuration():
    """El modelo, la temperatura y el límite de tokens llegan a NVIDIA."""

    client = _build_nvidia_client_mock()

    with patch.object(
        llm_client, "DEFAULT_PROVIDER", "nvidia"
    ), patch.object(
        llm_client, "OpenAI", return_value=client
    ) as mock_openai, patch.object(
        llm_client, "NVIDIA_API_KEY", "clave-de-prueba"
    ), patch.object(
        llm_client, "NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1"
    ), patch.object(
        llm_client, "NVIDIA_MODEL", "proveedor/modelo"
    ), patch.object(
        llm_client, "LLM_TEMPERATURE", 0.2
    ), patch.object(
        llm_client, "LLM_MAX_TOKENS", 1024
    ), patch.object(
        llm_client, "LLM_TIMEOUT_SECONDS", 60.0
    ):

        answer = llm_client.generate_response("PROMPT DE PRUEBA")

    assert answer == "Respuesta simulada."

    mock_openai.assert_called_once_with(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="clave-de-prueba",
        timeout=60.0,
    )

    client.chat.completions.create.assert_called_once_with(
        model="proveedor/modelo",
        messages=[
            {
                "role": "user",
                "content": "PROMPT DE PRUEBA",
            }
        ],
        temperature=0.2,
        max_tokens=1024,
    )


def test_generate_response_nvidia_without_api_key_raises():
    """Sin credenciales de NVIDIA se informa un error explícito."""

    with patch.object(
        llm_client, "DEFAULT_PROVIDER", "nvidia"
    ), patch.object(
        llm_client, "NVIDIA_API_KEY", None
    ):

        with pytest.raises(ValueError) as error:

            llm_client.generate_response("PROMPT")

    assert "NVIDIA" in str(error.value)


def test_generate_response_nvidia_raises_on_empty_content():
    """Una respuesta vacía se reporta como fallo, no como éxito.

    Los modelos de razonamiento pueden consumir todo el presupuesto de
    salida en tokens internos y devolver content vacío con la llamada
    en estado exitoso. Ese caso no debe llegar a la interfaz.
    """

    client = _build_nvidia_client_mock(content="")

    with patch.object(
        llm_client, "DEFAULT_PROVIDER", "nvidia"
    ), patch.object(
        llm_client, "OpenAI", return_value=client
    ), patch.object(
        llm_client, "NVIDIA_API_KEY", "clave-de-prueba"
    ):

        with pytest.raises(llm_client.LLMProviderError) as error:

            llm_client.generate_response("PROMPT")

    assert "vacía" in str(error.value)


def test_generate_response_nvidia_rejects_truncated_answer():
    """Una respuesta truncada no debe llegar al usuario como si estuviera completa.

    El consumo de razonamiento varía entre llamadas, así que agotar el
    presupuesto de salida a mitad de la respuesta es un caso real y no
    teórico: la respuesta parece buena pero puede cortar una instrucción
    de seguridad por la mitad.
    """

    client = _build_nvidia_client_mock(
        content="Hola, vecino. Evacúa de inmediato hacia las zo"
    )

    client.chat.completions.create.return_value.choices[0].finish_reason = "length"

    with patch.object(
        llm_client, "DEFAULT_PROVIDER", "nvidia"
    ), patch.object(
        llm_client, "OpenAI", return_value=client
    ), patch.object(
        llm_client, "NVIDIA_API_KEY", "clave-de-prueba"
    ):

        with pytest.raises(llm_client.LLMProviderError) as error:

            llm_client.generate_response("PROMPT")

    assert "truncó" in str(error.value)


def test_generate_response_nvidia_wraps_provider_errors():
    """Un error del SDK se traduce a LLMProviderError."""

    client = _build_nvidia_client_mock()

    client.chat.completions.create.side_effect = llm_client.OpenAIError(
        "conexión rechazada"
    )

    with patch.object(
        llm_client, "DEFAULT_PROVIDER", "nvidia"
    ), patch.object(
        llm_client, "OpenAI", return_value=client
    ), patch.object(
        llm_client, "NVIDIA_API_KEY", "clave-de-prueba"
    ):

        with pytest.raises(llm_client.LLMProviderError) as error:

            llm_client.generate_response("PROMPT")

    assert "conexión rechazada" in str(error.value)


# --------------------------------------------------------------
# Selección de proveedor
# --------------------------------------------------------------

def test_generate_response_rejects_unknown_provider():
    """Un proveedor no soportado produce un error explícito."""

    with patch.object(
        llm_client, "DEFAULT_PROVIDER", "proveedor-inexistente"
    ):

        with pytest.raises(ValueError) as error:

            llm_client.generate_response("PROMPT")

    assert "no soportado" in str(error.value)
