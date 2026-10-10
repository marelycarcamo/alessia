"""
Archivo:
    test_llm_client.py

Módulo:
    Tests del cliente LLM.

Objetivo:
    Validar que la configuración del proveedor (modelo, temperatura y
    endpoint) se aplique efectivamente al enviar el prompt, y que la
    ausencia de credenciales produzca un error explícito.

Descripción:
    El proveedor real se reemplaza mediante mocks, de modo que las
    pruebas no realizan llamadas a la red.

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


def test_generate_response_applies_configuration():
    """El modelo, la temperatura y la URL base se envían al proveedor."""

    client = _build_client_mock()

    with patch.object(
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
        llm_client, "OpenRouter", return_value=client
    ) as mock_openrouter, patch.object(
        llm_client, "OPENROUTER_API_KEY", "clave-de-prueba"
    ), patch.object(
        llm_client, "OPENROUTER_API_BASE_URL", None
    ):

        llm_client.generate_response("PROMPT")

    assert "server_url" not in mock_openrouter.call_args.kwargs


def test_generate_response_without_api_key_raises():
    """Sin credenciales se informa un error explícito."""

    with patch.object(
        llm_client, "OPENROUTER_API_KEY", None
    ):

        with pytest.raises(ValueError) as error:

            llm_client.generate_response("PROMPT")

    assert "API Key" in str(error.value)


def test_generate_response_rejects_unknown_provider():
    """Un proveedor no soportado produce un error explícito."""

    with patch.object(
        llm_client, "DEFAULT_PROVIDER", "proveedor-inexistente"
    ):

        with pytest.raises(ValueError) as error:

            llm_client.generate_response("PROMPT")

    assert "no soportado" in str(error.value)
