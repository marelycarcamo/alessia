"""
Archivo: settings.py

Módulo:
    Configuración (Configuration)

Objetivo:
    Centralizar la configuración del proyecto ALESSIA mediante
    variables de entorno.

Descripción:
    Este módulo carga las variables definidas en el archivo .env
    y las pone a disposición del resto del sistema. De esta forma,
    la configuración permanece desacoplada del código fuente y
    permite modificar parámetros sin necesidad de realizar cambios
    en la implementación.

    Este módulo SOLO lee valores: no instancia clientes, no escribe
    en consola y no realiza llamadas de red. La comunicación con los
    proveedores de modelos de lenguaje es responsabilidad exclusiva
    de src.processing.llm_client.

    Mantener esa separación importa: settings se importa desde toda
    la aplicación (incluidos los tests), así que cualquier efecto
    secundario aquí se ejecuta siempre y sin que nadie lo pida.

Proyecto:
    ALESSIA
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# --------------------------------------------------------------
# Cargar variables de entorno
# --------------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

load_dotenv(
    project_root / ".env"
)


# --------------------------------------------------------------
# Funciones auxiliares de configuración
# --------------------------------------------------------------

def _get_setting(
    key: str,
    default: str | None = None,
) -> str | None:
    """
    Obtiene una configuración desde el entorno disponible.

    Prioridad:
    1. Variables de entorno, utilizadas durante el desarrollo local.
    2. Secrets de Streamlit Cloud, utilizados en el despliegue.

    Parameters
    ----------
    key : str
        Nombre de la configuración.

    default : str | None
        Valor utilizado si la configuración no existe.

    Returns
    -------
    str | None
        Valor de la configuración.
    """

    value = os.getenv(
        key
    )

    if value:
        return value

    try:
        return st.secrets.get(
            key,
            default,
        )

    except Exception:
        # Fuera de Streamlit no existe archivo de secrets. No es un
        # error: simplemente se usa el valor por defecto.
        return default


def _get_float_setting(
    key: str,
    default: float,
) -> float:
    """
    Lee una configuración numérica decimal.

    Si el valor falta o no es convertible, se usa el valor por
    defecto en lugar de fallar al importar el módulo: una variable
    de entorno mal escrita no debería tumbar la aplicación.
    """

    raw_value = _get_setting(
        key
    )

    if raw_value is None:
        return default

    try:
        return float(raw_value)

    except (TypeError, ValueError):
        return default


def _get_int_setting(
    key: str,
    default: int,
) -> int:
    """Lee una configuración numérica entera, con el mismo criterio."""

    raw_value = _get_setting(
        key
    )

    if raw_value is None:
        return default

    try:
        return int(raw_value)

    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------
# Proveedor activo
# --------------------------------------------------------------

# El switch de proveedor se resuelve en tiempo de arranque, no durante
# la generación: el resto del pipeline no necesita saber cuál está activo.
DEFAULT_PROVIDER = (
    _get_setting(
        "DEFAULT_PROVIDER",
        "openrouter",
    )
    or "openrouter"
).strip().lower()


# --------------------------------------------------------------
# Configuración del proveedor LLM OPENROUTER
# --------------------------------------------------------------

OPENROUTER_API_KEY = _get_setting(
    "OPENROUTER_API_KEY"
)


# Permite apuntar a un endpoint alternativo (proxy o gateway propio).
# Si no se configura, el SDK utiliza el servicio oficial de OpenRouter.
OPENROUTER_API_BASE_URL = _get_setting(
    "OPENROUTER_API_BASE_URL"
)


OPENROUTER_MODEL = _get_setting(
    "OPENROUTER_MODEL",
    "openrouter/free",
)


# --------------------------------------------------------------
# Configuración del proveedor LLM NVIDIA (NIM)
# --------------------------------------------------------------

# NVIDIA expone una API compatible con OpenAI, por eso llm_client
# reutiliza el SDK de OpenAI apuntando a esta URL base.
NVIDIA_API_KEY = _get_setting(
    "NVIDIA_API_KEY"
)


NVIDIA_API_BASE_URL = (
    _get_setting(
        "NVIDIA_API_BASE_URL",
        "https://integrate.api.nvidia.com/v1",
    )
    or "https://integrate.api.nvidia.com/v1"
)


# El catálogo de NVIDIA lista modelos que luego no responden o que
# devuelven 404 según la cuenta. Si se cambia de modelo, conviene
# probarlo con una llamada real antes de desplegar.
NVIDIA_MODEL = (
    _get_setting(
        "NVIDIA_MODEL",
        "minimaxai/minimax-m3.1-flash",
    )
    or "minimaxai/minimax-m3.1-flash"
)


# --------------------------------------------------------------
# Modelos de recuperación (embeddings y reranking)
# --------------------------------------------------------------

# Modelo que transforma texto en vectores. Debe entender español.
#
# La documentación de ALESSIA es administrativa y local ("COGRID",
# "Mando Técnico", "anegamientos"), y un modelo entrenado solo en inglés
# degrada la recuperación aunque el texto se vea perfecto: los vectores
# quedan mal ubicados y los fragmentos relevantes no aparecen.
#
# ATENCIÓN: los vectores del índice se generan con este modelo. Si se
# cambia, hay que reconstruir el índice
# (`python scripts/build_index.py --recreate`). Si no, la búsqueda
# compara vectores de dos modelos distintos y devuelve fragmentos sin
# relación con la consulta, sin ningún error visible.
EMBEDDING_MODEL = (
    _get_setting(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# Modelo CrossEncoder que reordena los fragmentos recuperados.
# Debe estar entrenado en el idioma de los documentos, no en el de la
# interfaz: un reranker en inglés puntúa como irrelevantes fragmentos
# que sí responden a la consulta.
RERANKER_MODEL = (
    _get_setting(
        "RERANKER_MODEL",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )
    or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)


# --------------------------------------------------------------
# Umbral de relevancia
# --------------------------------------------------------------

# Probabilidad mínima que debe alcanzar un fragmento para ser usado como
# contexto. Por debajo, ALESSIA responde que no encontró información en
# lugar de construir una respuesta sobre material que no la contiene.
#
# 0.5 equivale a un puntaje de 0: es el punto donde el modelo deja de
# considerar el fragmento relevante, no un número ajustado a este
# corpus. Con la documentación actual la separación es amplia: el
# fragmento útil con menor probabilidad quedó en 0,60 y el mejor
# fragmento de una consulta sin respuesta, en 0,007.
#
# Si se agregan documentos, hay que recalibrar con
# `python scripts/evaluate_retrieval.py`.
RERANK_MIN_PROBABILITY = _get_float_setting(
    "RERANK_MIN_PROBABILITY",
    0.5,
)


# --------------------------------------------------------------
# Parámetros comunes de generación
# --------------------------------------------------------------

# Temperatura baja: las respuestas deben ceñirse a la documentación
# recuperada, no improvisar.
LLM_TEMPERATURE = _get_float_setting(
    "LLM_TEMPERATURE",
    0.2,
)


# Los modelos servidos por NVIDIA razonan antes de responder y esos
# tokens de razonamiento salen del mismo presupuesto de salida. Con un
# valor bajo, message.content llega vacío aunque la llamada sea exitosa.
#
# Valor medido con el prompt real de ALESSIA: el razonamiento consume
# entre 985 y 1.753 tokens según la corrida, y la respuesta entre 1.100
# y 1.550. Con 1024 la respuesta llega vacía; con 2048 una de tres
# corridas se truncó a mitad de frase. 4096 cubre el peor caso observado
# y no penaliza la latencia, porque el límite es un tope, no una cuota.
LLM_MAX_TOKENS = _get_int_setting(
    "LLM_MAX_TOKENS",
    4096,
)


# Tiempo máximo de espera por respuesta. Sin este límite, un proveedor
# colgado bloquea la interfaz hasta que el sistema operativo corta la
# conexión, que puede tardar minutos.
#
# Debe quedar por encima de la latencia real observada: una respuesta
# válida del modelo tardó entre 92 y 113 s, así que un límite de 120 s
# la descartaría por muy poco margen.
LLM_TIMEOUT_SECONDS = _get_float_setting(
    "LLM_TIMEOUT_SECONDS",
    180.0,
)
