"""
Archivo:
    conftest.py

Módulo:
    Configuración de las pruebas.

Objetivo:
    Proporcionar el directorio de trabajo temporal utilizado por las
    pruebas que necesitan escribir archivos en disco.

Descripción:
    Las fixtures ``tmp_path`` y ``tmp_path_factory`` de pytest crean
    directorios fuera del proyecto. En entornos de ejecución
    restringidos ese directorio puede no ser escribible para el usuario
    que ejecuta las pruebas, por lo que aquí se ofrece una fixture que
    crea el directorio temporal dentro del propio proyecto.

Proyecto:
    ALESSIA
    Asistente basado en RAG para la gestión del riesgo de desastres.
"""


import shutil
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def workspace_tmp() -> Path:
    """
    Directorio temporal escribible para una prueba.

    Returns
    -------
    Path
        Ruta de un directorio temporal vacío, eliminado al finalizar.
    """

    directory = (
        PROJECT_ROOT
        / ".pytest_work"
        / uuid.uuid4().hex
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    yield directory

    shutil.rmtree(
        directory,
        ignore_errors=True,
    )
