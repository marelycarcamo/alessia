
![ALESSIA](https://raw.githubusercontent.com/marelycarcamo/alessia/main/app/assets/images/alessia_img_2.png)
---

> **Versión actual: 1.1** — pipeline RAG con proveedor configurable, recuperación en español y control de relevancia.

<p align="center">
	<a href="https://alessia.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_PROBAR_ALESSIA-Streamlit-red?style=for-the-badge&logo=streamlit" alt="Probar ALESSIA">
  </a>
</p>

---

## 1. ¿PorquÉ nace ALESSIA?

<p align="center">
  <a href="https://www.youtube.com/watch?v=UZ_mJdIBY8s" target="_blank">
    <img src="https://img.youtube.com/vi/UZ_mJdIBY8s/0.jpg" alt="Ver video del proyecto" width="560" height="315">
  </a>
</p>

La gestión del riesgo de desastres requiere acceso oportuno a información confiable para apoyar la preparación, prevención y respuesta ante distintos escenarios de emergencia.

Las instituciones públicas generan una gran cantidad de documentación oficial, como planes comunales de emergencia, protocolos, normativas y recomendaciones. Sin embargo, esta información suele encontrarse distribuida en múltiples documentos, almacenada en formatos poco interactivos y escrita en un lenguaje técnico que puede dificultar su consulta rápida.

Ante una situación de necesidad, encontrar una respuesta específica puede requerir revisar manualmente extensos documentos, identificar la fuente correcta e interpretar información especializada.

**ALESSIA** nace para reducir esta brecha entre las personas y el conocimiento institucional disponible, facilitando el acceso conversacional a información oficial relacionada con la gestión del riesgo de desastres.


| Antes de ALESSIA                                        | Con ALESSIA                                                |
| ------------------------------------------------------- | ---------------------------------------------------------- |
| Buscar manualmente entre múltiples documentos oficiales | Consultar información mediante lenguaje natural            |
| Revisar extensos archivos PDF                           | Recuperar información relevante automáticamente            |
| Interpretar documentos técnicos                         | Recibir respuestas claras basadas en documentación oficial |
| Dificultad para identificar rápidamente la fuente       | Mantener trazabilidad hacia la información utilizada       |

>**ALESSIA** no reemplaza la información oficial ni la toma de decisiones de las autoridades. Su propósito es facilitar el acceso, comprensión y consulta del conocimiento institucional disponible.

---

## ¿Qué es ALESSIA?

**ALESSIA**: **Asistente Local de Enlace, Soluciones, Soporte e Inteligencia Autónoma**, es un asistente basado en Inteligencia Artificial Generativa y arquitectura RAG (Retrieval Augmented Generation), diseñado para facilitar el acceso a información oficial relacionada con la gestión del riesgo de desastres en la comuna de Valdivia.

A través de una interfaz conversacional, ALESSIA permite realizar consultas utilizando lenguaje natural y genera respuestas fundamentadas en documentación institucional previamente procesada, reduciendo la necesidad de revisar manualmente múltiples fuentes.

Su diseño prioriza:

* **Confiabilidad:** las respuestas se generan utilizando información proveniente de fuentes oficiales.
* **Trazabilidad:** el sistema mantiene relación entre las respuestas generadas y la documentación utilizada.
* **Responsabilidad:** **ALESSIA** informa y orienta, pero no reemplaza los protocolos oficiales ni la toma de decisiones de las autoridades.

---

## 3. Qué aporta ALESSIA?

**ALESSIA** facilita el acceso a información oficial relacionada con la gestión del riesgo de desastres mediante una experiencia conversacional respaldada por documentación institucional.

Sus principales aportes son:

* **Acceso simplificado a la información:** permite consultar documentos oficiales utilizando lenguaje natural, reduciendo el tiempo necesario para encontrar información relevante.
* **Respuestas fundamentadas:** genera respuestas utilizando únicamente el contexto recuperado desde documentación previamente procesada, priorizando la confiabilidad de la información entregada.
* **Mayor trazabilidad:** mantiene la relación entre las respuestas generadas y las fuentes documentales utilizadas, promoviendo la transparencia en el acceso a la información.
* **Arquitectura escalable:** su diseño modular permite incorporar nuevas capacidades, como memoria conversacional, integración con APIs oficiales y herramientas de apoyo a la gestión administrativa, sin modificar la estructura principal del sistema.

---

## 4. ¿Cuál es el proyecto detrás de ALESSIA?

**ALESSIA** es un proyecto concebido para evolucionar progresivamente desde un asistente documental basado en arquitectura RAG hacia una plataforma inteligente de apoyo a la gestión del riesgo de desastres.

La versión actual (1.1) consolida un pipeline completo de recuperación y generación de información sobre documentación oficial, con controles de calidad en la recuperación: modelos entrenados en español, umbral de relevancia y una ruta de respuesta inmediata para emergencias en curso. Sobre esta arquitectura se proyecta la incorporación de nuevas capacidades orientadas a mejorar la experiencia de los usuarios, fortalecer la trazabilidad de la información y facilitar la gestión del conocimiento institucional.

Su diseño modular permite incorporar futuras funcionalidades sin modificar la arquitectura principal del sistema, favoreciendo su evolución de manera gradual y mantenible.

| Versión | Enfoque                                                                                                            |
| ------- | ------------------------------------------------------------------------------------------------------------------ |
| **1.0** | Asistente documental basado en RAG.                                                                                |
| **1.1** | Robustez del motor: proveedor configurable, recuperación en español, umbral de relevancia y ruta de emergencia.    |
| **2.0** | Memoria conversacional, trazabilidad visible, metadatos e integración con APIs oficiales.                          |
| **3.0** | Plataforma de apoyo a emergencias con perfiles de usuario y administrador, y motor de protocolos basado en reglas. |

> **Nota:** Las funcionalidades descritas para las versiones 2.0 y 3.0 corresponden al roadmap de evolución del proyecto y no forman parte de la implementación actual.

---

## 5. Ciclo de interacción con ALESSIA

![ciclo de funcionamiento](docs/README/diagrams/diagrams_ciclo_conversacional.png)

---

## 6. ALESSIA en acción


### Captura 1: Saludo inicial.

![Screensot saludo inicial](docs/README/screenshots/screeshot_saludo_inicial.png)


---
## 7. Arquitectura del sistema

**ALESSIA** utiliza una arquitectura modular organizada en capas, donde cada componente tiene una responsabilidad específica. Esta separación permite mantener el sistema claro, facilitar las pruebas y favorecer su evolución futura.


### Capas de Arquitectura

![Capas de Arquitectura](docs/README/diagrams/diagrama_capas_de_arquitectura.png)


---
## 8. Flujo del pipeline RAG

![Flujo de Procesamiento RAG](docs/README/diagrams/diagrama_flujo_de_procesamiento_RAG.png)


---

## 9. Estructura del proyecto

La estructura del proyecto sigue una separación de responsabilidades que permite mantener el código modular, facilitar las pruebas y favorecer la evolución futura de ALESSIA.

```
alessia/
├── .streamlit/
├── app/
│   ├── assets/
│   │   ├── avatars/
│   │   │   ├── alessia_avatar.png
│   │   │   └── user_avatar.png
│   │   ├── images/
│   │   │   ├── alessia_img_1.png
│   │   │   ├── alessia_img_2.png
│   │   │   └── alessia_img_3.png
│   │   └── logo/
│   │       ├── alessia_logo_2.png
│   │       └── alessia_logo.png
│   ├── components/
│   │   └── chat.py
│   ├── bootstrap.py
│   ├── config.py
│   └── streamlit_app.py
├── data/
│   ├── processed/
│   │   └── Plan-Comunal-de-Emergencia-2025-2.json
│   ├── raw/
│   │   ├── muni_valdivia/
│   │   │   └── planes/
│   │   │       └── Plan-Comunal-de-Emergencia-2025-2.pdf
│   │   └── test/
│   │       └── test_plan_comunal.pdf
│   └── vector_store/
│       ├── f0875f3a-d239-4f67-88c2-75417ba9f54d/
│       │   ├── data_level0.bin
│       │   ├── header.bin
│       │   ├── length.bin
│       │   └── link_lists.bin
│       └── chroma.sqlite3
├── docs/
│   ├── README/
│   │   ├── diagrams/
│   │   │   ├── diagrama_capas_de_arquitectura.png
│   │   │   ├── diagrama_flujo_de_procesamiento_RAG.png
│   │   │   ├── diagrama_funciones_implementadas.png
│   │   │   ├── diagrama_roadmap.png
│   │   │   └── diagrams_ciclo_conversacional.png
│   │   ├── infograph/
│   │   │   └── IA_para_Gestión_de_Desastres.png
│   │   └── screenshots/
│   │       └── screeshot_saludo_inicial.png
│   ├── 01. La historia de ALESSIA.md
│   ├── 02. La arquitectura.md
│   ├── 03. Decisiones de diseño.md
│   ├── 04. Estrategia de pruebas.md
│   ├── 05. Despliegue.md
│   └── 06. Roadmap.md
├── notebooks/
│   ├── data/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_embeddings_vector_store.ipynb
│   ├── 03_retrieval.ipynb
│   ├── 04_reranking.ipynb
│   ├── 05_generation_validation.ipynb
│   └── 06_streamlit_app.ipynb
├── src/
│   ├── ingestion/
│   │   ├── loaders.py
│   │   └── scraper.py
│   ├── models/
│   │   ├── _init_.py
│   │   ├── chunk.py
│   │   └── document.py
│   ├── processing/
│   │   ├── chunker.py
│   │   ├── cleaner.py
│   │   ├── embedder.py
│   │   ├── generation.py
│   │   ├── llm_client.py
│   │   ├── prompt_builder.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── agent_service.py
│   └── settings.py
├── scripts/
│   ├── __init__.py
│   ├── build_index.py
│   └── evaluate_retrieval.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_agent_service.py
│   ├── test_bootstrap.py
│   ├── test_build_index.py
│   ├── test_chat.py
│   ├── test_chunk.py
│   ├── test_cleaner.py
│   ├── test_embedder.py
│   ├── test_generation_integration.py
│   ├── test_generation.py
│   ├── test_llm_client.py
│   ├── test_loaders.py
│   ├── test_prompt_builder.py
│   ├── test_reranker.py
│   ├── test_retriever.py
│   └── test_vector_store.py
├── .env.example
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt

```
---
## 10. Tecnologías utilizadas

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)[![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-5.6.0-yellow?logo=sentence_transformers)](https://www.sbert.net/)[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Database-purple?logo=chromadb)](https://www.trychroma.com/)[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM_API-black?logo=openrouter)](https://openrouter.ai/)[![NVIDIA](https://img.shields.io/badge/NVIDIA-LLM_API-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)[![Pytest](https://img.shields.io/badge/Pytest-9.1.1-green?logo=pytest)](https://docs.pytest.org/)




| Tecnología            | Uso en ALESSIA                                |
| --------------------- | --------------------------------------------- |
| Python                | Lenguaje principal                            |
| Streamlit             | Interfaz conversacional                       |
| Sentence Transformers | Embeddings y reranking                        |
| ChromaDB              | Almacenamiento vectorial                      |
| OpenRouter / NVIDIA   | Acceso al modelo de lenguaje (proveedor configurable) |
| Pytest                | Pruebas automatizadas                         |


---
## 11. Características implementadas (Versión 1.1)

![Infografía Gestión de Desastres](docs/README/infograph/IA_para_Gestión_de_Desastres.png)

---

### Funcionalidades existentes

ALESSIA cuenta actualmente con una versión 1.1 funcional que implementa un pipeline RAG completo para responder consultas utilizando información proveniente de documentos institucionales oficiales.

![Diagrama de Funcionalidades Implementadas 1.0](docs/README/diagrams/diagrama_funciones_implementadas.png)

### Novedades de la versión 1.1

* **Proveedor de lenguaje configurable** entre OpenRouter y NVIDIA desde el archivo `.env`, con detección de respuestas vacías y de respuestas truncadas, que son las más peligrosas porque parecen completas.
* **Recuperación en español:** los modelos de embeddings y de reranking están entrenados en el idioma de los documentos. Con modelos solo en inglés, el sistema no fallaba: simplemente dejaba de ordenar por utilidad.
* **Umbral de relevancia:** si ningún fragmento supera la probabilidad mínima configurada, ALESSIA responde que no encontró información en lugar de construir una respuesta sobre material que no la contiene.
* **Pertinencia visible:** las fuentes se muestran con su porcentaje de pertinencia, en lugar de un puntaje interno sin escala interpretable.
* **Limpieza de cabeceras repetidas:** las cabeceras y pies de página que se repiten en cada página se eliminan antes de fragmentar el texto, para que no compitan con el contenido real durante la búsqueda.
* **Ruta de emergencia inmediata:** las consultas que describen una emergencia en curso se responden con un protocolo fijo, sin depender del índice vectorial ni del proveedor del modelo.
* **Instrucciones del prompt reducidas** en cerca de un 58 %, con un test que impide que vuelvan a crecer sin control.
* **Suite de 53 pruebas** y `scripts/evaluate_retrieval.py` para medir la recuperación de forma repetible.

---

## 12. Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/marelycarcamo/alessia.git
cd alessia
```
### 2. Crear y activar el entorno virtual

En Windows:

```bash
py -m venv .venv
source .venv\Scripts\activate
```

En MacOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar la API KEY

Copiar el archivo de ejemplo y completar la credencial:

```bash
cp .env.example .env
```

Las variables disponibles están documentadas en `.env.example`. La credencial obligatoria depende del proveedor elegido en `DEFAULT_PROVIDER`:

```env
# Proveedor activo: "openrouter" o "nvidia"
DEFAULT_PROVIDER=nvidia

# Si DEFAULT_PROVIDER=openrouter
OPENROUTER_API_KEY=tu_api_key

# Si DEFAULT_PROVIDER=nvidia
NVIDIA_API_KEY=tu_api_key
NVIDIA_MODEL=minimaxai/minimax-m3.1-flash
```

### 5. Construir el índice vectorial

El índice (`data/vector_store/`) no se versiona: se genera a partir de los documentos oficiales de `data/raw/`.

```bash
python scripts/build_index.py --recreate
```

### 6. Ejecutar ALESSIA

```bash
streamlit run app/streamlit_app.py
```

### 7. Ejecutar las pruebas

```bash
python -m pytest
```

### 8. Medir la recuperación

```bash
python scripts/evaluate_retrieval.py
```

Ejecuta un conjunto corto de consultas de referencia contra el índice real e informa, para cada una, si el fragmento esperado llega al top-k y con qué pertinencia.

Es una prueba de humo, no una evaluación completa: sirve para detectar regresiones cuando se cambian modelos, documentos o la estrategia de fragmentación.

---

## 13. Roadmap

ALESSIA está diseñada para evolucionar progresivamente desde un asistente documental basado en RAG hacia una plataforma inteligente de apoyo a la gestión del riesgo de desastres.

![Roadmap](docs/README/diagrams/diagrama_roadmap.png)

---


## 14. Author & Licence

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Marely_Cárcamo-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marely/)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


![Logo ALESSIA](https://raw.githubusercontent.com/marelycarcamo/alessia/main/app/assets/logo/alessia_logo.png)
