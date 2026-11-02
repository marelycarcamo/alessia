"""
Archivo: agent_service.py

Módulo:
    src.services

Objetivo:
    Proporcionar una capa de servicio que actúe como fachada del pipeline
    RAG de ALESSIA, coordinando la recuperación del contexto, el proceso
    de reranking y la generación de respuestas.

Descripción:
    Este módulo encapsula la interacción entre la capa de presentación
    y los componentes internos del sistema, evitando que la interfaz
    conozca los detalles de implementación del pipeline RAG.

    AgentService recibe una consulta del usuario y orquesta la ejecución
    de los módulos de Retrieval, Reranking y Generation para obtener una
    respuesta fundamentada en los documentos oficiales disponibles.

Proyecto:
    ALESSIA
    Asistente basado en RAG para la gestión del riesgo de desastres
    en la comuna de Valdivia.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection
from sentence_transformers import CrossEncoder, SentenceTransformer

from src import settings
from src.processing.generation import generate_answer
from src.processing.llm_client import LLMProviderError
from src.processing.reranker import rerank_chunks
from src.processing.retriever import retrieve_context
from src.processing.vector_store import get_or_create_collection
from src.services.conversation import (
    CONVERSATIONAL_ANSWERS,
    detect_conversational_intent,
    normalize_text,
)


@dataclass
class RAGAnswer:
    """
    Resultado de una consulta al pipeline RAG.

    Mantiene unida la respuesta generada y los fragmentos documentales
    utilizados durante su construcción, permitiendo que la capa de
    presentación muestre las fuentes consultadas.
    """

    answer: str
    sources: list[dict[str, Any]] = field(
        default_factory=list
    )


NO_CONTEXT_ANSWER = (
    "No encontré información sobre tu consulta en la "
    "documentación oficial disponible.\n\n"
    "Puedes intentar reformular la pregunta o consultar "
    "directamente con la Municipalidad de Valdivia. "
    "En caso de emergencia, comunícate con Bomberos (132), "
    "Carabineros (133) o SAMU (131)."
)


# --------------------------------------------------------------
# Respuesta de emergencia en curso
# --------------------------------------------------------------

# Palabras que nombran una amenaza concreta en desarrollo.
EMERGENCY_HAZARDS = (
    "incendio",
    "fuego",
    "humo",
    "inundacion",
    "tsunami",
    "terremoto",
    "temblor",
    "derrumbe",
    "remocion en masa",
)

# Marcadores de que la amenaza está ocurriendo ahora y no se pregunta
# de forma hipotética. Sin uno de estos, la consulta sigue el camino
# normal: "¿qué previene los incendios forestales?" menciona incendios
# pero no es una emergencia.
EMERGENCY_URGENCY_MARKERS = (
    "ahora",
    "urgente",
    "urgencia",
    "ayuda",
    "auxilio",
    "socorro",
    "mi casa",
    "estoy",
    "necesito",
    "en curso",
    "se esta",
)

# Respuesta determinista, sin modelo de lenguaje.
#
# Dos razones para no consultar al proveedor aquí:
#
# 1. Tiempo. Una respuesta medida tardó entre 40 y 113 segundos. En una
#    emergencia eso es inaceptable.
# 2. Disponibilidad. Si el proveedor falla, la respuesta de emergencia
#    no puede ser lo que se caiga: es la única que no debería depender
#    de un servicio externo.
#
# El contenido replica el protocolo del prompt, sin agregar
# recomendaciones que no estén en la documentación.
EMERGENCY_ANSWER = (
    "**Esto es una emergencia. Tu seguridad es lo más importante.**\n\n"
    "1. Aléjate del peligro y dirígete a un lugar seguro: en una "
    "inundación o tsunami, a un sector alto; en un incendio, fuera del "
    "área y contra el viento.\n"
    "2. No vuelvas a entrar a una vivienda dañada ni intentes cruzar "
    "agua en movimiento.\n"
    "3. Llama a los servicios de emergencia: **Bomberos 132**, "
    "**Carabineros 133**, **SAMU 131**.\n"
    "4. Sigue las instrucciones de las autoridades en terreno.\n"
    "5. Si es seguro, avisa a tus vecinos, en especial a personas "
    "mayores o con dificultad para moverse.\n\n"
    "ALESSIA es un asistente informativo. Para decisiones críticas, "
    "sigue las instrucciones de las autoridades en terreno."
)


def looks_like_emergency(question: str) -> bool:
    """
    Indica si la consulta describe una emergencia en curso.

    Es una heurística de palabras clave, no un clasificador: nombra la
    amenaza y un marcador de urgencia al mismo tiempo. Un modelo podría
    hacerlo mejor, pero agregaría latencia justo donde no la hay.
    """

    text = normalize_text(question)

    has_hazard = any(
        hazard in text
        for hazard in EMERGENCY_HAZARDS
    )

    has_urgency = any(
        marker in text
        for marker in EMERGENCY_URGENCY_MARKERS
    )

    return has_hazard and has_urgency


class AgentService:
    """
    Fachada del pipeline RAG.

    Esta clase coordina la ejecución del pipeline completo,
    desacoplando la interfaz de usuario de los módulos internos
    del sistema.
    """

    def __init__(
        self,
        collection: Collection,
        embedding_model: SentenceTransformer,
        reranker_model: CrossEncoder,
    ) -> None:
        """
        Inicializa el servicio con los componentes del pipeline.

        Parameters
        ----------
        collection : Collection
            Colección persistente de ChromaDB.

        embedding_model : SentenceTransformer
            Modelo utilizado para generar embeddings.

        reranker_model : CrossEncoder
            Modelo utilizado para realizar reranking.
        """

        self.collection = collection
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model

    @classmethod
    def create(cls) -> "AgentService":
        """
        Crea una instancia completamente configurada de AgentService.

        Returns
        -------
        AgentService
            Servicio inicializado.
        """

        project_root = Path(__file__).parent.parent.parent

        vector_store_path = (
            project_root
            / "data"
            / "vector_store"
        )

        collection = get_or_create_collection(
            collection_name="alessia_collection",
            vector_store_path=vector_store_path,
        )

        # Los modelos se leen de settings y no se escriben aquí: el
        # índice vectorial se construye con el mismo modelo, así que
        # ambos lados deben salir de una única fuente. Si divergen, la
        # búsqueda compara vectores de modelos distintos y no falla:
        # devuelve fragmentos sin relación, que es peor.
        embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL
        )

        reranker_model = CrossEncoder(
            settings.RERANKER_MODEL
        )

        return cls(
            collection=collection,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )

    def ask(
        self,
        question: str,
    ) -> RAGAnswer:
        """
        Ejecuta el pipeline RAG completo.

        Parameters
        ----------
        question : str
            Consulta realizada por el usuario.

        Returns
        -------
        RAGAnswer
            Respuesta generada por el modelo de lenguaje junto con los
            fragmentos documentales utilizados para construirla.

        Notes
        -----
        Si no se recupera contexto, o si el proveedor del modelo de
        lenguaje no está disponible, se retorna una respuesta
        informativa en lugar de propagar la excepción a la interfaz.

        Las consultas que describen una emergencia en curso se responden
        con un protocolo fijo, sin recuperar contexto ni llamar al
        modelo de lenguaje.

        Los mensajes conversacionales (saludos, identidad, capacidades,
        agradecimientos) se responden igual: de forma determinista y sin
        consultar la documentación.
        """

        # Se resuelve antes de recuperar contexto y antes de llamar al
        # modelo: una emergencia en curso no puede depender de que el
        # retrieval acierte ni de que el proveedor responda.
        if looks_like_emergency(question):

            return RAGAnswer(
                answer=EMERGENCY_ANSWER,
                sources=[],
            )

        # Los mensajes conversacionales tampoco necesitan documentos: un
        # saludo o una pregunta sobre la identidad del asistente no están
        # en ningún plan de emergencia, y buscarlos produce una negativa
        # que hace parecer que el sistema no entendió nada.
        intent = detect_conversational_intent(
            question,
            EMERGENCY_HAZARDS,
        )

        if intent is not None:

            return RAGAnswer(
                answer=CONVERSATIONAL_ANSWERS[intent],
                sources=[],
            )

        chunks = self._retrieve_context(question)

        # Sin contexto recuperado no se consulta al modelo: responder
        # sin documentación contradice el propósito del sistema.
        if not chunks:

            return RAGAnswer(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
            )

        ranked_chunks = self._rerank_chunks(
            question,
            chunks,
        )

        # Un fragmento puede pertenecer al tema correcto y aun así no
        # responder la consulta. Si ninguno alcanza el umbral, se prefiere
        # admitir que no hay información antes que construir una respuesta
        # sobre material que no la contiene: en una guía de emergencia, una
        # respuesta plausible pero vacía es peor que una negativa clara.
        relevant_chunks = self._filter_relevant_chunks(
            ranked_chunks
        )

        if not relevant_chunks:

            return RAGAnswer(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
            )

        try:
            answer = self._generate_answer(
                question,
                relevant_chunks,
            )

        except (ValueError, LLMProviderError) as error:

            # Se cubren los dos fallos posibles del proveedor: problemas
            # de configuración (ValueError, por ejemplo una credencial
            # ausente) y fallos del servicio (LLMProviderError, como un
            # timeout). En ambos casos el usuario necesita un mensaje
            # entendible, no un traceback.
            return RAGAnswer(
                answer=(
                    "ALESSIA no pudo generar la respuesta en este "
                    f"momento ({error}).\n\n"
                    "Revisa la configuración del servicio o intenta "
                    "nuevamente en unos minutos."
                ),
                sources=[],
            )

        return RAGAnswer(
            answer=answer,
            sources=self._build_sources(relevant_chunks),
        )

    def _build_sources(
        self,
        chunks,
    ) -> list[dict[str, Any]]:
        """
        Construye el detalle de los fragmentos utilizados como contexto.

        Permite mantener la trazabilidad entre la respuesta generada y
        la documentación oficial que la respalda.
        """

        sources = []

        for chunk in chunks:

            metadata = dict(chunk.metadata)

            sources.append(
                {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "page": metadata.get("page"),
                    "rerank_score": metadata.get("rerank_score"),
                    "rerank_probability": metadata.get(
                        "rerank_probability"
                    ),
                }
            )

        return sources

    def _retrieve_context(
        self,
        question: str,
    ):
        """
        Recupera los fragmentos más relevantes para la consulta.
        """

        return retrieve_context(
            query=question,
            collection=self.collection,
            model=self.embedding_model,
            k=5,
        )

    def _rerank_chunks(
        self,
        question: str,
        chunks,
    ):
        """
        Reordena los fragmentos recuperados según su relevancia.
        """

        return rerank_chunks(
            query=question,
            chunks=chunks,
            model=self.reranker_model,
            top_k=3,
        )

    def _filter_relevant_chunks(
        self,
        chunks,
    ):
        """
        Descarta los fragmentos que no alcanzan el umbral de relevancia.

        El umbral se expresa como probabilidad y no como puntaje crudo
        porque el puntaje del CrossEncoder no tiene escala interpretable:
        varía según el modelo y no es comparable entre consultas. La
        probabilidad sí permite fijar un valor con significado.
        """

        return [
            chunk
            for chunk in chunks
            if chunk.metadata.get(
                "rerank_probability",
                0.0,
            )
            >= settings.RERANK_MIN_PROBABILITY
        ]

    def _generate_answer(
        self,
        question: str,
        chunks,
    ) -> str:
        """
        Genera una respuesta utilizando el contexto recuperado.
        """

        return generate_answer(
            question=question,
            context=chunks,
        )