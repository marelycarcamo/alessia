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

from src.processing.generation import generate_answer
from src.processing.reranker import rerank_chunks
from src.processing.retriever import retrieve_context
from src.processing.vector_store import get_or_create_collection


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

        embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
        """

        chunks = self._retrieve_context(question)

        # Sin contexto recuperado no se consulta al modelo: responder
        # sin documentación contradice el propósito del sistema.
        if not chunks:

            return RAGAnswer(
                answer=(
                    "No encontré información sobre tu consulta en la "
                    "documentación oficial disponible.\n\n"
                    "Puedes intentar reformular la pregunta o consultar "
                    "directamente con la Municipalidad de Valdivia. "
                    "En caso de emergencia, comunícate con Bomberos (132), "
                    "Carabineros (133) o SAMU (131)."
                ),
                sources=[],
            )

        ranked_chunks = self._rerank_chunks(
            question,
            chunks,
        )

        try:
            answer = self._generate_answer(
                question,
                ranked_chunks,
            )

        except ValueError as error:

            # Errores de configuración del proveedor (por ejemplo, la
            # ausencia de credenciales) se informan al usuario.
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
            sources=self._build_sources(ranked_chunks),
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