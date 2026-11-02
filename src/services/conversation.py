"""
Archivo:
    conversation.py

Módulo:
    src.services

Objetivo:
    Resolver los mensajes conversacionales antes de entrar al pipeline RAG.

Descripción:
    Un asistente documental recibe también mensajes que no son consultas:
    saludos, preguntas sobre su identidad, sobre lo que puede hacer, o un
    agradecimiento. Si esos mensajes se envían al pipeline, la búsqueda no
    encuentra nada parecido en los planes de emergencia y el sistema
    responde "no encontré información". Técnicamente es correcto y la
    experiencia es pésima: parece que no entendió nada.

    Aquí se detectan esas intenciones y se responde de forma determinista,
    sin recuperación y sin modelo de lenguaje. Es el mismo patrón que usa
    la ruta de emergencia: lo que no necesita documentos, no los consulta.

    La detección es deliberadamente conservadora. Un mensaje que nombra una
    amenaza se trata como consulta aunque empiece con un saludo, porque
    "hola, ¿qué hago si hay un tsunami?" es una consulta, no un saludo.

Proyecto:
    ALESSIA - Asistente de gestión del riesgo de desastres.
"""

import re
import unicodedata


# Identificadores de intención. Se usan como claves del diccionario de
# respuestas, así que un error de tipeo tiene que fallar al importar y no
# en tiempo de ejecución.
GREETING = "greeting"

IDENTITY = "identity"

CAPABILITIES = "capabilities"

THANKS = "thanks"

GOODBYE = "goodbye"


CONVERSATIONAL_ANSWERS: dict[str, str] = {

    GREETING: (
        "¡Hola! Soy ALESSIA. Respondo consultas con la documentación "
        "oficial de emergencia de la Municipalidad de Valdivia: planes "
        "comunales, anexos por amenaza, roles y protocolos.\n\n"
        "¿Qué necesitas saber?"
    ),

    IDENTITY: (
        "Soy ALESSIA, un asistente que responde con la documentación "
        "oficial de emergencia de la Municipalidad de Valdivia.\n\n"
        "No soy una persona ni tengo experiencia propia: busco en los "
        "planes comunales y te cito lo que dicen, indicando la fuente. "
        "Cuando algo no está en los documentos, te lo digo en vez de "
        "inventarlo.\n\n"
        "¿Qué necesitas saber?"
    ),

    CAPABILITIES: (
        "Puedo responderte sobre los planes de emergencia de la comuna de "
        "Valdivia: amenazas cubiertas (tsunami, incendio forestal, "
        "remoción en masa, inundación), roles de las instituciones, "
        "comités y protocolos de activación.\n\n"
        "No puedo responder sobre temas que no estén en esos documentos, "
        "ni dar opiniones ni información en tiempo real.\n\n"
        "¿Qué quieres consultar?"
    ),

    THANKS: (
        "De nada. Si te surge otra duda sobre los planes de emergencia, "
        "aquí estoy."
    ),

    GOODBYE: (
        "Hasta luego. Si es una emergencia en curso, llama al 132 "
        "(Bomberos), 133 (Carabineros) o 131 (SAMU)."
    ),
}


# Patrones que se comparan contra el mensaje normalizado.
#
# Los de saludo, agradecimiento y despedida exigen coincidencia completa:
# "hola" sí, "hola necesito ayuda" no. Los de identidad y capacidades se
# buscan dentro del texto, porque son preguntas que varían mucho de forma
# ("¿quién eres?", "¿tú quién eres?", "¿a qué te dedicas?").
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (

    (
        CAPABILITIES,
        re.compile(
            r"(que puedes hacer|para que sirves|en que me puedes ayudar"
            r"|que sabes hacer|que informacion tienes|a que te dedicas)\b"
        ),
    ),

    (
        IDENTITY,
        re.compile(
            r"(quien eres|quien sos|como te llamas|eres un bot"
            r"|eres un asistente|eres una persona|eres valdiviana"
            r"|de donde eres)\b"
        ),
    ),

    (
        GREETING,
        re.compile(
            r"^(hola|holi|buenas|buen dia|buenos dias|buenas tardes"
            r"|buenas noches|hey|alo)\b[\s!.,¿?]*$"
        ),
    ),

    (
        THANKS,
        re.compile(
            r"^(gracias|muchas gracias|mil gracias|te agradezco)"
            r"\b[\s!.,]*$"
        ),
    ),

    (
        GOODBYE,
        re.compile(
            r"^(chao|adios|hasta luego|nos vemos|bye)\b[\s!.,]*$"
        ),
    ),
)


def normalize_text(text: str) -> str:
    """
    Pasa el texto a minúsculas y sin tildes.

    Permite comparar sin depender de cómo escribió la persona. "¿Quién
    eres?" y "quien eres" tienen que caer en la misma intención.
    """

    decomposed = unicodedata.normalize(
        "NFKD",
        text.lower(),
    )

    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )


def detect_conversational_intent(
    question: str,
    hazards: tuple[str, ...],
) -> str | None:
    """
    Detecta si el mensaje es conversacional y no una consulta documental.

    Parámetros
    ----------
    question : str
        Mensaje recibido.

    hazards : tuple[str, ...]
        Palabras que nombran una amenaza. Si aparece alguna, el mensaje se
        trata como consulta: quien escribe "hola, ¿qué hago si hay un
        tsunami?" necesita información, no un saludo.

    Retorna
    -------
    str | None
        Identificador de la intención, o ``None`` si es una consulta.
    """

    text = normalize_text(question).strip()

    if not text:
        return None

    if any(
        hazard in text
        for hazard in hazards
    ):
        return None

    for intent, pattern in _PATTERNS:

        if pattern.search(text):
            return intent

    return None
