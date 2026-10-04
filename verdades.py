"""Las verdades que el golden set calcula y que no caben en una línea.

La Tarea C trae su corpus dentro del enunciado. Aquí se reproduce, tal cual, para que el
evaluador recalcule las métricas de referencia con los parámetros que el enunciado fija
(`TfidfVectorizer()` por defecto y `TruncatedSVD(n_components=4, random_state=0)`). Si el
enunciado cambia, esta lista cambia con él: la comprobación `corpus_coincide()` lo exige.
"""
from __future__ import annotations

from functools import lru_cache

CORPUS = {
    "d01": "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "d02": "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "d03": "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "d04": "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "d05": "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "d06": "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "d07": "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "d08": "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "d09": "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "d10": "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
}
CONSULTAS = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}


@lru_cache(maxsize=None)
def recuperacion(metodo: str) -> dict:
    """Hit@1, Hit@3, MRR y el rango del relevante por consulta, con `tfidf` o `lsa`."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    ids = list(CORPUS)
    vec = TfidfVectorizer().fit(CORPUS.values())
    D = vec.transform(CORPUS.values())
    Q = vec.transform([q for q, _ in CONSULTAS.values()])
    if metodo == "lsa":
        svd = TruncatedSVD(n_components=4, random_state=0).fit(D)
        D, Q = svd.transform(D), svd.transform(Q)
    sim = cosine_similarity(Q, D)
    rangos = {}
    for fila, (qid, (_, rel)) in zip(sim, CONSULTAS.items()):
        orden = [ids[i] for i in (-fila).argsort(kind="stable")]
        rangos[qid] = orden.index(rel) + 1
    n = len(rangos)
    return {"hit@1": sum(r <= 1 for r in rangos.values()) / n,
            "hit@3": sum(r <= 3 for r in rangos.values()) / n,
            "mrr": sum(1 / r for r in rangos.values()) / n,
            "rangos": rangos}


if __name__ == "__main__":
    for m in ("tfidf", "lsa"):
        print(m, recuperacion(m))
