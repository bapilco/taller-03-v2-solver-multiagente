#!/usr/bin/env python3
"""Parte 0.b — El RAG plano que pierde la referencia. Sin modelo, sin red.

    python parte0/b_rag_plano.py

Un enunciado no es una bolsa de párrafos: sus partes se citan entre sí. La Parte 3 de la
Tarea A dice «sobre la misma división de la Parte 1», y lo que esa división ES —el
conjunto de datos, la proporción, la semilla— solo está escrito en la Parte 1.

Este script hace lo que hace un RAG vectorial plano: parte el enunciado en fragmentos (uno
por sección), los indexa con TF-IDF y recupera los `k` más parecidos a la pregunta que un
subagente programador necesita contestar antes de escribir la curva de aprendizaje. Después
hace lo mismo **siguiendo una arista**: la mención «Parte 1» dentro de la Parte 3 es una
relación `depende_de`, y un recorrido de un salto por el grafo trae el fragmento que la
similitud nunca iba a traer.

Ningún modelo interviene: la falla es de la recuperación, no de la generación. Un LLM que
recibe el primer contexto inventa la partición o pregunta por ella; uno que recibe
el segundo la copia.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

AQUI = Path(__file__).resolve().parents[1]
PDF = AQUI / "enunciados" / "tarea-a-generativo-discriminativo.pdf"
PREGUNTA = "¿Con qué datos y con qué partición se calcula la curva de aprendizaje?"
# Lo que el programador necesita saber, y dónde está escrito.
HECHOS = {"conjunto de datos": "Breast Cancer", "proporción de prueba": "30 %",
          "prueba intocable": "no se toca"}
K = 2


def fragmentos(pdf: Path) -> dict[str, str]:
    texto = unicodedata.normalize("NFKC", "\n".join(p.get_text() for p in pymupdf.open(pdf)))
    trozos = re.split(r"\n(?=Parte \d+ —|El reporte\n)", texto)
    salida = {"Preámbulo": trozos[0]}
    for t in trozos[1:]:
        salida[t.splitlines()[0].split(" —")[0].strip()] = t
    return salida


def cubre(texto: str) -> str:
    texto = " ".join(texto.split())          # el PDF parte las líneas donde quiere
    return "  ".join(f"{'✓' if marca in texto else '✗'} {hecho}" for hecho, marca in HECHOS.items())


def main() -> int:
    if not PDF.exists():
        sys.exit("Falta el PDF: corre antes `python generar_enunciados.py`.")
    frag = fragmentos(PDF)
    nombres = list(frag)
    tfidf = TfidfVectorizer().fit(list(frag.values()) + [PREGUNTA])
    sim = cosine_similarity(tfidf.transform([PREGUNTA]), tfidf.transform(frag.values()))[0]
    orden = sorted(range(len(nombres)), key=lambda i: -sim[i])

    print(f"Pregunta del programador: «{PREGUNTA}»\n")
    print(f"1) RAG plano: los {K} fragmentos más similares (TF-IDF, coseno)")
    plano = [nombres[i] for i in orden[:K]]
    for i in orden[:K]:
        print(f"   {nombres[i]:<12} sim={sim[i]:.3f}")
    print(f"   El contexto cubre: {cubre(' '.join(frag[n] for n in plano))}\n")

    # Las aristas: cada «Parte N» que un fragmento menciona y que no es él mismo.
    aristas = {n: sorted({f"Parte {m}" for m in re.findall(r"Parte (\d+)", frag[n])} - {n})
               for n in nombres}
    print(f"2) Los mismos {K} fragmentos, más un salto por las aristas depende_de del grafo")
    grafo = list(plano)
    for semilla in plano:
        vecinos = [v for v in aristas[semilla] if v in frag]
        print(f"   {semilla} —depende_de→ {', '.join(vecinos) or '(ninguna)'}")
        grafo += [v for v in vecinos if v not in grafo]
    print(f"   El contexto cubre: {cubre(' '.join(frag[n] for n in grafo))}\n")

    print("Las aristas que el regex encontró en todo el enunciado:")
    for n, vs in aristas.items():
        if vs:
            print(f"   {n} → {', '.join(vs)}")
    def cuenta(nombres_: list[str]) -> int:
        texto = " ".join(" ".join(frag[n] for n in nombres_).split())
        return sum(m in texto for m in HECHOS.values())
    print(f"\nRAG plano: {cuenta(plano)} de {len(HECHOS)} hechos."
          f" Con el salto: {cuenta(grafo)} de {len(HECHOS)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
