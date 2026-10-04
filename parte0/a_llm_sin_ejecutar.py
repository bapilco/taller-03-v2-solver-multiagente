#!/usr/bin/env python3
"""Parte 0.a — El solver que no ejecuta. Con la H200 (VPN).

    python parte0/a_llm_sin_ejecutar.py
    python parte0/a_llm_sin_ejecutar.py --solo-medir     # sin red: solo la medición real

Le da al LLM el enunciado de la Tarea A completo y le pide el reporte, sin herramientas.
Después corre el experimento de verdad —diez líneas de scikit-learn, un segundo de CPU— y
pone una cifra al lado de la otra.

El punto no es si el modelo acierta. Puede acercarse: el conjunto de datos es famoso y sus
exactitudes están en mil tutoriales. El punto es la **procedencia**: ninguna de las cifras
del reporte sale de una ejecución, y el reporte no lo dice. Un solver sin ejecutor no
resuelve la tarea: escribe un texto con la forma de la solución.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

import pymupdf

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI))
PDF = AQUI / "enunciados" / "tarea-a-generativo-discriminativo.pdf"


def medir() -> dict[str, float]:
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    salida = {}
    for nombre, m in [("NB", GaussianNB()),
                      ("LR", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)))]:
        p = m.fit(Xtr, ytr).predict(Xte)
        salida[f"{nombre} accuracy"] = accuracy_score(yte, p)
        salida[f"{nombre} F1 macro"] = f1_score(yte, p, average="macro")
    return salida


def cifras(texto: str) -> list[str]:
    # Fuera los bloques de código: ahí van semillas e hiperparámetros, no resultados.
    texto = re.sub(r"```.*?```", "", texto, flags=re.S)
    return re.findall(r"(?<![\w.])(?:0[.,]\d{2,4}|\d{2}[.,]\d{1,2}\s?%)", texto)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-medir", action="store_true")
    args = ap.parse_args()

    reales = medir()
    print("Medido ejecutando el código (sklearn, semilla 42):")
    for k, v in reales.items():
        print(f"   {k:<14} {v:.4f}")
    if args.solo_medir:
        return 0

    from h200 import H200
    llm = H200()
    enunciado = unicodedata.normalize("NFKC", "\n".join(p.get_text() for p in pymupdf.open(PDF)))
    # Sin «no tienes intérprete», el modelo intenta EJECUTAR el experimento en su cabeza:
    # el 2026-10-02 razonó 40 000 tokens (137 000 caracteres) y no escribió nada.
    r = llm.chat([{"role": "system", "content": "Eres un estudiante de maestría en IA. No "
                   "tienes intérprete de Python: escribe el reporte directamente, en menos "
                   "de 1000 palabras."},
                  {"role": "user", "content": "Resuelve esta tarea y entrega el reporte "
                   "completo, con todas las cifras:\n\n" + enunciado}], max_tokens=40000)
    if r["fin"] == "length" and not r["contenido"].strip():
        # El razonamiento se cobra del mismo cupo: medido el 2026-10-02, con 16 384 tokens
        # el modelo se quedó pensando y devolvió un `content` vacío.
        print(f"\nEl modelo agotó max_tokens razonando ({len(r['razonamiento'])} car.) "
              "y no escribió nada: sube max_tokens.")
        return 1
    reporte = r["contenido"]
    salida = AQUI / "parte0" / "reporte_sin_ejecutar.md"
    salida.write_text(reporte, encoding="utf-8")
    encontradas = cifras(reporte)
    print(f"\nLLM sin herramientas: {llm.modelo}, {date.today()}, "
          f"{r['uso'].get('completion_tokens', '?')} tokens de salida, {r['latencia_s']} s, "
          f"fin={r['fin']}")
    print(f"   reporte en {salida.relative_to(AQUI)}; {len(encontradas)} cifras con decimales:")
    print("   " + ", ".join(encontradas[:20]) + (" …" if len(encontradas) > 20 else ""))
    print("\n   Procedencia: 0 de", len(encontradas), "cifras salen de una ejecución "
          "(no hubo ninguna). Compáralas con las medidas arriba.")
    print("   ¿Dice el reporte que las cifras son estimadas?",
          "sí" if re.search(r"estimad|aproximad|ilustrativ|hipot", reporte, re.I) else "no")
    (AQUI / "parte0" / "medido.json").write_text(json.dumps(reales, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
