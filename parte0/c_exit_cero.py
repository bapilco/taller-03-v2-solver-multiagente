#!/usr/bin/env python3
"""Parte 0.c — Un `returncode == 0` no es un experimento. Sin modelo, sin red.

    python parte0/c_exit_cero.py

Un solver multiagente cierra el ciclo programar → ejecutar → evaluar, y el eslabón que
decide si se reintenta es el evaluador. El evaluador más barato pregunta dos cosas: ¿el
proceso terminó con código 0? ¿escribió el archivo de resultados? Este script le entrega
el código que un programador LLM escribe con frecuencia para la Parte 2 de la Tarea A —un
árbol de decisión que se evalúa **sobre los mismos datos con que se entrenó**— y deja que
ese evaluador lo juzgue.

Después corre dos comprobaciones que no necesitan modelo: una estática (¿se predice sobre
la misma variable con que se ajustó?) y una de plausibilidad (¿una exactitud de 1,000 en
un problema con ruido?). Las dos son código, y por eso no se pueden convencer.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Lo que un «programador» guionizado entrega. Corre, no falla y miente.
CODIGO = '''
import json
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
modelo = DecisionTreeClassifier(random_state=42).fit(X, y)
pred = modelo.predict(X)
acc = accuracy_score(y, pred)
print(f"Exactitud: {acc:.3f}")
json.dump({"modelo": "arbol", "accuracy": acc}, open("resultados.json", "w"))
'''


def evaluador_ingenuo(proc: subprocess.CompletedProcess, trabajo: Path) -> tuple[bool, str]:
    ok = proc.returncode == 0 and (trabajo / "resultados.json").exists()
    return ok, f"returncode={proc.returncode}, resultados.json={'sí' if ok else 'no'}"


def evalua_sobre_entrenamiento(codigo: str) -> list[str]:
    """Las variables que aparecen como X en `.fit(X, …)` y otra vez en `.predict(X)` o
    `.score(X, …)`. No prueba que haya fuga —una validación cruzada bien hecha puede
    reutilizar nombres—, pero obliga a mirar."""
    ajustadas, evaluadas = set(), []
    for nodo in ast.walk(ast.parse(codigo)):
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute) and nodo.args:
            primero = nodo.args[0]
            if not isinstance(primero, ast.Name):
                continue
            if nodo.func.attr in {"fit", "fit_transform"}:
                ajustadas.add(primero.id)
            elif nodo.func.attr in {"predict", "predict_proba", "score"}:
                evaluadas.append(primero.id)
    return sorted({v for v in evaluadas if v in ajustadas})


def implausibles(resultados: dict, techo: float = 0.999) -> list[str]:
    return [f"{k}={v}" for k, v in resultados.items()
            if isinstance(v, (int, float)) and k.lower().startswith(("acc", "f1", "exact"))
            and v >= techo]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="parte0c-") as tmp:
        trabajo = Path(tmp)
        (trabajo / "experimento.py").write_text(CODIGO)
        proc = subprocess.run([sys.executable, "experimento.py"], cwd=trabajo,
                              capture_output=True, text=True, timeout=120)
        print("stdout del experimento:", proc.stdout.strip())
        ok, motivo = evaluador_ingenuo(proc, trabajo)
        print(f"\n1) Evaluador ingenuo: {'APROBADO' if ok else 'RECHAZADO'}  ({motivo})")

        resultados = json.loads((trabajo / "resultados.json").read_text())
        fugas = evalua_sobre_entrenamiento(CODIGO)
        raras = implausibles(resultados)
        print("\n2) Evaluador con dos comprobaciones de código:")
        print(f"   estática  — se predice sobre lo mismo que se ajustó: {fugas or 'no'}")
        print(f"   plausible — métricas ≥ 0,999 en un problema con ruido: {raras or 'no'}")
        veredicto = "RECHAZADO" if fugas or raras else "APROBADO"
        print(f"   → {veredicto}: el programador recibe esto como retroalimentación y reintenta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
