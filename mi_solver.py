#!/usr/bin/env python3
"""mi_solver.py

Esqueleto mínimo de Solver para probar `evaluar_solver.py`.

Implementa la clase Solver con el método `solve(pdf_path: str, salida: str) -> dict`.
- Copia el PDF de entrada a la carpeta `salida`.
- Escribe `resultados.json` con una métrica de ejemplo.
- Escribe `trace.jsonl` con una entrada de traza mínima.
- Devuelve el dict que exige el contrato del Taller.

Usar solo para pruebas rápidas/depuración; reemplazar por la implementación real.
"""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import traceback
from datetime import datetime


class Solver:
    """Esqueleto mínimo de Solver.

    El método `solve` debe ser no interactivo y escribir en la carpeta `salida` los
    archivos que declara como entregables. Aquí se generan `resultados.json` y
    `trace.jsonl` como ejemplos.
    """

    def solve(self, pdf_path: str, salida: str) -> dict:
        out = Path(salida)
        out.mkdir(parents=True, exist_ok=True)
        trace_path = out / "trace.jsonl"
        try:
            src = Path(pdf_path).resolve()
            dest_pdf = out / src.name
            # Copia el PDF de entrada como entregable de prueba
            shutil.copy2(src, dest_pdf)

            # Escritura de un JSON de resultados ficticio (para que el evaluador tenga algo que leer)
            resultados = {"accuracy": 0.5}
            (out / "resultados.json").write_text(json.dumps(resultados, ensure_ascii=False), encoding="utf-8")

            # Traza mínima en formato jsonl
            trace_entry = {
                "event": "stub-solve",
                "model": "local-stub",
                "pdf": str(src),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            trace_path.write_text(json.dumps(trace_entry, ensure_ascii=False) + "\n", encoding="utf-8")

            return {
                "status": "completado",
                "entregables": [str(dest_pdf.name)],
                "subtareas": [],
                "usage": {"tokens_entrada": 0, "tokens_salida": 0},
                "model": "local-stub",
                "trace": str(trace_path)
            }
        except Exception:
            (out / "excepcion.txt").write_text(traceback.format_exc(), encoding="utf-8")
            # asegurar que haya una traza aunque haya fallado
            trace_path.write_text(json.dumps({"event": "error", "timestamp": datetime.utcnow().isoformat() + "Z"}) + "\n",
                                     encoding="utf-8")
            return {
                "status": "fallido",
                "entregables": [],
                "subtareas": [],
                "usage": {},
                "model": "local-stub",
                "trace": str(trace_path)
            }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ejecuta el Solver esqueleto contra un PDF y una carpeta de salida.")
    ap.add_argument("pdf", help="ruta al PDF de enunciado")
    ap.add_argument("salida", help="carpeta de salida donde escribir entregables")
    args = ap.parse_args()

    s = Solver()
    res = s.solve(args.pdf, args.salida)
    print(json.dumps(res, indent=2, ensure_ascii=False))
