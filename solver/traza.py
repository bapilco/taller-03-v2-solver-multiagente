"""traza.py — Regla 6 obligatoria: «Siempre se guarda la traza, también si algo falla: por
llamada, el agente, el modelo, los tokens de entrada y salida, la latencia y el error; por
ejecución, el código de salida, la duración y los archivos creados.»

Adaptado de `mini-opencode/traza.py` (mismo formato JSONL, un evento por línea) al
vocabulario de este solver: cada agente del pipeline (Lector, Indexador, Planificador,
Investigador, Programador, Ejecutor, Revisor, Redactor) llama a la misma instancia, así que
`resumen_por_agente()` es exactamente la tabla que pide la Parte 2 («qué agente gasta más
tokens y por qué», `resumen_solver.csv`).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


class Traza:
    def __init__(self, carpeta: Path, nombre_archivo: str = "traza.jsonl"):
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        self.ruta = carpeta / nombre_archivo
        self.eventos: list[dict] = []
        self._t0 = time.perf_counter()

    def __call__(self, tipo: str, agente: str, paso: int | None = None, **datos) -> None:
        """Mismo contrato que usan todos los agentes: `traza(tipo, agente, paso, **datos)`.
        Se escribe también si algo falla —no hay ninguna rama que la salte—, porque cada
        agente la llama directamente, no al final de una ejecución exitosa."""
        ev = {"t": round(time.perf_counter() - self._t0, 2),
             "hora": datetime.now().isoformat(timespec="seconds"),
             "tipo": tipo, "agente": agente, "paso": paso, **datos}
        self.eventos.append(ev)
        with self.ruta.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")

    def resumen_por_agente(self) -> list[dict]:
        """Una fila por agente: llamadas al LLM, tokens de entrada/salida, latencia
        acumulada, herramientas/ejecuciones y rechazos. La tabla base de `resumen_solver.csv`
        y de la Parte 5, pregunta 2 («¿qué agente gasta más tokens y por qué?»)."""
        filas: dict[str, dict] = {}
        for ev in self.eventos:
            f = filas.setdefault(ev["agente"], {"agente": ev["agente"], "llamadas_llm": 0,
                                                "tokens_entrada": 0, "tokens_salida": 0,
                                                "latencia_s": 0.0, "ejecuciones": 0,
                                                "rechazos": 0})
            if ev["tipo"] == "llm":
                f["llamadas_llm"] += 1
                f["tokens_entrada"] += ev.get("tokens_entrada", 0)
                f["tokens_salida"] += ev.get("tokens_salida", 0)
                f["latencia_s"] = round(f["latencia_s"] + ev.get("latencia_s", 0), 2)
            elif ev["tipo"] == "ejecucion":
                f["ejecuciones"] += 1
            elif ev["tipo"] in {"sandbox", "revision_codigo", "revision_llm", "validacion_plan",
                               "validacion_entregable"} and ev.get("aprobado") is False:
                f["rechazos"] += 1
        return [f for f in filas.values() if f["llamadas_llm"] or f["ejecuciones"]]

    def totales(self) -> dict:
        resumen = self.resumen_por_agente()
        return {"tokens_entrada": sum(f["tokens_entrada"] for f in resumen),
               "tokens_salida": sum(f["tokens_salida"] for f in resumen)}
