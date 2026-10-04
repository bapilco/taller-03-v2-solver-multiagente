"""planificador.py — el rol Planificador: divide el enunciado en subtareas con tipo y
dependencias. Taller 03 v2, Parte 1, Regla 1 obligatoria:

    «El plan se valida con código: ids únicos, dependencias que existen y sin ciclos, y
    cada sección del enunciado cubierta. Si falla, vuelve al planificador con la lista de
    problemas.»

El LLM PROPONE el plan; el código lo VALIDA. Nunca al revés: un plan que «parece» correcto
pero tiene un ciclo o deja una sección sin cubrir es el mismo fallo silencioso de la Parte 0,
solo que en el plan en vez de en las cifras. Si la validación encuentra problemas, se
devuelven al planificador como texto (`ValidadorPlan` nunca llama al LLM) y se reintenta
hasta `MAX_REINTENTOS`; si no converge, `PlanInvalido` lleva el último plan y la lista de
problemas para que el orquestador decida (parcial/fallido), nunca para que se publique.

    from solver.planificador import Planificador
    plan = Planificador().planificar(documento, h200)
    plan.orden_topologico()      # ids en un orden de ejecución válido
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import networkx as nx

from solver.lector import DocumentoLeido

# Los dos tipos que distingue el enunciado: «el Programador escribe un script por subtarea
# DE CÁLCULO» — es decir, hay subtareas que no lo son. Fijo, no lo inventa el LLM: si la
# propuesta trae otro, es un problema de validación, no una categoría nueva.
TIPOS_VALIDOS = frozenset({"calculo", "conceptual"})


# ---------------------------------------------------------------- estructuras
@dataclass
class Subtarea:
    id: str
    tipo: str
    descripcion: str
    dependencias: list[str] = field(default_factory=list)
    secciones: list[str] = field(default_factory=list)   # ids de sección que esta subtarea cubre


@dataclass
class Plan:
    subtareas: list[Subtarea]
    intentos: int = 1

    def por_id(self, id_: str) -> Subtarea | None:
        return next((s for s in self.subtareas if s.id == id_), None)

    def grafo(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(s.id for s in self.subtareas)
        for s in self.subtareas:
            for d in s.dependencias:
                if d in g:
                    g.add_edge(d, s.id)          # la dependencia va ANTES: arista dep → s
        return g

    def orden_topologico(self) -> list[str]:
        """Un orden de ejecución válido: cada subtarea después de todas sus dependencias."""
        return list(nx.topological_sort(self.grafo()))

    def como_dict(self) -> list[dict]:
        return [{"id": s.id, "tipo": s.tipo, "descripcion": s.descripcion,
                "dependencias": s.dependencias, "secciones": s.secciones} for s in self.subtareas]


class PlanInvalido(RuntimeError):
    """El plan no pasó la validación tras agotar los reintentos. Lleva el último plan y los
    problemas para que el orquestador decida —nunca para que el plan se use tal cual."""

    def __init__(self, mensaje: str, ultimo_plan: list[Subtarea], problemas: list[str]):
        super().__init__(mensaje)
        self.ultimo_plan = ultimo_plan
        self.problemas = problemas


# ---------------------------------------------------------------- la validación (sin LLM)
def validar_plan(subtareas: list[Subtarea], ids_cobertura: set[str]) -> list[str]:
    """Regla 1, literal: ids únicos, dependencias que existen, sin ciclos, cobertura total.
    Devuelve la lista de problemas (vacía = el plan es válido). No llama a ningún modelo:
    un plan se valida igual la primera vez que la quinta."""
    problemas: list[str] = []

    ids = [s.id for s in subtareas]
    if not ids:
        return ["el plan no tiene ninguna subtarea"]
    duplicados = sorted({i for i in ids if ids.count(i) > 1})
    if duplicados:
        problemas.append(f"ids duplicados: {duplicados}")
    ids_unicos = set(ids)

    for s in subtareas:
        if s.tipo not in TIPOS_VALIDOS:
            problemas.append(f"«{s.id}»: tipo «{s.tipo}» no es uno de {sorted(TIPOS_VALIDOS)}")
        faltantes = [d for d in s.dependencias if d not in ids_unicos]
        if faltantes:
            problemas.append(f"«{s.id}»: depende de id(s) que no existen en el plan: {faltantes}")

    g = nx.DiGraph()
    g.add_nodes_from(ids_unicos)
    for s in subtareas:
        for d in s.dependencias:
            if d in ids_unicos:
                g.add_edge(s.id, d)
    if not nx.is_directed_acyclic_graph(g):
        ciclos = [c for c in nx.simple_cycles(g)][:3]
        problemas.append(f"el plan tiene ciclo(s) de dependencias: {ciclos}")

    cubiertas = {sec for s in subtareas for sec in s.secciones}
    sin_cubrir = sorted(ids_cobertura - cubiertas)
    if sin_cubrir:
        problemas.append(f"secciones del enunciado sin cubrir por ninguna subtarea: {sin_cubrir}")
    inventadas = sorted(cubiertas - ids_cobertura)
    if inventadas:
        problemas.append(f"secciones referenciadas que no existen en el enunciado: {inventadas}")

    return problemas


# ---------------------------------------------------------------- el prompt
_SISTEMA = """\
Divides el enunciado de una tarea en subtareas para un equipo de agentes que la resuelve.

Reglas del plan (las revisa código, no un criterio tuyo):
- Cada subtarea tiene un "id" corto y único (minúsculas, guiones: "curva-aprendizaje").
- "tipo" es EXACTAMENTE "calculo" (necesita escribir y ejecutar código: entrenar un modelo,
  calcular una métrica, dibujar una figura) o "conceptual" (una pregunta que se responde en
  prosa, sin ejecutar nada: una discusión, una explicación).
- "dependencias": ids de OTRAS subtareas de este mismo plan que deben resolverse antes
  (p. ej., la curva de aprendizaje depende de la subtarea que define la partición de datos).
- "secciones": los ids de sección de la lista de abajo que esta subtarea resuelve. ENTRE
  TODAS las subtareas tienen que quedar cubiertas TODAS las secciones listadas — ninguna se
  deja fuera, y no inventes ids que no estén en la lista.
- "descripcion": una frase concreta de qué debe producir (qué calcula o qué pregunta responde).

Secciones del enunciado (id · título · primeras palabras):
{secciones}

Responde SOLO JSON: {{"subtareas": [{{"id": "...", "tipo": "calculo"|"conceptual",
"descripcion": "...", "dependencias": ["..."], "secciones": ["..."]}}]}}
"""


def _resumen_secciones(documento: DocumentoLeido) -> str:
    return "\n".join(f"- {s.id} · {s.titulo} · {' '.join(s.texto.split())[:120]}…"
                     for s in documento.secciones)


def _parsear_subtareas(contenido: str) -> list[Subtarea]:
    """Del JSON del LLM a `Subtarea`s. Un JSON roto o con forma inesperada no revienta el
    bucle: se convierte en un único problema de validación («propuesta vacía»), y se
    reintenta igual que un plan con ciclos."""
    try:
        datos = json.loads(contenido)
        bruto = datos["subtareas"] if isinstance(datos, dict) else datos
        if not isinstance(bruto, list):
            raise ValueError("«subtareas» no es una lista")
    except (json.JSONDecodeError, KeyError, ValueError):
        return []
    subtareas = []
    for item in bruto:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        id_ = re.sub(r"[^a-z0-9-]+", "-", str(item["id"]).lower()).strip("-") or "sin-id"
        subtareas.append(Subtarea(
            id=id_, tipo=str(item.get("tipo", "")).lower().strip(),
            descripcion=str(item.get("descripcion", ""))[:500],
            dependencias=[str(d) for d in item.get("dependencias", []) if isinstance(d, (str, int))],
            secciones=[str(s) for s in item.get("secciones", []) if isinstance(s, (str, int))]))
    return subtareas


# ---------------------------------------------------------------- el agente
class Planificador:
    MAX_REINTENTOS = 3
    MAX_TOKENS = 8192

    def planificar(self, documento: DocumentoLeido, h200, traza=None) -> Plan:
        ids_cobertura = {s.id for s in documento.secciones}
        mensajes = [{"role": "system", "content": _SISTEMA.format(
            secciones=_resumen_secciones(documento))}]
        problemas: list[str] = []
        subtareas: list[Subtarea] = []

        for intento in range(1, self.MAX_REINTENTOS + 1):
            turno = list(mensajes)
            if problemas:
                turno.append({"role": "user", "content":
                              "El plan anterior no pasó la validación de código:\n" +
                              "\n".join(f"- {p}" for p in problemas) +
                              "\nCorrígelo y responde con el plan completo de nuevo."})
            else:
                turno.append({"role": "user", "content": "Genera el plan."})

            r = h200.chat(turno, json_mode=True, max_tokens=self.MAX_TOKENS)
            if traza:
                traza("llm", "planificador", intento,
                     tokens_entrada=r["uso"].get("prompt_tokens", 0),
                     tokens_salida=r["uso"].get("completion_tokens", 0),
                     latencia_s=r["latencia_s"], fin=r["fin"])

            if r["fin"] == "length" and not r["contenido"].strip():
                problemas = ["la propuesta se quedó sin tokens razonando y no llegó a "
                            "escribir el JSON: hay que subir max_tokens o acortar el prompt"]
                continue

            subtareas = _parsear_subtareas(r["contenido"])
            if not subtareas:
                problemas = ["la propuesta no es un JSON válido con una lista «subtareas» "
                            "no vacía"]
                continue

            problemas = validar_plan(subtareas, ids_cobertura)
            if traza:
                traza("validacion_plan", "planificador", intento,
                     valido=not problemas, problemas=problemas)
            if not problemas:
                return Plan(subtareas=subtareas, intentos=intento)

        raise PlanInvalido(
            f"el plan no pasó la validación tras {self.MAX_REINTENTOS} intento(s): {problemas}",
            subtareas, problemas)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from solver.lector import Lector

    nombre = sys.argv[1] if len(sys.argv) > 1 else "tarea-a-generativo-discriminativo"
    raiz = Path(__file__).resolve().parents[1]
    doc = Lector().leer(raiz / "enunciados" / f"{nombre}.pdf")
    print(f"Secciones a cubrir: {[s.id for s in doc.secciones]}\n")

    from h200 import H200
    plan = Planificador().planificar(doc, H200())
    print(f"Plan válido en {plan.intentos} intento(s):")
    for s in plan.subtareas:
        print(f"  [{s.id}] ({s.tipo}) dep={s.dependencias} secciones={s.secciones}")
        print(f"      {s.descripcion}")
    print(f"\nOrden topológico: {plan.orden_topologico()}")
