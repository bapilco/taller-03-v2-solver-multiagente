"""investigador.py — el rol Investigador: junta el contexto de cada subtarea.

Taller 03 v2, Parte 1. Lo que revisa el código de este rol, según la tabla de roles: «que
vayan la sección y sus dependencias». Es decir, no basta con que el contexto "parezca"
completo: el código comprueba, subtarea por subtarea, que el texto literal de su propia
sección y el de cada subtarea de la que depende están realmente en lo que se le entrega al
Programador. Lo que entrega GraphRAG por similitud (incluido el material del curso) es un
añadido, no un reemplazo de esa garantía.

    from solver.investigador import Investigador
    contexto, problemas = Investigador().investigar(subtarea, plan, indexador, fuente_id="tarea-a")
    contexto.texto_citado()     # lo que recibe el Programador, cada fragmento con su cita
"""
from __future__ import annotations

from dataclasses import dataclass, field

from solver.indexador import Fragmento, Indexador
from solver.planificador import Plan, Subtarea


@dataclass
class ContextoSubtarea:
    subtarea_id: str
    fragmentos: list[Fragmento] = field(default_factory=list)

    def texto_citado(self) -> str:
        """Lo que recibe el Programador: cada fragmento con su cita delante, para que
        cualquier cifra que copie pueda rastrearse hasta su origen (Regla 5)."""
        return "\n\n".join(f"{f.cita()}\n{f.texto}" for f in self.fragmentos)

    def ids(self) -> set[str]:
        return {f.id for f in self.fragmentos}


def validar_contexto(subtarea: Subtarea, plan: Plan, contexto: ContextoSubtarea,
                     fuente_id: str) -> list[str]:
    """Código, no criterio: ¿están la sección propia y las de cada dependencia? Si falta
    una, se dice CUÁL —no «el contexto parece corto»— para que el investigador pueda
    corregirlo con un reintento dirigido en vez de adivinar."""
    problemas: list[str] = []
    presentes = contexto.ids()

    faltan_propias = [s for s in subtarea.secciones if f"{fuente_id}:{s}" not in presentes]
    if faltan_propias:
        problemas.append(f"«{subtarea.id}»: faltan secciones propias en el contexto: {faltan_propias}")

    for dep_id in subtarea.dependencias:
        dep = plan.por_id(dep_id)
        if dep is None:
            problemas.append(f"«{subtarea.id}»: depende de «{dep_id}», que no está en el plan")
            continue
        faltan_dep = [s for s in dep.secciones if f"{fuente_id}:{s}" not in presentes]
        if faltan_dep:
            problemas.append(f"«{subtarea.id}»: faltan secciones de su dependencia "
                            f"«{dep_id}» en el contexto: {faltan_dep}")
    return problemas


class Investigador:
    """Arma el contexto de una subtarea: su sección literal, las de sus dependencias, y lo
    que el GraphRAG recupere por similitud (enunciado + curso), cada fragmento con su cita."""

    K_EXTRA = 2       # fragmentos adicionales por similitud, además de los garantizados
    SALTOS_EXTRA = 1  # saltos por depende_de/menciona dentro de esos fragmentos adicionales

    def investigar(self, subtarea: Subtarea, plan: Plan, indexador: Indexador,
                   fuente_id: str, h200=None, traza=None) -> tuple[ContextoSubtarea, list[str]]:
        fragmentos: dict[str, Fragmento] = {}

        def _agregar(seccion_id: str, via: str) -> None:
            nid = f"{fuente_id}:{seccion_id}"
            if nid in fragmentos or nid not in indexador.grafo:
                return
            d = indexador.grafo.nodes[nid]
            fragmentos[nid] = Fragmento(id=nid, titulo=d["titulo"], texto=d["texto"],
                                       fuente=d["fuente"], archivo=d["archivo"],
                                       ubicacion=d["ubicacion"], via=via)

        # 1 · lo garantizado: la sección literal de ESTA subtarea…
        for sec in subtarea.secciones:
            _agregar(sec, "subtarea")

        # 2 · …y la de CADA subtarea de la que depende (no solo su id: su contenido)
        for dep_id in subtarea.dependencias:
            dep = plan.por_id(dep_id)
            if dep is None:
                continue
            for sec in dep.secciones:
                _agregar(sec, f"dependencia:{dep_id}")

        # 3 · lo que añade el GraphRAG: similitud con la descripción de la subtarea, con un
        # salto por depende_de/menciona — puede traer material del curso, no solo del enunciado
        for frag in indexador.contexto_para(subtarea.descripcion, k=self.K_EXTRA,
                                            saltos=self.SALTOS_EXTRA, h200=h200):
            fragmentos.setdefault(frag.id, frag)

        contexto = ContextoSubtarea(subtarea_id=subtarea.id, fragmentos=list(fragmentos.values()))
        problemas = validar_contexto(subtarea, plan, contexto, fuente_id)
        if traza:
            traza("contexto_subtarea", "investigador", None, subtarea=subtarea.id,
                 fragmentos=sorted(contexto.ids()), completo=not problemas, problemas=problemas)
        return contexto, problemas


if __name__ == "__main__":
    from pathlib import Path

    from solver.lector import Lector

    raiz = Path(__file__).resolve().parents[1]
    doc = Lector().leer(raiz / "enunciados" / "tarea-a-generativo-discriminativo.pdf")

    idx = Indexador()
    idx.agregar_documento(doc, fuente_id="tarea-a")
    idx.agregar_notas_curso(raiz.parent / "notas-teoricas")

    # Un plan de muestra, igual al que validamos a mano para el Planificador: sin H200 no
    # hay plan real, pero el Investigador no necesita uno real para probarse — necesita uno
    # VÁLIDO, y este lo es.
    plan = Plan(subtareas=[
        Subtarea("datos", "calculo", "carga y partición 70/30", secciones=["tarea-a", "parte-1"]),
        Subtarea("clasificadores", "calculo", "entrena NB y LR", dependencias=["datos"],
                secciones=["parte-2"]),
        Subtarea("curva", "calculo", "con qué datos y partición se calcula la curva de "
                "aprendizaje de NB y regresión logística", dependencias=["datos"],
                secciones=["parte-3"]),
        Subtarea("discusion", "conceptual", "explica el cruce generativo-discriminativo",
                dependencias=["clasificadores", "curva"], secciones=["parte-4"]),
    ])

    for s in plan.subtareas:
        contexto, problemas = Investigador().investigar(s, plan, idx, fuente_id="tarea-a")
        print(f"[{s.id}] {len(contexto.fragmentos)} fragmento(s), "
             f"{'OK' if not problemas else 'PROBLEMAS: ' + str(problemas)}")
        for f in contexto.fragmentos:
            print(f"    {f.via:22s} {f.cita()}")
        print()
