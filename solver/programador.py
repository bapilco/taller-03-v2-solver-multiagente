"""programador.py — el rol Programador: escribe un script por subtarea de cálculo.

Taller 03 v2. Qué revisa tu código, según la tabla de roles: «el filtro del sandbox». El LLM
escribe el script; `sandbox.revisar_codigo` lo valida ANTES de que el Ejecutor lo toque —el
Programador nunca entrega un script directamente al Ejecutor sin que pase por ahí. Si el
sandbox lo rechaza, se reintenta con la lista exacta de hallazgos (mismo patrón que
`Planificador.planificar` e `Investigador.investigar`: el modelo corrige con la regla y la
línea que violó, no con una pista genérica).

    from solver.programador import Programador
    codigo, veredicto = Programador().programar(subtarea, contexto, h200)
"""
from __future__ import annotations

import re

from solver.investigador import ContextoSubtarea
from solver.planificador import Subtarea
from solver.sandbox import NOMBRE_RESULTADOS, Hallazgo, VeredictoSandbox, revisar_codigo

_SISTEMA = """\
Escribes un script de Python que resuelve UNA subtarea de cálculo de una tarea académica,
usando el contexto citado del enunciado (y del material del curso, si aplica).

Reglas del sandbox donde correrá (las revisa código, no tu criterio; si las rompes, el
script se rechaza antes de ejecutarse y tienes que reescribirlo):
- Nada de red: ni requests, ni urllib.request, ni socket, ni descargar nada (sin
  download=True, sin fetch_openml, sin load_dataset). Usa solo los datos del contexto o los
  datasets que YA trae scikit-learn sin descargar nada nuevo.
- Nada de lanzar procesos (subprocess, multiprocessing) ni de borrar archivos (os.remove,
  shutil.rmtree, Path.unlink).
- Nada de eval/exec/compile/__import__.
- Ninguna ruta absoluta ni con «..»: todo lo que leas o escribas es relativo a tu propia
  carpeta.
- Guarda TODAS las cifras que calcules —todas— en un archivo "{nombre_resultados}" (ese
  nombre exacto), con claves descriptivas y valores numéricos (no texto). Si produces una
  figura, guárdala como PNG en la misma carpeta y referencia su nombre en el JSON.
- No inventes datos: si el contexto no alcanza para algo, dilo en un comentario y resuelve
  lo que sí puedas con los datos reales.

Responde SOLO con el código Python completo del script. Sin explicación, sin markdown, sin
```.

Contexto citado del enunciado:
{contexto}
"""


class ScriptInvalido(RuntimeError):
    """El script no pasó el sandbox tras agotar los reintentos. Lleva el último código y el
    veredicto para que el orquestador decida —nunca para que se ejecute de todos modos."""

    def __init__(self, mensaje: str, ultimo_codigo: str, veredicto: VeredictoSandbox):
        super().__init__(mensaje)
        self.ultimo_codigo = ultimo_codigo
        self.veredicto = veredicto


def _limpiar_markdown(texto: str) -> str:
    """El LLM a veces envuelve el código en ```python … ``` aunque se le pida que no lo
    haga: quitarlo aquí es más barato que gastar un reintento completo por eso."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", texto, re.S)
    return m.group(1) if m else texto


class Programador:
    MAX_REINTENTOS = 3
    MAX_TOKENS = 4096

    def programar(self, subtarea: Subtarea, contexto: ContextoSubtarea, h200,
                 traza=None) -> tuple[str, VeredictoSandbox]:
        mensajes = [{"role": "system", "content": _SISTEMA.format(
            nombre_resultados=NOMBRE_RESULTADOS, contexto=contexto.texto_citado())}]
        veredicto = VeredictoSandbox(False)
        codigo = ""

        for intento in range(1, self.MAX_REINTENTOS + 1):
            turno = list(mensajes)
            if veredicto.hallazgos:
                turno.append({"role": "user", "content":
                              "El sandbox rechazó el script anterior:\n" +
                              "\n".join(f"- {p}" for p in veredicto.problemas()) +
                              "\nReescríbelo evitando exactamente eso. Responde solo con el "
                              "código completo."})
            else:
                turno.append({"role": "user", "content":
                              f"Subtarea «{subtarea.id}»: {subtarea.descripcion}"})

            r = h200.chat(turno, max_tokens=self.MAX_TOKENS)
            if traza:
                traza("llm", "programador", intento, subtarea=subtarea.id,
                     tokens_entrada=r["uso"].get("prompt_tokens", 0),
                     tokens_salida=r["uso"].get("completion_tokens", 0),
                     latencia_s=r["latencia_s"], fin=r["fin"])

            if r["fin"] == "length" and not r["contenido"].strip():
                veredicto = VeredictoSandbox(False, [Hallazgo(
                    "R0", 0, "el modelo se quedó sin tokens razonando y no escribió código")])
                continue

            codigo = _limpiar_markdown(r["contenido"])
            veredicto = revisar_codigo(codigo)
            if traza:
                traza("sandbox", "programador", intento, subtarea=subtarea.id,
                     permitido=veredicto.permitido, hallazgos=veredicto.problemas())
            if veredicto.permitido:
                return codigo, veredicto

        raise ScriptInvalido(
            f"«{subtarea.id}»: el script no pasó el sandbox tras {self.MAX_REINTENTOS} "
            f"intento(s): {veredicto.problemas()}", codigo, veredicto)


if __name__ == "__main__":
    from pathlib import Path

    from solver.indexador import Indexador
    from solver.investigador import Investigador
    from solver.lector import Lector
    from solver.planificador import Plan

    raiz = Path(__file__).resolve().parents[1]
    doc = Lector().leer(raiz / "enunciados" / "tarea-a-generativo-discriminativo.pdf")
    idx = Indexador()
    idx.agregar_documento(doc, fuente_id="tarea-a")

    plan = Plan(subtareas=[
        Subtarea("datos", "calculo", "carga y partición 70/30", secciones=["tarea-a", "parte-1"]),
        Subtarea("clasificadores", "calculo",
                "entrena Naive Bayes gaussiano y regresión logística (con escalador ajustado "
                "solo en entrenamiento) y reporta accuracy y F1 macro de cada uno",
                dependencias=["datos"], secciones=["parte-2"]),
    ])
    subtarea = plan.subtareas[1]
    contexto, problemas = Investigador().investigar(subtarea, plan, idx, fuente_id="tarea-a")
    assert not problemas, problemas

    from h200 import H200
    codigo, veredicto = Programador().programar(subtarea, contexto, H200())
    print(f"veredicto del sandbox: {veredicto}\n")
    print(codigo)
