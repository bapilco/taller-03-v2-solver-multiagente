"""sandbox.py — la guarda del código generado: revisión estática antes de ejecutar.

Taller 03 v2, Regla 3 obligatoria:

    «El código corre en un sandbox: se revisa antes de ejecutar (sin red, sin procesos,
    sin borrar, sin eval, sin salir de la carpeta); se ejecuta sin variables de entorno
    (ninguna clave llega al script), en su propia carpeta y con tiempo máximo que mata
    todos sus procesos.»

Esta revisión (`revisar_codigo`) es código determinístico sobre el AST del script —no pasa
por ningún LLM, como la guarda de `mini-opencode/guardia.py` para comandos de shell. Mismo
principio, adaptado de tokens de shell a nodos de Python: **lista negra de lo explícitamente
prohibido**, no lista blanca de imports —porque un script de cálculo legítimo puede necesitar
cualquier submódulo de scikit-learn, numpy o matplotlib, y enumerarlos a mano sería repetir
el error de sobre-restringir que la sesión 14 ya señaló para el caso contrario (lista blanca
demasiado angosta).

Las cinco familias que prohíbe la Regla 3, con su regla de detección:

    R0  forma         el código no compila (`ast.parse` falla) o está vacío
    R1  red           import de un módulo de red (socket, requests, urllib.request…)
    R2  procesos      import de un módulo que lanza procesos (subprocess, multiprocessing…)
    R3  borrado       llamada a os.remove/unlink/rmdir, shutil.rmtree, o `.unlink()`/`.rmdir()`
                      de cualquier objeto (cubre `pathlib.Path(...).unlink()`)
    R4  eval          llamada a eval/exec/compile/__import__
    R5  confinamiento os.chdir, o una cadena que parece una ruta fuera de la carpeta de
                      trabajo (absoluta, o con `..`)

A diferencia de la guarda de shell, aquí se acumulan TODOS los hallazgos en una pasada: el
Programador necesita la lista completa para corregir en un solo reintento, no una regla a la
vez (ver Planificador.planificar, que reintenta igual con la lista completa de problemas).

Lo que NO hace esta revisión —y por qué no hace falta— está en `herramientas.py` del Ejecutor:
`shell=False` (no hay intérprete que componga nada), un entorno mínimo sin credenciales, y un
tiempo máximo que mata el proceso. Los import de proceso que aquí se prohíben son justamente
lo que haría innecesario ese límite de «matar todos sus procesos»: si el script no puede
lanzar hijos, matar el proceso basta.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

NOMBRE_RESULTADOS = "resultados.json"     # Regla 4: nombre fijo, igual en todas las subtareas

# ---------------------------------------------------------------- R1 — red
MODULOS_RED = frozenset({
    "socket", "ssl", "requests", "httpx", "urllib.request", "urllib.error", "urllib3",
    "http.client", "http.server", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
    "paramiko", "aiohttp", "websocket", "websockets", "xmlrpc.client", "nntplib",
})

# ---------------------------------------------------------------- R2 — procesos
# No se prohíbe `threading`/`asyncio`: la Regla 3 dice «sin procesos», no «sin concurrencia»,
# y lo que hace que «matar el proceso basta» sea cierto es que el script no pueda lanzar
# OTRO PROCESO del sistema operativo, no que sea estrictamente secuencial.
MODULOS_PROCESOS = frozenset({"subprocess", "multiprocessing", "ctypes", "_thread", "pty"})

# ---------------------------------------------------------------- R3 — borrado
LLAMADAS_BORRADO: dict[str, dict[str, str]] = {
    "os": {"remove": "os.remove borra un archivo", "unlink": "os.unlink borra un archivo",
          "rmdir": "os.rmdir borra un directorio",
          "removedirs": "os.removedirs borra directorios"},
    "shutil": {"rmtree": "shutil.rmtree borra un árbol de directorios"},
}
METODOS_BORRADO_CUALQUIER_OBJETO = frozenset({"unlink", "rmdir"})   # pathlib.Path(...).unlink()

# ---------------------------------------------------------------- R4 — eval
FUNCIONES_EVAL = frozenset({"eval", "exec", "compile", "__import__"})

# ---------------------------------------------------------------- R5 — confinamiento
LLAMADAS_CONFINAMIENTO: dict[str, dict[str, str]] = {
    "os": {"chdir": "os.chdir cambia el directorio de trabajo fuera del control del sandbox"},
}
_PATRON_RUTA_FUERA = re.compile(r'^(?:[A-Za-z]:[\\/]|/)|(?:\.\.[\\/])')


@dataclass
class Hallazgo:
    regla: str
    linea: int
    motivo: str

    def __str__(self) -> str:
        return f"[{self.regla}] L{self.linea}: {self.motivo}"


@dataclass
class VeredictoSandbox:
    permitido: bool
    hallazgos: list[Hallazgo] = field(default_factory=list)

    def __str__(self) -> str:
        if self.permitido:
            return "PERMITIDO"
        return "RECHAZADO — " + "; ".join(str(h) for h in self.hallazgos)

    def problemas(self) -> list[str]:
        """Como lista de texto, en el mismo formato que `validar_plan`/`validar_contexto`:
        para que el Programador reciba retroalimentación con la misma forma en todo el
        sistema, no un formato distinto por cada agente."""
        return [str(h) for h in self.hallazgos]


def _alias_de_modulos(arbol: ast.AST) -> dict[str, str]:
    """`import shutil as sh` → {"sh": "shutil"}, para resolver `sh.rmtree(...)`."""
    alias: dict[str, str] = {}
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for a in nodo.names:
                alias[a.asname or a.name.split(".")[0]] = a.name
    return alias


def _alias_de_funciones(arbol: ast.AST) -> dict[str, tuple[str, str]]:
    """`from os import remove as rm` → {"rm": ("os", "remove")}. Sin esto, `remove(...)`
    llamada como nombre suelto (no `os.remove(...)`) se cuela: la Regla 3 prohíbe la
    OPERACIÓN, no una forma particular de escribirla."""
    alias: dict[str, tuple[str, str]] = {}
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and nodo.module:
            for a in nodo.names:
                alias[a.asname or a.name] = (nodo.module, a.name)
    return alias


def _modulo_importado(nodo: ast.Import | ast.ImportFrom) -> list[tuple[str, int]]:
    if isinstance(nodo, ast.Import):
        return [(a.name, nodo.lineno) for a in nodo.names]
    if nodo.module:
        return [(nodo.module, nodo.lineno)]
    return []


def _coincide_modulo(importado: str, prohibidos: frozenset[str]) -> str | None:
    """«urllib.request» prohíbe también «urllib.request.something», pero NO «urllib.parse»."""
    for p in prohibidos:
        if importado == p or importado.startswith(p + "."):
            return p
    return None


def revisar_codigo(codigo: str) -> VeredictoSandbox:
    """Todos los hallazgos de una pasada; `permitido` es `True` solo si no hay ninguno."""
    if not codigo.strip():
        return VeredictoSandbox(False, [Hallazgo("R0", 0, "el script está vacío")])
    try:
        arbol = ast.parse(codigo)
    except SyntaxError as err:
        return VeredictoSandbox(False, [Hallazgo("R0", err.lineno or 0, f"no compila: {err.msg}")])

    hallazgos: list[Hallazgo] = []
    alias_modulos = _alias_de_modulos(arbol)
    alias_funciones = _alias_de_funciones(arbol)

    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Import, ast.ImportFrom)):
            for nombre, linea in _modulo_importado(nodo):
                if p := _coincide_modulo(nombre, MODULOS_RED):
                    hallazgos.append(Hallazgo("R1", linea, f"import de red: «{p}»"))
                if p := _coincide_modulo(nombre, MODULOS_PROCESOS):
                    hallazgos.append(Hallazgo("R2", linea, f"import que lanza procesos: «{p}»"))

        elif isinstance(nodo, ast.Call):
            linea = nodo.lineno
            if isinstance(nodo.func, ast.Name):
                if nodo.func.id in FUNCIONES_EVAL:
                    hallazgos.append(Hallazgo("R4", linea, f"llamada a {nodo.func.id}(...)"))
                elif resuelto := alias_funciones.get(nodo.func.id):
                    modulo_real, attr = resuelto
                    if attr in LLAMADAS_BORRADO.get(modulo_real, {}):
                        hallazgos.append(Hallazgo("R3", linea,
                                                  f"{LLAMADAS_BORRADO[modulo_real][attr]} "
                                                  f"(importado como «{nodo.func.id}»)"))
                    elif attr in LLAMADAS_CONFINAMIENTO.get(modulo_real, {}):
                        hallazgos.append(Hallazgo("R5", linea,
                                                  f"{LLAMADAS_CONFINAMIENTO[modulo_real][attr]} "
                                                  f"(importado como «{nodo.func.id}»)"))
            elif isinstance(nodo.func, ast.Attribute):
                attr = nodo.func.attr
                modulo_real = None
                if isinstance(nodo.func.value, ast.Name):
                    modulo_real = alias_modulos.get(nodo.func.value.id, nodo.func.value.id)
                if modulo_real and attr in LLAMADAS_BORRADO.get(modulo_real, {}):
                    hallazgos.append(Hallazgo("R3", linea, LLAMADAS_BORRADO[modulo_real][attr]))
                elif attr in METODOS_BORRADO_CUALQUIER_OBJETO:
                    hallazgos.append(Hallazgo("R3", linea,
                                              f".{attr}() borra algo (objeto no resuelto "
                                              "estáticamente: se asume que es un Path)"))
                elif modulo_real and attr in LLAMADAS_CONFINAMIENTO.get(modulo_real, {}):
                    hallazgos.append(Hallazgo("R5", linea, LLAMADAS_CONFINAMIENTO[modulo_real][attr]))

        elif isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
            if _PATRON_RUTA_FUERA.search(nodo.value):
                hallazgos.append(Hallazgo("R5", nodo.lineno,
                                          f"cadena con pinta de ruta fuera de la carpeta: {nodo.value!r}"))

    return VeredictoSandbox(permitido=not hallazgos, hallazgos=hallazgos)


if __name__ == "__main__":
    EJEMPLOS = {
        "limpio": """
import json
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
json.dump({"n": len(X)}, open("resultados.json", "w"))
""",
        "red": "import requests\nrequests.get('http://evil.com')\n",
        "procesos": "import subprocess\nsubprocess.run(['ls'])\n",
        "borrado": "import os\nos.remove('resultados.json')\n",
        "eval": "eval('1+1')\n",
        "confinamiento": "open('/etc/passwd').read()\n",
        "roto": "def f(:\n",
    }
    for nombre, codigo in EJEMPLOS.items():
        print(f"{nombre:14s} → {revisar_codigo(codigo)}")
