"""ejecutor.py — el rol Ejecutor: corre el script en el sandbox. «Es código, sin LLM.»

Taller 03 v2, Regla 3 (la parte que no es revisión estática, sino ejecución):

    «se ejecuta sin variables de entorno (ninguna clave llega al script), en su propia
    carpeta y con tiempo máximo que mata todos sus procesos.»

Por qué «matar el proceso» basta para «matar todos sus procesos»: el script ya pasó por
`sandbox.revisar_codigo`, que prohíbe (R2) justamente los imports que lanzarían hijos
(`subprocess`, `multiprocessing`…). Las dos capas se refuerzan: si el script no puede tener
hijos, matar el único proceso que existe mata todos los que existen. Ninguna sustituye a la
otra —el mismo principio que los frenos de la sesión 14—: esta ejecución igual aplica el
tiempo máximo aunque la revisión estática ya lo haga improbable, por si una regla nueva de
Python la elude.

Entorno mínimo: ni las credenciales del solver (H200_HOST, claves de API) ni nada del
usuario llegan al script —solo lo que Windows necesita para arrancar un intérprete
(`SystemRoot`, rutas temporales). Ningún script generado necesita más que eso para leer su
contexto (que ya viene como texto, no como variable de entorno) y escribir su JSON.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from solver.sandbox import NOMBRE_RESULTADOS

TIMEOUT_S_DEFECTO = 60
# Solo lo que el intérprete necesita para arrancar en Windows/POSIX; ninguna clave propia
# del solver (H200_HOST, credenciales) está en esta lista a propósito.
_VARS_ARRANQUE = ("SystemRoot", "windir", "TEMP", "TMP", "NUMBER_OF_PROCESSORS",
                  "PROCESSOR_ARCHITECTURE", "PATH")


def _entorno_minimo(carpeta_trabajo: Path) -> dict[str, str]:
    entorno = {v: os.environ[v] for v in _VARS_ARRANQUE if v in os.environ}
    # No es una credencial: fija cómo el hijo codifica su propia salida. Sin esto, el hijo
    # usa la codificación de la consola (cp1252 en Windows) y el padre —que decodifica como
    # UTF-8 para leer stdout/stderr de forma consistente entre plataformas— revienta con
    # `UnicodeDecodeError` en cuanto el script imprime un carácter no ASCII (se vio con un
    # script real generado por el Programador, 2026-10-04).
    entorno["PYTHONIOENCODING"] = "utf-8"
    entorno["PYTHONUTF8"] = "1"
    # matplotlib busca su caché de configuración con `Path.home()`, que falla si no hay
    # USERPROFILE en el entorno (el caso normal aquí: Rule 3 no permite variables del
    # usuario). La solución correcta no es exponer el home real —eso sería salir de la
    # carpeta—, sino darle una caché DENTRO del sandbox (se ve con un script real que
    # importa matplotlib, 2026-10-04).
    entorno["MPLCONFIGDIR"] = str(carpeta_trabajo / ".mplconfig")
    return entorno


@dataclass
class ResultadoEjecucion:
    codigo_salida: int
    stdout: str
    stderr: str
    duracion_s: float
    archivos_creados: list[str] = field(default_factory=list)
    agotado_tiempo: bool = False
    resultados: dict | None = None     # el contenido de resultados.json, si el script lo escribió

    @property
    def ok(self) -> bool:
        return self.codigo_salida == 0 and not self.agotado_tiempo


class Ejecutor:
    """Corre un script ya aprobado por el sandbox, en su propia carpeta, con env mínimo y
    tiempo máximo. No valida nada del código: eso es trabajo del Sandbox, antes; y del
    Revisor, después. El Ejecutor solo ejecuta y registra lo que pasó —por eso «es código,
    sin LLM»: nada aquí decide si el resultado es bueno."""

    def ejecutar(self, ruta_script: Path, carpeta_trabajo: Path,
                timeout_s: int = TIMEOUT_S_DEFECTO) -> ResultadoEjecucion:
        carpeta_trabajo = Path(carpeta_trabajo).resolve()
        ruta_script = Path(ruta_script).resolve()
        if carpeta_trabajo not in ruta_script.parents and ruta_script.parent != carpeta_trabajo:
            raise ValueError(f"«{ruta_script}» no está dentro de «{carpeta_trabajo}»: "
                            "el Ejecutor no corre scripts fuera de su propia carpeta")

        antes = self._listar(carpeta_trabajo)
        t0 = time.perf_counter()
        agotado = False
        proc = subprocess.Popen(
            [sys.executable, ruta_script.name], cwd=carpeta_trabajo,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", env=_entorno_minimo(carpeta_trabajo))
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()                        # mata el proceso; no tiene hijos (R2 lo impide)
            stdout, stderr = proc.communicate()
            agotado = True
            stderr = (stderr or "") + f"\n[SANDBOX] tiempo máximo de {timeout_s}s agotado: proceso terminado"
        duracion = round(time.perf_counter() - t0, 2)

        despues = self._listar(carpeta_trabajo)
        archivos_creados = sorted(despues - antes)

        resultados = None
        ruta_resultados = carpeta_trabajo / NOMBRE_RESULTADOS
        if ruta_resultados.exists():
            try:
                resultados = json.loads(ruta_resultados.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                resultados = None          # un JSON roto no es un resultado: el Revisor lo verá como ausente

        return ResultadoEjecucion(
            codigo_salida=-1 if agotado else proc.returncode,
            stdout=stdout, stderr=stderr, duracion_s=duracion,
            archivos_creados=archivos_creados, agotado_tiempo=agotado, resultados=resultados)

    @staticmethod
    def _listar(carpeta: Path) -> set[str]:
        return {str(p.relative_to(carpeta)) for p in carpeta.rglob("*") if p.is_file()}


if __name__ == "__main__":
    import tempfile

    codigo_ejemplo = """
import json
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
acc = accuracy_score(yte, GaussianNB().fit(Xtr, ytr).predict(Xte))
print(f"accuracy: {acc:.4f}")
json.dump({"NB accuracy": acc}, open("resultados.json", "w"))
"""
    with tempfile.TemporaryDirectory(prefix="ejecutor-demo-") as tmp:
        carpeta = Path(tmp)
        script = carpeta / "experimento.py"
        script.write_text(codigo_ejemplo, encoding="utf-8")
        r = Ejecutor().ejecutar(script, carpeta, timeout_s=30)
        print(f"ok={r.ok} codigo_salida={r.codigo_salida} duracion_s={r.duracion_s}")
        print(f"stdout: {r.stdout.strip()}")
        print(f"archivos creados: {r.archivos_creados}")
        print(f"resultados.json: {r.resultados}")

        print("\n--- ahora con timeout agotado a propósito ---")
        script2 = carpeta / "lento.py"
        script2.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
        r2 = Ejecutor().ejecutar(script2, carpeta, timeout_s=1)
        print(f"ok={r2.ok} agotado_tiempo={r2.agotado_tiempo} duracion_s={r2.duracion_s}")
        print(f"stderr: {r2.stderr.strip()}")
