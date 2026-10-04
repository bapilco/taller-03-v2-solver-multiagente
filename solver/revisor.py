"""revisor.py — el rol Revisor: aprueba o devuelve con la corrección.

Taller 03 v2. Qué revisa tu código, según la tabla de roles: «errores, resultados, NaN,
figuras, fugas». Y la Regla 4, que fija el ORDEN:

    «Resultados en un archivo fijo y revisión con código primero. Cada script guarda sus
    cifras en un JSON con nombre fijo. El revisor corre primero las revisiones de la 0.c y
    solo si pasan consulta al LLM.»

Dos capas, nunca en el otro orden:

    1. CÓDIGO (sin LLM)   lo mismo que `parte0/c_exit_cero.py` demostró: ¿terminó con
                          código 0? ¿escribió resultados.json? ¿hay fuga (predecir sobre la
                          misma variable con la que se ajustó)? ¿hay NaN/inf? ¿una métrica
                          ≥0,999 en un problema con ruido? ¿falta una figura esperada?
    2. LLM (opcional)     SOLO si la capa de código aprobó: ¿el resultado responde lo que
                          pedía la subtarea? Un revisor LLM nunca ve un script que ya falló
                          una comprobación objetiva —eso sería pedirle que opine sobre algo
                          que el código ya sabe que está roto.

Si cualquier capa rechaza, el veredicto trae el motivo para que el Programador reintente
(mismo patrón que Sandbox → Programador): la retroalimentación es siempre accionable, nunca
«no pasó la revisión».
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field

from solver.ejecutor import ResultadoEjecucion
from solver.planificador import Subtarea
from solver.sandbox import NOMBRE_RESULTADOS

UMBRAL_METRICA_IMPLAUSIBLE = 0.999
PREFIJOS_METRICA = ("acc", "f1", "exact", "r2", "auc", "precision", "recall")


# ---------------------------------------------------------------- capa 1: código (0.c)
def _variables_evaluadas_sobre_entrenamiento(codigo: str) -> list[str]:
    """El mismo patrón que `parte0/c_exit_cero.py`: variables que aparecen como X en
    `.fit(X, …)` y otra vez en `.predict(X)`/`.score(X, …)`. No prueba que haya fuga —una
    validación cruzada bien hecha puede reutilizar nombres—, pero obliga a mirar."""
    try:
        arbol = ast.parse(codigo)
    except SyntaxError:
        return []
    ajustadas, evaluadas = set(), []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute) and nodo.args:
            primero = nodo.args[0]
            if not isinstance(primero, ast.Name):
                continue
            if nodo.func.attr in {"fit", "fit_transform"}:
                ajustadas.add(primero.id)
            elif nodo.func.attr in {"predict", "predict_proba", "score"}:
                evaluadas.append(primero.id)
    return sorted({v for v in evaluadas if v in ajustadas})


def _metricas_implausibles(resultados: dict, techo: float = UMBRAL_METRICA_IMPLAUSIBLE) -> list[str]:
    return [f"{k}={v}" for k, v in resultados.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k.lower().startswith(PREFIJOS_METRICA) and v >= techo]


def _valores_no_finitos(resultados, prefijo: str = "") -> list[str]:
    malos: list[str] = []
    if isinstance(resultados, dict):
        for k, v in resultados.items():
            malos += _valores_no_finitos(v, f"{prefijo}.{k}" if prefijo else str(k))
    elif isinstance(resultados, list):
        for i, v in enumerate(resultados):
            malos += _valores_no_finitos(v, f"{prefijo}[{i}]")
    elif isinstance(resultados, float) and not math.isfinite(resultados):
        malos.append(f"{prefijo}={resultados}")
    return malos


def revisar_con_codigo(codigo: str, resultado: ResultadoEjecucion,
                       espera_figura: bool = False) -> list[str]:
    """La capa que corre SIEMPRE primero. Nunca llama a ningún modelo: una ejecución se
    juzga igual la primera vez que la quinta."""
    if resultado.agotado_tiempo:
        return ["el script agotó el tiempo máximo: no terminó de ejecutarse"]
    if resultado.codigo_salida != 0:
        return [f"el script terminó con código {resultado.codigo_salida} (se esperaba 0): "
                f"{resultado.stderr[-500:].strip()}"]
    if resultado.resultados is None:
        return [f"no se encontró o no se pudo leer «{NOMBRE_RESULTADOS}»: el script debe "
                "guardar sus cifras ahí (Regla 4)"]
    if not resultado.resultados:
        return [f"«{NOMBRE_RESULTADOS}» está vacío: el script no guardó ninguna cifra"]

    problemas: list[str] = []
    if fugas := _variables_evaluadas_sobre_entrenamiento(codigo):
        problemas.append("posible fuga de datos: se predice/evalúa sobre la misma variable "
                        f"con la que se ajustó: {fugas}")
    if no_finitos := _valores_no_finitos(resultado.resultados):
        problemas.append(f"valores no finitos (NaN/inf) en los resultados: {no_finitos}")
    if raras := _metricas_implausibles(resultado.resultados):
        problemas.append("métrica(s) sospechosamente perfecta(s) (≥0,999) en un problema "
                        f"con ruido: {raras}")
    if espera_figura and not any(f.endswith(".png") for f in resultado.archivos_creados):
        problemas.append("se esperaba una figura (PNG) y el script no creó ninguna")
    return problemas


# ---------------------------------------------------------------- capa 2: LLM (opcional)
_SISTEMA_LLM = """\
Revisas el resultado de un script que resuelve una subtarea de cálculo de una tarea
académica. Ya pasó las comprobaciones de código: sin fuga de datos evidente, sin NaN, sin
métricas sospechosamente perfectas, terminó sin error y guardó sus cifras. Ahora juzgas algo
que el código no puede: ¿el resultado responde lo que pedía la subtarea? ¿El método es el
que se pidió? ¿Falta alguna cifra que la subtarea pedía explícitamente?

Responde SOLO JSON: {{"aprobado": true|false, "motivo": "..."}}. Si rechazas, el motivo debe
ser accionable —qué cambiar en el script—, no una opinión de estilo.

Subtarea: {descripcion}

Código:
{codigo}

stdout del script:
{stdout}

resultados.json:
{resultados}
"""


@dataclass
class VeredictoRevision:
    aprobado: bool
    problemas: list[str] = field(default_factory=list)
    fuente: str = "codigo"      # "codigo" | "llm" — en qué capa se decidió

    def __str__(self) -> str:
        if self.aprobado:
            return f"APROBADO (capa: {self.fuente})"
        return f"RECHAZADO (capa: {self.fuente}) — " + "; ".join(self.problemas)


class Revisor:
    MAX_TOKENS_LLM = 2048

    def revisar(self, subtarea: Subtarea, codigo: str, resultado: ResultadoEjecucion,
               h200=None, espera_figura: bool = False, traza=None) -> VeredictoRevision:
        # 1 · código, SIEMPRE primero (Regla 4)
        problemas_codigo = revisar_con_codigo(codigo, resultado, espera_figura)
        if traza:
            traza("revision_codigo", "revisor", None, subtarea=subtarea.id,
                 aprobado=not problemas_codigo, problemas=problemas_codigo)
        if problemas_codigo:
            return VeredictoRevision(False, problemas_codigo, fuente="codigo")

        # 2 · LLM, SOLO si la capa de código aprobó, y solo si hay con qué consultarlo
        if h200 is None:
            return VeredictoRevision(True, [], fuente="codigo")

        r = h200.chat(
            [{"role": "system", "content": _SISTEMA_LLM.format(
                descripcion=subtarea.descripcion, codigo=codigo[:4000],
                stdout=resultado.stdout[:1000],
                resultados=json.dumps(resultado.resultados, ensure_ascii=False))}],
            json_mode=True, max_tokens=self.MAX_TOKENS_LLM)
        if traza:
            traza("llm", "revisor", None, subtarea=subtarea.id,
                 tokens_entrada=r["uso"].get("prompt_tokens", 0),
                 tokens_salida=r["uso"].get("completion_tokens", 0),
                 latencia_s=r["latencia_s"], fin=r["fin"])

        if r["fin"] == "length" and not r["contenido"].strip():
            # Sin veredicto claro no se aprueba por omisión: lo contrario premiaría quedarse
            # sin tokens, igual que la Parte 0.a enseña a no premiar una respuesta vacía.
            return VeredictoRevision(False, ["el revisor LLM se quedó sin tokens razonando y "
                                            "no emitió veredicto: se trata como no aprobado"],
                                    fuente="llm")
        try:
            datos = json.loads(r["contenido"])
            aprobado = bool(datos.get("aprobado"))
            motivo = str(datos.get("motivo", "")) or "(sin motivo)"
        except (json.JSONDecodeError, AttributeError):
            return VeredictoRevision(False, ["el revisor LLM no devolvió un JSON válido: se "
                                            "trata como no aprobado"], fuente="llm")
        if traza:
            traza("revision_llm", "revisor", None, subtarea=subtarea.id,
                 aprobado=aprobado, motivo=motivo)
        return VeredictoRevision(aprobado, [] if aprobado else [motivo], fuente="llm")


if __name__ == "__main__":
    from solver.ejecutor import ResultadoEjecucion

    casos = {
        "limpio": (
            'from sklearn.model_selection import train_test_split\n'
            'Xtr, Xte, ytr, yte = train_test_split(X, y)\n'
            'modelo.fit(Xtr, ytr)\nacc = modelo.score(Xte, yte)\n',
            ResultadoEjecucion(0, "ok", "", 1.0, ["resultados.json"], False,
                              {"accuracy": 0.87})),
        "fuga": (
            'modelo.fit(X, y)\nacc = modelo.score(X, y)\n',
            ResultadoEjecucion(0, "ok", "", 1.0, ["resultados.json"], False,
                              {"accuracy": 1.0})),
        "nan": (
            'pass\n',
            ResultadoEjecucion(0, "ok", "", 1.0, ["resultados.json"], False,
                              {"metrica": float("nan")})),
        "sin_resultados": (
            'pass\n',
            ResultadoEjecucion(0, "ok", "", 1.0, [], False, None)),
        "codigo_de_error": (
            'raise ValueError("boom")\n',
            ResultadoEjecucion(1, "", "ValueError: boom", 0.2, [], False, None)),
        "agotado": (
            'while True: pass\n',
            ResultadoEjecucion(-1, "", "timeout", 60.0, [], True, None)),
    }
    for nombre, (codigo, resultado) in casos.items():
        print(f"{nombre:16s} → {revisar_con_codigo(codigo, resultado)}")
