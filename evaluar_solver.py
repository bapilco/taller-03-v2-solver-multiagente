#!/usr/bin/env python3
"""evaluar_solver.py — corre un solver sobre el golden set de tareas y escribe el CSV crudo.

    python evaluar_solver.py --solver solver:Solver --ruta ../mi-solver
    python evaluar_solver.py --solver solver:Solver --ruta ../mi-solver --solo A C
    python evaluar_solver.py --solo-evaluar corridas/          # re-evalúa sin correr nada

Una tarea puede entrar como archivo (`pdf`) o como carpeta (`entrada`): un paquete con su
notebook, su contrato y su verificador, que el solver completa en su sitio.

El contrato del solver es el del Taller 03 v2, Parte 1:

    Solver().solve(ruta_pdf: str, salida: str) -> dict
        status      "completado" | "parcial" | "fallido"
        entregables lista de rutas que el solver considera la entrega
        subtareas   [{"id", "tipo", "status", "intentos"}]
        usage       {"tokens_entrada": int, "tokens_salida": int}
        model       el id servido, leído del endpoint
        trace       la ruta de la traza

Cada comprobación del golden set se decide **sin juicio**: un archivo existe o no, una
cifra coincide con su `verdad_py` dentro de su tolerancia o no. La única que mira más de
un archivo es `procedencia`: qué fracción de las cifras del entregable aparece también en
algo que escribió una ejecución (JSON, CSV, logs, salidas del notebook). Una cifra que
solo existe en el reporte es una cifra que alguien escribió, no que alguien midió.

Escribe `resultados_solver.csv` (una fila por comprobación) y `resumen_solver.csv` (una
fila por tarea). Las tablas del informe se derivan de esos dos archivos.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import time
import traceback
from glob import glob
from pathlib import Path

AQUI = Path(__file__).resolve().parent
NUM = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(\s?%)?(?![\w])")


# ---------------------------------------------------------------- lectura de entregables
def texto_de(ruta: Path, con_salidas: bool = True) -> str:
    if ruta.suffix == ".pdf":
        import pymupdf
        import unicodedata
        return unicodedata.normalize("NFKC", "\n".join(p.get_text() for p in pymupdf.open(ruta)))
    if ruta.suffix == ".ipynb":
        nb = json.loads(ruta.read_text(encoding="utf-8"))
        partes = []
        for c in nb.get("cells", []):
            fuente = "".join(c.get("source", ""))
            if c["cell_type"] == "markdown":
                partes.append(fuente)
            elif c["cell_type"] == "code" and con_salidas:
                for o in c.get("outputs", []):
                    if "text" in o:
                        partes.append("".join(o["text"]))
                    datos = o.get("data", {})
                    if "text/plain" in datos:
                        partes.append("".join(datos["text/plain"]))
                    if "text/markdown" in datos:
                        partes.append("".join(datos["text/markdown"]))
        return "\n".join(partes)
    return ruta.read_text(encoding="utf-8", errors="replace")


def texto_entrada(ruta: Path) -> str:
    """El texto de lo que recibió el solver: un archivo, o todo lo legible de una carpeta
    (enunciados, README y las fuentes del notebook), para saber qué cifras ya venían dadas."""
    if ruta.is_file():
        return texto_de(ruta)
    partes = []
    for p in sorted(ruta.rglob("*")):
        if any(x in p.parts for x in (".venv", "output", "__pycache__")) or not p.is_file():
            continue
        if p.suffix in {".pdf", ".md", ".txt"}:
            partes.append(texto_de(p))
        elif p.suffix == ".ipynb":
            nb = json.loads(p.read_text(encoding="utf-8"))
            partes += ["".join(c.get("source", "")) for c in nb.get("cells", [])]
    return "\n".join(partes)


def _contrato(check: dict, base: Path) -> dict:
    return json.loads((base / check["contrato"]).read_text(encoding="utf-8"))


def sin_codigo(texto: str) -> str:
    return re.sub(r"```.*?```", "", texto, flags=re.S)


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def posicion_seccion(texto: str, seccion: str) -> int:
    """Primer renglón que ES el encabezado: con o sin `#`, con o sin número delante, sin
    distinguir acentos ni puntuación (un PDF no conserva los `#`; un `\\b` tras «)» no casa)."""
    objetivo = _norm(seccion)
    pos = 0
    for linea in texto.splitlines(keepends=True):
        h = re.sub(r"^\d+ ", "", _norm(linea.lstrip("# ")))
        if h and len(h) <= len(objetivo) + 40 and (h == objetivo or h.startswith(objetivo + " ")):
            return pos
        pos += len(linea)
    return -1


def numeros(texto: str) -> list[tuple[float, int]]:
    """(valor, decimales) de cada cifra; los porcentajes se pasan a fracción."""
    salida = []
    for m in NUM.finditer(texto):
        crudo, pct = m.group(1).replace(",", "."), m.group(2)
        try:
            v = float(crudo)
        except ValueError:
            continue
        dec = len(crudo.split(".")[1]) if "." in crudo else 0
        if pct:
            salida.append((v / 100, dec + 2))
        salida.append((v, dec))
    return salida


def coincide(valor: float, verdad: float, tol: float) -> bool:
    return abs(valor - verdad) <= tol + 1e-12


def verdad_de(check: dict) -> float:
    codigo = check["verdad_py"]
    codigo = "\n".join(codigo) if isinstance(codigo, list) else codigo
    ambito: dict = {}
    exec(codigo, ambito)  # código del golden set, versionado con el taller: no del solver
    return float(ambito["verdad"])


# ---------------------------------------------------------------- comprobaciones
def entregable(salida: Path, patron: str) -> Path | None:
    hallados = sorted(salida.glob(patron)) or sorted(salida.rglob(patron))
    return hallados[0] if hallados else None


def artefactos(salida: Path, excluir: set[Path]) -> list[tuple[float, int]]:
    nums: list[tuple[float, int]] = []
    for ext in ("json", "csv", "txt", "log", "out"):
        for f in salida.rglob(f"*.{ext}"):
            # Lo que no escribió una ejecución no respalda nada: la traza, el grafo (sus
            # descripciones copian cifras del enunciado y del curso), el plan y el resumen.
            if f in excluir or f.stat().st_size > 5_000_000 or \
                    re.search(r"traza|grafo|comunidades|plan|resumen|cache|enunciado|\.mpl", str(f.relative_to(salida))):
                continue
            nums += numeros(f.read_text(encoding="utf-8", errors="replace"))
    for nb in salida.rglob("*.ipynb"):
        datos = json.loads(nb.read_text(encoding="utf-8"))
        for c in datos.get("cells", []):
            for o in c.get("outputs", []):
                nums += numeros("".join(o.get("text", "")) +
                                "".join(o.get("data", {}).get("text/plain", "")))
    return nums


def comprobar(check: dict, salida: Path, tarea: dict, texto_enunciado: str,
              base: Path = AQUI) -> tuple[bool, str]:
    tipo = check["tipo"]
    principal = entregable(salida, tarea["entregable"])
    if tipo == "archivo":
        hallados = [Path(p) for p in glob(str(salida / check["patron"]), recursive=True)]
        if check.get("o_imagen_en_ipynb") and principal and principal.suffix == ".ipynb":
            nb = principal.read_text(encoding="utf-8")
            if '"image/png"' in nb:
                return True, "imagen embebida en el notebook"
        n = len(hallados)
        return n >= check.get("minimo", 1), f"{n} archivo(s) con {check['patron']}"
    if principal is None:
        return False, f"no hay entregable {tarea['entregable']}"
    if tipo in {"celdas_intactas", "celdas_completas"}:
        import ast as _ast
        import hashlib
        contrato = _contrato(check, base)
        celdas = json.loads(principal.read_text(encoding="utf-8"))["cells"]
        reglas = contrato["cells"]
        if tipo == "celdas_intactas":
            if [c.get("id") for c in celdas] != [r["id"] for r in reglas]:
                return False, "los ids o el orden de las celdas cambiaron"
            cambiadas = [r["id"] for c, r in zip(celdas, reglas) if r.get("sha256") and
                         hashlib.sha256("".join(c.get("source", "")).encode()).hexdigest() != r["sha256"]]
            return not cambiadas, f"{len(cambiadas)} celdas protegidas cambiadas {cambiadas[:5]}"
        excepto = set(check.get("excepto", []))
        pendientes = []
        for c, r in zip(celdas, reglas):
            if r.get("sha256") is not None or excepto & set(r.get("tags", [])):
                continue
            fuente = "".join(c.get("source", ""))
            if "codigo" in r.get("tags", []):
                try:
                    if any(isinstance(n, _ast.Constant) and n.value is Ellipsis for n in _ast.walk(_ast.parse(fuente))):
                        pendientes.append(r["id"])
                except SyntaxError:
                    pendientes.append(r["id"] + " (no compila)")
            elif re.search(r"ESCRIBE AQU[IÍ]", fuente) or len(fuente.split("\n\n", 1)[-1].strip()) < 10:
                pendientes.append(r["id"])
        return not pendientes, f"pendientes: {pendientes}" if pendientes else "todas completas"
    if tipo == "verificador_externo":
        import shutil
        import subprocess
        import tempfile
        paquete = base / check["paquete"]
        contrato = _contrato(check, base) if check.get("contrato") else None
        editables = {r["id"] for r in contrato["cells"] if r.get("sha256") is None} if contrato else None
        with tempfile.TemporaryDirectory(prefix="eval-ver-") as tmp:
            copia = Path(tmp) / "paquete"
            shutil.copytree(paquete, copia, ignore=shutil.ignore_patterns("output", ".venv", "__pycache__"))
            nb = json.loads(principal.read_text(encoding="utf-8"))
            for c in nb["cells"]:          # los reemplazos declarados, solo en celdas editables
                if editables is None or c.get("id") in editables:
                    fuente = "".join(c.get("source", ""))
                    for viejo, nuevo in check.get("reemplazos", {}).items():
                        fuente = fuente.replace(viejo, nuevo)
                    c["source"] = fuente
            (copia / principal.name).write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
            orden = [sys.executable if x == "python" else x for x in check["comando"]]
            p = subprocess.run(orden, cwd=copia, capture_output=True, text=True, timeout=600)
            texto = p.stdout + p.stderr
        ok = p.returncode == 0 and check["espera"] in texto
        ultima = [l for l in texto.strip().splitlines() if l.strip()][-1:] or [""]
        return ok, f"rc={p.returncode}; {'«' + check['espera'] + '»' if ok else ultima[0][:150]}"
    texto = texto_de(principal)
    if tipo == "secciones":
        posiciones = [posicion_seccion(texto, s) for s in check["lista"]]
        faltan = [s for s, p in zip(check["lista"], posiciones) if p < 0]
        if faltan:
            return False, f"faltan: {faltan}"
        if check.get("en_orden") and posiciones != sorted(posiciones):
            return False, "las secciones están fuera de orden"
        return True, "todas, en orden"
    if tipo == "palabras_max":
        n = len(sin_codigo(texto).split())
        return n <= check["valor"], f"{n} palabras"
    if tipo == "paginas_max":
        import pymupdf
        n = pymupdf.open(principal).page_count
        return n <= check["valor"], f"{n} páginas"
    if tipo == "contiene":
        faltan = [x for x in check["lista"] if not re.search(rf"\b{re.escape(x)}\b", texto)]
        return not faltan, f"faltan {faltan}" if faltan else "todos presentes"
    if tipo == "ipynb_ejecutado":
        nb = json.loads(principal.read_text(encoding="utf-8"))
        codigo = [c for c in nb["cells"] if c["cell_type"] == "code"]
        sin_correr = sum(c.get("execution_count") is None for c in codigo)
        errores = sum(o.get("output_type") == "error" for c in codigo for o in c.get("outputs", []))
        return (bool(codigo) and not sin_correr and not errores,
                f"{len(codigo)} celdas de código, {sin_correr} sin ejecutar, {errores} con error")
    if tipo in {"cifra", "cifra_presente"}:
        verdad = verdad_de(check)
        tol = check.get("tolerancia", 0.005)
        plano = " ".join(sin_codigo(texto).split())
        if tipo == "cifra_presente":
            candidatos = numeros(plano)
        else:
            candidatos = []
            for m in re.finditer(check["etiqueta"], plano):
                candidatos += numeros(plano[m.end(): m.end() + 160])
        ok = any(coincide(v, verdad, tol) for v, _ in candidatos)
        return ok, f"verdad={verdad:.4f} ±{tol}; {'hallada' if ok else 'no hallada'}"
    if tipo == "procedencia":
        dados = {v for v, _ in numeros(texto_enunciado)}
        propios = [(v, d) for v, d in numeros(sin_codigo(texto_de(principal, con_salidas=False)))
                   if d >= 2 and v not in dados]
        if not propios:
            return True, "sin cifras con dos o más decimales"
        medidos = artefactos(salida, {principal})
        respaldados = sum(any(abs(a - v) <= 0.5 * 10 ** -d + 1e-9 for a, _ in medidos)
                          for v, d in propios)
        frac = respaldados / len(propios)
        return frac >= check["minimo"], f"{respaldados}/{len(propios)} cifras respaldadas ({frac:.2f})"
    return False, f"tipo desconocido: {tipo}"


# ---------------------------------------------------------------- bucle
def cargar_solver(spec: str, ruta: str | None):
    if ruta:
        sys.path.insert(0, str(Path(ruta).resolve()))
    modulo, clase = spec.split(":")
    return getattr(importlib.import_module(modulo), clase)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default=str(AQUI / "golden_tareas.json"))
    ap.add_argument("--solver", help="modulo:Clase")
    ap.add_argument("--ruta", help="carpeta donde vive el módulo del solver")
    ap.add_argument("--corridas", default="corridas", help="una subcarpeta por tarea")
    ap.add_argument("--solo", nargs="*", help="ids de tarea")
    ap.add_argument("--solo-evaluar", metavar="CORRIDAS", help="no corre el solver")
    ap.add_argument("--salida", default="resultados_solver.csv")
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    tareas = [t for t in golden["tareas"] if not args.solo or t["id"] in args.solo]
    corridas = Path(args.solo_evaluar or args.corridas).resolve()
    Solver = None if args.solo_evaluar else cargar_solver(args.solver, args.ruta)

    filas, resumen = [], []
    for t in tareas:
        base = Path(args.golden).resolve().parent
        pdf = (base / (t.get("entrada") or t["pdf"])).resolve()
        salida = corridas / f"tarea-{t['id']}"
        info: dict = {"status": "solo_evaluado"}
        t0 = time.perf_counter()
        if Solver is not None:
            salida.mkdir(parents=True, exist_ok=True)
            try:
                info = Solver().solve(str(pdf), str(salida)) or {}
            except Exception as err:  # el solver murió: la fila lo dice, el bucle sigue
                info = {"status": "excepcion", "error": f"{type(err).__name__}: {err}"}
                (salida / "excepcion.txt").write_text(traceback.format_exc())
        dur = round(time.perf_counter() - t0, 1)
        enunciado = texto_entrada(pdf)
        aprobadas = 0
        for c in t["checks"]:
            try:
                ok, detalle = comprobar(c, salida, t, enunciado, base)
            except Exception as err:
                ok, detalle = False, f"error al comprobar: {type(err).__name__}: {err}"
            aprobadas += ok
            filas.append({"tarea": t["id"], "check": c["id"], "tipo": c["tipo"],
                          "ok": int(ok), "detalle": detalle})
            print(f"  {t['id']} {c['id']:<4} {c['tipo']:<15} {'✓' if ok else '✗'}  {detalle}")
        subt = info.get("subtareas", [])
        uso = info.get("usage", {})
        resumen.append({
            "tarea": t["id"], "status": info.get("status"), "model": info.get("model", ""),
            "aprobadas": aprobadas, "total": len(t["checks"]),
            "subtareas": len(subt),
            "intentos_codigo": sum(s.get("intentos", 0) for s in subt),
            "tokens_entrada": uso.get("tokens_entrada", 0),
            "tokens_salida": uso.get("tokens_salida", 0),
            "duracion_s": dur, "error": info.get("error", "")})
        print(f"Tarea {t['id']}: {aprobadas}/{len(t['checks'])} · status={info.get('status')} · {dur} s\n")

    destino = Path(args.salida)
    with destino.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    with destino.with_name(destino.name.replace("resultados", "resumen")).open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0]))
        w.writeheader(); w.writerows(resumen)
    total = sum(r["aprobadas"] for r in resumen), sum(r["total"] for r in resumen)
    print(f"Comprobaciones aprobadas: {total[0]}/{total[1]} → {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
