"""redactor.py — el rol Redactor: escribe el entregable en el formato pedido.

Taller 03 v2. Qué revisa tu código, según la tabla de roles: «secciones, extensión, origen
de cada cifra». Y la Regla 5 obligatoria:

    «Origen de cada cifra. Antes de entregar, el código comprueba que cada cifra del
    entregable salió de una ejecución o está en el enunciado. Las que no, vuelven al
    redactor.»

El LLM redacta la prosa citando los resultados reales que ya calcularon el Programador y el
Ejecutor (y el enunciado, para lo que ya venía dado); el código decide qué cifras son
válidas —la unión de los `resultados.json` de las subtareas de cálculo y los números que ya
estaban en el enunciado— y comprueba, después de escribir, que ninguna cifra con dos o más
decimales se salió de esa lista. Si una se sale, no se "corrige" sola: se le devuelve al
redactor la cifra exacta y se reescribe (mismo patrón de reintento que Planificador,
Investigador y Programador).

No hay un rol aparte para las subtareas «conceptual»: su prosa también la escribe el
Redactor, en la misma pasada, grounded en el contexto que trajo el Investigador —el
enunciado no lista un noveno rol para eso, y separarlo solo fragmentaría el documento.

    from solver.redactor import Redactor, FormatoEntregable
    formato = FormatoEntregable("md", "reporte.md",
                                secciones=["Introducción", "Metodología", "Resultados",
                                           "Discusión", "Conclusiones"], max_palabras=1200)
    ruta = Redactor().redactar(plan, resultados, contextos, documento, formato, h200, salida)
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from solver.indexador import Fragmento
from solver.lector import DocumentoLeido
from solver.planificador import Plan

_NUM = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(\s?%)?(?![\w])")


def _numeros(texto: str) -> list[tuple[float, int]]:
    """(valor, decimales); los porcentajes se pasan a fracción —mismo criterio que
    `evaluar_solver.py`, para que lo que el Redactor exige de sí mismo sea al menos tan
    estricto como lo que el evaluador externo va a comprobar después."""
    salida = []
    for m in _NUM.finditer(texto):
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


def _valores_numericos(obj) -> list[float]:
    """Todos los `float`/`int` dentro de un `resultados.json`, recorriendo dict/list."""
    valores: list[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            valores += _valores_numericos(v)
    elif isinstance(obj, list):
        for v in obj:
            valores += _valores_numericos(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        valores.append(float(obj))
    return valores


def _sin_codigo(texto: str) -> str:
    return re.sub(r"```.*?```", "", texto, flags=re.S)


def verificar_procedencia(texto: str, pool: set[float]) -> list[str]:
    """Regla 5: cifras con dos o más decimales que no coinciden (con una tolerancia fina,
    por redondeo) con NINGÚN valor del `pool` —la unión de los resultados de ejecución y las
    cifras que ya traía el enunciado—. Una cifra con un decimal (p. ej. «0.3») se deja pasar:
    ahí vive la prosa normal («el 70 % de los datos»), y exigir procedencia de cada entero
    convertiría cualquier frase en una lista de citas."""
    plano = " ".join(_sin_codigo(texto).split())
    sospechosas = []
    for valor, dec in _numeros(plano):
        if dec < 2:
            continue
        tol = 0.5 * 10 ** -dec + 1e-9
        if not any(abs(valor - p) <= tol for p in pool):
            sospechosas.append(f"{valor:.{dec}f}")
    return sorted(set(sospechosas))


def _normalizar_encabezado(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def _posiciones_secciones(texto: str, secciones: list[str]) -> list[int]:
    objetivos = [_normalizar_encabezado(s) for s in secciones]
    posiciones = [-1] * len(secciones)
    pos = 0
    for linea in texto.splitlines(keepends=True):
        h = _normalizar_encabezado(linea.lstrip("#").strip())
        for i, obj in enumerate(objetivos):
            if posiciones[i] < 0 and h == obj:
                posiciones[i] = pos
        pos += len(linea)
    return posiciones


def _limpiar_markdown(texto: str) -> str:
    m = re.search(r"```(?:markdown|md)?\s*\n(.*)```\s*$", texto, re.S)
    return m.group(1) if m else texto


# ---------------------------------------------------------------- formato del entregable
@dataclass
class FormatoEntregable:
    tipo: str                                  # "md" | "pdf" | "ipynb"
    nombre_archivo: str
    secciones: list[str] = field(default_factory=list)   # vacío = sin exigir encabezados fijos
    max_palabras: int | None = None
    max_paginas: int | None = None              # solo aplica a "pdf"


class EntregableInvalido(RuntimeError):
    """El entregable no pasó la validación tras agotar los reintentos. Lleva el último
    texto y los problemas para que el orquestador decida (parcial/fallido)."""

    def __init__(self, mensaje: str, ultimo_texto: str, problemas: list[str]):
        super().__init__(mensaje)
        self.ultimo_texto = ultimo_texto
        self.problemas = problemas


# ---------------------------------------------------------------- deducir el formato pedido
_SISTEMA_FORMATO = """\
Del siguiente enunciado, extrae el formato EXACTO del entregable que pide. Responde SOLO
JSON: {{"tipo": "md"|"pdf"|"ipynb", "nombre_archivo": "...", "secciones": ["...", ...],
"max_palabras": entero o null, "max_paginas": entero o null}}.

"tipo" es "ipynb" si pide un notebook ejecutado, "pdf" si pide un PDF, "md" en cualquier
otro caso (incluido Markdown explícito). "secciones" es la lista de encabezados que exige,
en el orden pedido (lista vacía si no fija un orden). "nombre_archivo" es el nombre que
menciona el enunciado (p. ej. "reporte.md"); si no menciona ninguno, usa "reporte" con la
extensión de "tipo".

Enunciado:
{texto}
"""


def deducir_formato(documento: DocumentoLeido, h200, max_tokens: int = 4096) -> FormatoEntregable:
    """No se asume un formato fijo: el solver tiene que servir para cualquier enunciado, no
    solo para las tres tareas de práctica. Extraerlo con el LLM (en vez de con un regex
    atado a la redacción de este kit) es lo mismo que hace el Planificador con las
    subtareas: una extracción estructurada, no una regla escrita a mano por tarea."""
    r = h200.chat([{"role": "system", "content": _SISTEMA_FORMATO.format(texto=documento.texto[:6000])}],
                 json_mode=True, max_tokens=max_tokens)
    try:
        datos = json.loads(r["contenido"]) if r["contenido"].strip() else {}
    except json.JSONDecodeError:
        datos = {}
    tipo = str(datos.get("tipo", "md")).lower()
    if tipo not in {"md", "pdf", "ipynb"}:
        tipo = "md"
    extension = {"md": ".md", "pdf": ".pdf", "ipynb": ".ipynb"}[tipo]
    nombre = str(datos.get("nombre_archivo") or f"reporte{extension}")
    if not nombre.endswith(extension):
        nombre = Path(nombre).stem + extension
    max_palabras = datos.get("max_palabras")
    max_paginas = datos.get("max_paginas")
    return FormatoEntregable(
        tipo=tipo, nombre_archivo=nombre,
        secciones=[str(s) for s in datos.get("secciones", []) if isinstance(s, str)],
        max_palabras=int(max_palabras) if isinstance(max_palabras, (int, float)) else None,
        max_paginas=int(max_paginas) if isinstance(max_paginas, (int, float)) else None)


# ---------------------------------------------------------------- el prompt
_SISTEMA = """\
Escribes el entregable de una tarea académica a partir de lo que YA se calculó. No inventas
ninguna cifra: usa EXCLUSIVAMENTE los números que aparecen en «Lo que ya se calculó» de
abajo —cópialos tal cual, con sus decimales— o los que ya traía el enunciado. Si te falta un
dato para algo, dilo en prosa («no se dispone de X») en vez de inventarlo.

{restricciones}

Lo que ya se calculó o se investigó, subtarea por subtarea:
{subtareas}

Responde con el entregable completo en Markdown (encabezados con «##»), sin explicación
adicional ni delimitadores de bloque de código alrededor de todo el documento.
"""


def _bloque_subtareas(plan: Plan, resultados: dict[str, dict],
                      contextos: dict[str, list[Fragmento]]) -> str:
    bloques = []
    for sid in plan.orden_topologico():
        s = plan.por_id(sid)
        bloques.append(f"### «{s.id}» ({s.tipo}): {s.descripcion}")
        if s.tipo == "calculo":
            r = resultados.get(sid)
            bloques.append("Resultados reales (JSON, de la ejecución):\n" +
                          (json.dumps(r, ensure_ascii=False) if r else
                           "(esta subtarea no se completó: no hay cifras que usar)"))
        else:
            frags = contextos.get(sid) or []
            citas = "\n".join(f"- {f.cita()}: {' '.join(f.texto.split())[:400]}" for f in frags)
            bloques.append(f"Contexto del enunciado/curso:\n{citas or '(sin contexto adicional)'}")
        bloques.append("")
    return "\n".join(bloques)


# ---------------------------------------------------------------- el agente
class Redactor:
    MAX_REINTENTOS = 3
    MAX_TOKENS = 8192

    def redactar(self, plan: Plan, resultados: dict[str, dict],
                contextos: dict[str, list[Fragmento]], documento: DocumentoLeido,
                formato: FormatoEntregable, h200, carpeta_salida: str | Path,
                traza=None, codigos: dict[str, str] | None = None) -> Path:
        pool: set[float] = set(v for v, _ in _numeros(documento.texto))
        for r in resultados.values():
            if r:
                pool.update(_valores_numericos(r))

        restricciones = []
        if formato.secciones:
            restricciones.append("Usa EXACTAMENTE estos encabezados, en este orden: " +
                                " → ".join(formato.secciones))
        if formato.max_palabras:
            restricciones.append(f"No más de {formato.max_palabras} palabras en total "
                                "(sin contar bloques de código).")
        if formato.max_paginas:
            restricciones.append(f"Sé conciso: tiene que caber en {formato.max_paginas} "
                                "página(s) de PDF.")

        mensajes = [{"role": "system", "content": _SISTEMA.format(
            restricciones="\n".join(f"- {r}" for r in restricciones) or "(sin restricciones adicionales de formato)",
            subtareas=_bloque_subtareas(plan, resultados, contextos))}]

        problemas: list[str] = []
        texto = ""
        for intento in range(1, self.MAX_REINTENTOS + 1):
            turno = list(mensajes)
            if problemas:
                turno.append({"role": "user", "content":
                              "El entregable anterior no pasó la validación:\n" +
                              "\n".join(f"- {p}" for p in problemas) +
                              "\nCorrígelo y responde con el entregable completo de nuevo."})
            else:
                turno.append({"role": "user", "content": "Escribe el entregable."})

            r = h200.chat(turno, max_tokens=self.MAX_TOKENS)
            if traza:
                traza("llm", "redactor", intento,
                     tokens_entrada=r["uso"].get("prompt_tokens", 0),
                     tokens_salida=r["uso"].get("completion_tokens", 0),
                     latencia_s=r["latencia_s"], fin=r["fin"])

            if r["fin"] == "length" and not r["contenido"].strip():
                problemas = ["el modelo se quedó sin tokens razonando y no escribió el entregable"]
                continue

            texto = _limpiar_markdown(r["contenido"])
            problemas = self._validar_texto(texto, formato, pool)

            ruta_candidata = None
            if not problemas and formato.tipo == "pdf":
                ruta_candidata = self._exportar_pdf(texto, formato, Path(carpeta_salida))
                n_paginas = self._contar_paginas(ruta_candidata)
                if formato.max_paginas and n_paginas > formato.max_paginas:
                    problemas = [f"el PDF tiene {n_paginas} página(s); el máximo es "
                                f"{formato.max_paginas}: hay que acortar el texto"]
            elif not problemas and formato.tipo == "ipynb":
                ruta_candidata, problemas_nb = self._exportar_notebook(
                    texto, formato, Path(carpeta_salida), codigos or {})
                if problemas_nb:
                    problemas = problemas_nb

            if traza:
                traza("validacion_entregable", "redactor", intento,
                     aprobado=not problemas, problemas=problemas)
            if not problemas:
                if formato.tipo in {"pdf", "ipynb"}:
                    return ruta_candidata
                return self._escribir_md(texto, formato, Path(carpeta_salida))

        raise EntregableInvalido(
            f"el entregable no pasó la validación tras {self.MAX_REINTENTOS} intento(s): "
            f"{problemas}", texto, problemas)

    def _validar_texto(self, texto: str, formato: FormatoEntregable, pool: set[float]) -> list[str]:
        problemas: list[str] = []
        if formato.secciones:
            posiciones = _posiciones_secciones(texto, formato.secciones)
            faltan = [s for s, p in zip(formato.secciones, posiciones) if p < 0]
            if faltan:
                problemas.append(f"faltan secciones: {faltan}")
            elif posiciones != sorted(posiciones):
                problemas.append("las secciones están presentes pero fuera de orden")
        if formato.max_palabras:
            n = len(_sin_codigo(texto).split())
            if n > formato.max_palabras:
                problemas.append(f"{n} palabras: por encima del máximo de {formato.max_palabras}")
        if sospechosas := verificar_procedencia(texto, pool):
            problemas.append("cifras sin origen verificable —no están en ningún resultados.json "
                            f"ni en el enunciado—: {sospechosas}")
        return problemas

    def _escribir_md(self, texto: str, formato: FormatoEntregable, carpeta_salida: Path) -> Path:
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        ruta = carpeta_salida / formato.nombre_archivo
        ruta.write_text(texto, encoding="utf-8")
        return ruta

    def _exportar_pdf(self, texto_md: str, formato: FormatoEntregable, carpeta_salida: Path) -> Path:
        """El mismo patrón que `generar_enunciados.py`: PyMuPDF (`fitz.Story`), sin
        navegador ni LaTeX. Reutiliza la conversión Markdown→HTML del kit para que el
        Redactor y los enunciados de práctica se vean igual."""
        import pymupdf as fitz
        from markdown_it import MarkdownIt

        carpeta_salida.mkdir(parents=True, exist_ok=True)
        html = MarkdownIt("commonmark").enable("table").render(texto_md)
        historia = fitz.Story(html=html)
        ruta = carpeta_salida / formato.nombre_archivo
        escritor = fitz.DocumentWriter(str(ruta))
        a4 = fitz.paper_rect("a4")
        caja = a4 + (56, 56, -56, -56)
        mas = True
        while mas:
            dispositivo = escritor.begin_page(a4)
            mas, _ = historia.place(caja)
            historia.draw(dispositivo)
            escritor.end_page()
        escritor.close()
        return ruta

    @staticmethod
    def _contar_paginas(ruta_pdf: Path) -> int:
        import pymupdf
        return pymupdf.open(ruta_pdf).page_count

    def _exportar_notebook(self, texto_md: str, formato: FormatoEntregable,
                          carpeta_salida: Path, codigos: dict[str, str]) -> tuple[Path | None, list[str]]:
        """Un notebook **ejecutado**, no solo escrito: una celda de Markdown por sección (lo
        que pide el formato de la Tarea B) y, al final, una celda de código por subtarea de
        cálculo con el MISMO script que ya aprobaron el sandbox y el revisor —no se
        regenera el código aquí, se reutiliza el que ya se ejecutó de verdad—. Se ejecuta
        con `nbclient` antes de guardar: un notebook con celdas sin correr no es un
        entregable, es un borrador."""
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError

        nb = nbformat.v4.new_notebook()
        for bloque in re.split(r"\n(?=##\s)", texto_md):
            if bloque.strip():
                nb.cells.append(nbformat.v4.new_markdown_cell(bloque.strip()))
        if codigos:
            nb.cells.append(nbformat.v4.new_markdown_cell("## Código"))
            for sid, codigo in codigos.items():
                nb.cells.append(nbformat.v4.new_markdown_cell(f"### `{sid}`"))
                nb.cells.append(nbformat.v4.new_code_cell(codigo))

        carpeta_salida.mkdir(parents=True, exist_ok=True)
        ruta = carpeta_salida / formato.nombre_archivo
        cliente = NotebookClient(nb, timeout=120, resources={"metadata": {"path": str(carpeta_salida)}})
        try:
            cliente.execute()
        except CellExecutionError as err:
            return None, [f"una celda de código falló al ejecutar el notebook: {err}"]
        finally:
            nbformat.write(nb, ruta)

        errores = [c for c in nb.cells if c.get("cell_type") == "code"
                  for o in c.get("outputs", []) if o.get("output_type") == "error"]
        if errores:
            return ruta, [f"{len(errores)} celda(s) de código con error tras ejecutar"]
        return ruta, []
