"""lector.py — el rol Lector: PDF → texto por página, con tablas.

Taller 03 v2, Parte 1. Lo que este agente hace NO pasa por el LLM: es código
determinístico, como pide la tabla de roles («Qué revisa tu código» para el Lector es
«texto pegado o partido, PDF escaneado»). El resto del pipeline (Indexador, Planificador,
Investigador…) consume su salida, nunca el PDF crudo.

    from solver.lector import Lector
    doc = Lector().leer("enunciados/tarea-a-generativo-discriminativo.pdf")
    doc.texto            # todo el documento, normalizado
    doc.secciones         # partido por encabezados, con su página de inicio
    doc.advertencias      # lo que el código detectó: texto pegado/partido, páginas escaneadas

Diseño de la detección de encabezados: ninguna tarea trae su propio regex a mano. Un PDF no
conserva los `#` de Markdown (ver `evaluar_solver.py::posicion_seccion`), así que un
encabezado en el texto extraído es una línea corta, sin puntuación final, que empieza con
una palabra típica de sección («Parte», «Sección», «Tarea», «Capítulo», «Ejercicio») o con
una numeración («1.», «2.3)»). La heurística es deliberadamente conservadora: una sección de
más no rompe nada (el investigador la recupera igual), una línea de prosa mal marcada como
encabezado sí perdería contexto.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# ---------------------------------------------------------------- normalización
_ESPACIOS = re.compile(r"[ \t]+")
# «pala-\nbra» → «palabra»: el guion de fin de línea es un artefacto de maquetación, no una
# palabra compuesta. Solo se reúne si lo que sigue es minúscula: «2024-\n2025» no se toca.
_GUION_CORTE = re.compile(r"(\w)-\n([a-záéíóúñü])")


def _normalizar(texto: str) -> tuple[str, int]:
    """Normaliza unicode y reúne palabras partidas por guion de línea.

    Devuelve (texto_normalizado, cuántas palabras reunió): ese conteo es la evidencia de
    «texto partido» que reporta el Lector, no una opinión.
    """
    texto = unicodedata.normalize("NFKC", texto)
    texto, n_partidas = _GUION_CORTE.subn(r"\1\2", texto)
    texto = _ESPACIOS.sub(" ", texto)
    return texto, n_partidas


# ---------------------------------------------------------------- detección de encabezados
_PATRONES_ENCABEZADO = (
    re.compile(r"^(parte|secci[oó]n|punto|ejercicio|pregunta|cap[ií]tulo|anexo)\s+\d+\w*\b", re.I),
    re.compile(r"^tarea\s+[a-z0-9]+\b", re.I),
    re.compile(r"^\d+(\.\d+)*[.)]\s+\S"),
)
_MAX_LARGO_ENCABEZADO = 90


def _es_encabezado(linea: str) -> bool:
    l = linea.strip()
    if not l or len(l) > _MAX_LARGO_ENCABEZADO or l.endswith((".", ",", ";", ":")):
        return False
    return any(p.match(l) for p in _PATRONES_ENCABEZADO)


def _id_seccion(titulo: str, indice: int) -> str:
    m = re.match(r"^(parte|secci[oó]n|tarea|cap[ií]tulo)\s+(\w+)", titulo, re.I)
    return f"{m.group(1).lower()}-{m.group(2).lower()}" if m else f"seccion-{indice}"


# ---------------------------------------------------------------- estructuras de salida
@dataclass
class Tabla:
    pagina: int
    filas: list[list[str | None]]

    def como_texto(self) -> str:
        return "\n".join(" | ".join("" if c is None else str(c) for c in fila) for fila in self.filas)


@dataclass
class Pagina:
    numero: int
    texto: str
    tablas: list[Tabla] = field(default_factory=list)
    probable_escaneada: bool = False


@dataclass
class Seccion:
    id: str
    titulo: str
    texto: str
    pagina_inicio: int


@dataclass
class DocumentoLeido:
    ruta: Path
    paginas: list[Pagina]
    secciones: list[Seccion]
    texto: str
    advertencias: list[str]

    def resumen(self) -> str:
        filas = [f"{self.ruta.name}: {len(self.paginas)} página(s), {len(self.secciones)} sección(es)"]
        for s in self.secciones:
            filas.append(f"  [{s.id}] {s.titulo}  (p.{s.pagina_inicio}, {len(s.texto)} car.)")
        if self.advertencias:
            filas.append("  advertencias:")
            filas += [f"    - {a}" for a in self.advertencias]
        return "\n".join(filas)


# ---------------------------------------------------------------- el agente
class Lector:
    """Convierte un PDF de enunciado en un `DocumentoLeido`, con sus comprobaciones de código."""

    UMBRAL_TEXTO_ESCANEADO = 20      # caracteres crudos; menos que esto con imágenes en la página
    UMBRAL_PALABRA_PEGADA = 28       # letras seguidas sin espacio
    FRACCION_PEGADO_ALERTA = 0.02    # 2 % de las palabras así ya es sospechoso
    MIN_PARTIDAS_ALERTA = 3          # menos que esto es ruido normal de maquetación

    def leer(self, ruta: str | Path) -> DocumentoLeido:
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"no existe el PDF: {ruta}")

        doc = pymupdf.open(ruta)
        paginas: list[Pagina] = []
        advertencias: list[str] = []
        partidas_total = 0

        for i, page in enumerate(doc, start=1):
            crudo = page.get_text()
            texto, n_partidas = _normalizar(crudo)
            partidas_total += n_partidas
            escaneada = len(crudo.strip()) < self.UMBRAL_TEXTO_ESCANEADO and bool(page.get_images())
            if escaneada:
                advertencias.append(f"página {i}: probable PDF escaneado (sin texto extraíble "
                                    "pero con imágenes); el resto del pipeline no tendrá nada que leer ahí")
            paginas.append(Pagina(numero=i, texto=texto,
                                  tablas=self._tablas(page, i), probable_escaneada=escaneada))

        if partidas_total >= self.MIN_PARTIDAS_ALERTA:
            advertencias.append(f"texto partido: {partidas_total} palabra(s) cortadas por guion "
                                "de línea (corregidas automáticamente al unirlas)")
        advertencias += self._revisar_pegado(paginas)

        secciones = self._partir_en_secciones(paginas)
        if len(secciones) <= 1:
            advertencias.append("no se detectaron encabezados de sección: el documento entero "
                                "queda como una sola sección (el investigador pierde granularidad)")

        texto_completo = "\n".join(p.texto for p in paginas)
        return DocumentoLeido(ruta=ruta, paginas=paginas, secciones=secciones,
                              texto=texto_completo, advertencias=advertencias)

    # ---------------------------------------------------------------- tablas
    def _tablas(self, page, numero_pagina: int) -> list[Tabla]:
        try:
            hallazgo = page.find_tables()
        except Exception:
            return []
        # find_tables() confunde párrafos de dos columnas con una tabla de 1 fila: una tabla
        # de verdad tiene al menos un encabezado y una fila de datos.
        return [Tabla(pagina=numero_pagina, filas=filas) for t in hallazgo.tables
                if len(filas := t.extract()) >= 2]

    # ---------------------------------------------------------------- texto pegado
    def _revisar_pegado(self, paginas: list[Pagina]) -> list[str]:
        palabras = re.findall(r"[^\W\d_]+", " ".join(p.texto for p in paginas), re.UNICODE)
        if not palabras:
            return []
        largas = [p for p in palabras if len(p) >= self.UMBRAL_PALABRA_PEGADA]
        if largas and len(largas) / len(palabras) > self.FRACCION_PEGADO_ALERTA:
            ejemplos = ", ".join(largas[:3])
            return [f"texto posiblemente pegado: {len(largas)} palabra(s) de "
                    f"{self.UMBRAL_PALABRA_PEGADA}+ letras sin espacios (p. ej. {ejemplos})"]
        return []

    # ---------------------------------------------------------------- secciones
    def _partir_en_secciones(self, paginas: list[Pagina]) -> list[Seccion]:
        lineas: list[tuple[int, str]] = []       # (num_pagina, línea)
        for p in paginas:
            lineas += [(p.numero, l) for l in p.texto.split("\n")]

        indices_encabezado = [i for i, (_, l) in enumerate(lineas) if _es_encabezado(l)]
        if not indices_encabezado:
            texto = "\n".join(l for _, l in lineas)
            return [Seccion(id="documento", titulo=paginas[0].texto[:60] if paginas else "",
                            texto=texto, pagina_inicio=1)] if texto.strip() else []

        secciones: list[Seccion] = []
        if indices_encabezado[0] > 0:
            preambulo = "\n".join(l for _, l in lineas[: indices_encabezado[0]])
            if preambulo.strip():
                secciones.append(Seccion(id="preambulo", titulo="(preámbulo)",
                                         texto=preambulo, pagina_inicio=lineas[0][0]))

        limites = indices_encabezado + [len(lineas)]
        for n, (inicio, fin) in enumerate(zip(limites, limites[1:]), start=1):
            titulo = lineas[inicio][1].strip()
            cuerpo = "\n".join(l for _, l in lineas[inicio:fin])
            secciones.append(Seccion(id=_id_seccion(titulo, n), titulo=titulo,
                                     texto=cuerpo, pagina_inicio=lineas[inicio][0]))
        return secciones


if __name__ == "__main__":
    import sys

    for ruta in sys.argv[1:] or [str(Path(__file__).resolve().parents[1] / "enunciados" /
                                      "tarea-a-generativo-discriminativo.pdf")]:
        print(Lector().leer(ruta).resumen())
        print()
