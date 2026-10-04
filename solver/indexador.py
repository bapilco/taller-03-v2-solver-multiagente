"""indexador.py — el rol Indexador: arma el grafo del enunciado y del material del curso
(GraphRAG), Regla 2 del Taller 03 v2.

    «Las secciones son nodos y cada referencia (“Parte 1”, “la pregunta anterior”) es una
    conexión depende_de que se extrae con una regla, no con el LLM. Encima van las entidades
    que extrae el LLM (datasets, métodos, métricas, conceptos), unidas con las del material
    del curso. El investigador entrega a cada subtarea su sección literal y las de sus
    dependencias, cada fragmento con su cita.»

Dos capas, construidas por separado:

    CAPA BASE (fija, sin LLM)     nodos = secciones del enunciado y de las notas del curso
                                  aristas depende_de = referencias cruzadas, por regex
    CAPA DE ENTIDADES (con LLM)  nodos = entidades (dataset, método, métrica, concepto)
                                  aristas menciona = sección → entidad
                                  la misma entidad en dos documentos es el MISMO nodo: ahí
                                  se cruzan el enunciado y el curso.

La capa base nunca depende del LLM: un GraphRAG cuya navegación dependiera de que el modelo
«decida» seguir una referencia repetiría el fallo de la Parte 0.b. La capa de entidades es
lo que el LLM aporta y la regla no puede: saber que «TF-IDF» y «la línea base léxica» son
la misma idea.

    from solver.indexador import Indexador
    idx = Indexador()
    idx.agregar_documento(doc, fuente_id="tarea-a", fuente="enunciado")
    idx.agregar_notas_curso(Path("../notas-teoricas"))
    idx.contexto_para("¿con qué datos se calcula la curva de aprendizaje?", k=2)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from solver.lector import DocumentoLeido

# ---------------------------------------------------------------- referencias (regla, no LLM)
_REF_NUMERADA = re.compile(
    r"\b(parte|secci[oó]n|ejercicio|punto|tarea|cap[ií]tulo)\s+([a-z0-9]+)\b", re.I)
_REF_RELATIVA = re.compile(
    r"\b(la|el)\s+(parte|secci[oó]n|ejercicio|punto|pregunta|apartado)\s+anterior\b"
    r"|\blo anterior\b", re.I)
_ENCABEZADO_MD = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _despojar(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _alias_de_id(id_seccion: str) -> tuple[str, str] | None:
    """«parte-3» → ("parte", "3"); «preambulo» (sin numeración) → None."""
    tipo, _, numero = id_seccion.partition("-")
    if not numero:
        return None
    return _despojar(tipo), numero.lower()


# ---------------------------------------------------------------- nodos de las notas del curso
def _secciones_markdown(texto: str) -> list[tuple[str, str, int]]:
    """(titulo, cuerpo, linea_inicio) partido por encabezados Markdown reales (con `#`)."""
    lineas = texto.splitlines()
    indices = [i for i, l in enumerate(lineas) if _ENCABEZADO_MD.match(l)]
    if not indices:
        return [("(documento completo)", texto, 1)]
    secciones = []
    limites = indices + [len(lineas)]
    for inicio, fin in zip(limites, limites[1:]):
        m = _ENCABEZADO_MD.match(lineas[inicio])
        titulo = m.group(2) if m else lineas[inicio].strip()
        cuerpo = "\n".join(lineas[inicio:fin])
        secciones.append((titulo, cuerpo, inicio + 1))
    return secciones


# ---------------------------------------------------------------- el grafo
@dataclass
class Fragmento:
    """Lo que el investigador recibe: una sección, con su cita."""
    id: str
    titulo: str
    texto: str
    fuente: str          # "enunciado" | "curso"
    archivo: str
    ubicacion: str        # "p.N" (PDF) o "L.N" (notas del curso)
    via: str = "similitud"  # "similitud" | "depende_de" | "entidad:<nombre>"

    def cita(self) -> str:
        return f"[{self.fuente}:{self.archivo} · {self.ubicacion} · {self.titulo}]"


class Indexador:
    """El GraphRAG: construye el grafo y responde `contexto_para` al investigador."""

    def __init__(self) -> None:
        self.grafo = nx.MultiDiGraph()
        self._alias: dict[tuple[str, str, str], str] = {}   # (fuente_id, tipo, numero) → nodo
        self._orden: dict[str, list[str]] = {}               # fuente_id → [nodo, ...] en orden
        self._emb_cache: dict[str, np.ndarray] = {}

    # ---- capa base: el enunciado ------------------------------------------------------
    def agregar_documento(self, documento: DocumentoLeido, fuente_id: str,
                          fuente: str = "enunciado") -> list[str]:
        """Secciones de un `DocumentoLeido` (el Lector) como nodos, con sus `depende_de`."""
        ids: list[str] = []
        for sec in documento.secciones:
            nid = f"{fuente_id}:{sec.id}"
            self.grafo.add_node(nid, titulo=sec.titulo, texto=sec.texto, fuente=fuente,
                               archivo=fuente_id, ubicacion=f"p.{sec.pagina_inicio}")
            if alias := _alias_de_id(sec.id):
                self._alias[(fuente_id, *alias)] = nid
            ids.append(nid)
        self._orden[fuente_id] = ids
        self._vincular_referencias(fuente_id, ids)
        return ids

    # ---- capa base: el material del curso ---------------------------------------------
    def agregar_notas_curso(self, carpeta: Path, fuente: str = "curso") -> list[str]:
        """Markdown de `notas-teoricas/`: un nodo por encabezado real (con `#`), un
        `fuente_id` por archivo para que sus propias referencias internas («la sección
        anterior») se resuelvan dentro de SU documento, no se mezclen con el enunciado."""
        ids: list[str] = []
        for ruta in sorted(Path(carpeta).glob("*.md")):
            fuente_id = ruta.stem
            texto = ruta.read_text(encoding="utf-8", errors="replace")
            doc_ids = []
            for n, (titulo, cuerpo, linea) in enumerate(_secciones_markdown(texto), start=1):
                nid = f"{fuente_id}:sec-{n}"
                self.grafo.add_node(nid, titulo=titulo, texto=cuerpo, fuente=fuente,
                                   archivo=fuente_id, ubicacion=f"L.{linea}")
                if alias := _alias_de_id(f"sec-{n}"):
                    self._alias[(fuente_id, *alias)] = nid
                doc_ids.append(nid)
            self._orden[fuente_id] = doc_ids
            self._vincular_referencias(fuente_id, doc_ids)
            ids += doc_ids
        return ids

    def _vincular_referencias(self, fuente_id: str, ids: list[str]) -> None:
        """La Regla 2: `depende_de` sale de un regex sobre el texto, nunca de pedírselo al
        LLM. Dos formas de referencia: numerada («Parte 1») y relativa («la parte anterior»,
        que apunta a la sección inmediatamente previa EN EL ORDEN DEL DOCUMENTO)."""
        for i, nid in enumerate(ids):
            texto = self.grafo.nodes[nid]["texto"]
            for tipo, numero in _REF_NUMERADA.findall(texto):
                destino = self._alias.get((fuente_id, _despojar(tipo), numero.lower()))
                if destino and destino != nid:
                    self.grafo.add_edge(nid, destino, tipo="depende_de", via="numerada")
            if _REF_RELATIVA.search(texto) and i > 0:
                self.grafo.add_edge(nid, ids[i - 1], tipo="depende_de", via="relativa")

    # ---- capa de entidades (con LLM) ---------------------------------------------------
    def extraer_entidades(self, h200, nodos: list[str] | None = None,
                          max_tokens: int = 2048) -> int:
        """Una llamada por sección: el LLM propone entidades (dataset/método/métrica/
        concepto); el código decide el nodo (nombre normalizado) y la arista `menciona`. Si
        dos secciones —una del enunciado, una del curso— mencionan la misma entidad
        normalizada, es el mismo nodo: ahí se cruzan. Requiere H200 (VPN); sin red, no se
        ejecuta esta capa y el grafo sigue sirviendo solo con `depende_de`."""
        import json

        sistema = ('Lista las entidades técnicas de este fragmento de un enunciado o unas '
                  'notas de clase: datasets, métodos/algoritmos, métricas y conceptos. '
                  'Responde SOLO JSON: {"entidades": [{"nombre": "...", "tipo": '
                  '"dataset|metodo|metrica|concepto"}]}. Como mucho 8 entidades, las más '
                  'centrales. Si no hay ninguna clara, "entidades": [].')
        agregadas = 0
        objetivo = nodos or list(self.grafo.nodes)
        for nid in objetivo:
            if self.grafo.nodes[nid].get("fuente") not in {"enunciado", "curso"}:
                continue
            texto = self.grafo.nodes[nid]["texto"][:3000]
            r = h200.chat([{"role": "system", "content": sistema},
                          {"role": "user", "content": texto}],
                         json_mode=True, max_tokens=max_tokens)
            if r["fin"] == "length" and not r["contenido"].strip():
                continue       # se quedó razonando; esta sección se salta, no se inventa nada
            try:
                entidades = json.loads(r["contenido"]).get("entidades", [])
            except (json.JSONDecodeError, AttributeError):
                continue
            for e in entidades:
                nombre = str(e.get("nombre", "")).strip()
                if not nombre:
                    continue
                eid = f"entidad:{_despojar(nombre)}"
                if eid not in self.grafo:
                    self.grafo.add_node(eid, titulo=nombre, tipo_entidad=e.get("tipo", "concepto"),
                                       fuente="entidad")
                    agregadas += 1
                self.grafo.add_edge(nid, eid, tipo="menciona")
        return agregadas

    # ---- recuperación: lo que usa el Investigador --------------------------------------
    def _secciones(self) -> list[str]:
        return [n for n, d in self.grafo.nodes(data=True) if d.get("fuente") in {"enunciado", "curso"}]

    def _similares(self, pregunta: str, k: int, h200=None) -> list[tuple[str, float]]:
        nodos = self._secciones()
        if not nodos:
            return []
        textos = [self.grafo.nodes[n]["texto"] for n in nodos]
        if h200 is not None:
            vectores = np.array(self._embeber(h200, nodos, textos) + h200.embed([pregunta]))
            q = vectores[-1]
            puntajes = vectores[:-1] @ q
        else:
            # Sin H200 (sin VPN): TF-IDF + coseno, igual que la Parte 0.b. Degradado, no roto.
            tfidf = TfidfVectorizer().fit(textos + [pregunta])
            puntajes = cosine_similarity(tfidf.transform([pregunta]), tfidf.transform(textos))[0]
        orden = np.argsort(-puntajes)[:k]
        return [(nodos[i], float(puntajes[i])) for i in orden]

    def _embeber(self, h200, nodos: list[str], textos: list[str]) -> list[list[float]]:
        faltan = [n for n in nodos if n not in self._emb_cache]
        if faltan:
            vectores = h200.embed([self.grafo.nodes[n]["texto"][:2000] for n in faltan])
            self._emb_cache.update(zip(faltan, vectores))
        return [self._emb_cache[n] for n in nodos]

    def contexto_para(self, pregunta: str, k: int = 2, saltos: int = 1,
                      h200=None) -> list[Fragmento]:
        """Lo que entrega el investigador a una subtarea: los `k` fragmentos más parecidos a
        la pregunta MÁS las secciones de las que dependen (`depende_de`, un salto) y las
        secciones que comparten una entidad con ellas (si se corrió `extraer_entidades`).
        Cada fragmento trae su cita: de dónde salió."""
        vistos: dict[str, Fragmento] = {}

        def _agregar(nid: str, via: str) -> None:
            if nid in vistos:
                return
            d = self.grafo.nodes[nid]
            vistos[nid] = Fragmento(id=nid, titulo=d["titulo"], texto=d["texto"],
                                   fuente=d["fuente"], archivo=d["archivo"],
                                   ubicacion=d["ubicacion"], via=via)

        for nid, _ in self._similares(pregunta, k, h200):
            _agregar(nid, "similitud")

        frontera = list(vistos)
        for _ in range(saltos):
            siguiente = []
            for nid in frontera:
                for _, destino, datos in self.grafo.out_edges(nid, data=True):
                    if datos.get("tipo") == "depende_de" and destino not in vistos:
                        _agregar(destino, "depende_de")
                        siguiente.append(destino)
                    elif datos.get("tipo") == "menciona":          # nid → entidad
                        for vecino, _, d2 in self.grafo.in_edges(destino, data=True):
                            if d2.get("tipo") == "menciona" and vecino not in vistos:
                                _agregar(vecino, f"entidad:{self.grafo.nodes[destino]['titulo']}")
                                siguiente.append(vecino)
            frontera = siguiente
        return list(vistos.values())

    # ---- para el entregable: «el grafo de la Tarea A dibujado» -------------------------
    def exportar_dot(self, fuente_id: str | None = None) -> str:
        """Graphviz DOT: nodos de sección (rectángulo) y de entidad (óvalo), aristas
        `depende_de` sólidas y `menciona` punteadas. `dot -Tpng grafo.dot -o grafo.png`."""
        lineas = ["digraph GraphRAG {", '  rankdir=LR; node [fontname="Helvetica"];']
        for n, d in self.grafo.nodes(data=True):
            if fuente_id and d.get("archivo") not in {fuente_id, None} and d.get("fuente") != "entidad":
                continue
            etiqueta = d["titulo"].replace('"', "'")[:40]
            forma = "ellipse" if d.get("fuente") == "entidad" else "box"
            lineas.append(f'  "{n}" [label="{etiqueta}", shape={forma}];')
        for o, dst, datos in self.grafo.edges(data=True):
            if fuente_id and not (o.startswith(f"{fuente_id}:") or dst.startswith(f"{fuente_id}:")):
                continue
            estilo = "solid" if datos.get("tipo") == "depende_de" else "dashed"
            lineas.append(f'  "{o}" -> "{dst}" [style={estilo}, label="{datos.get("tipo", "")}"];')
        lineas.append("}")
        return "\n".join(lineas)

    def resumen(self) -> str:
        n_sec = len(self._secciones())
        n_ent = sum(1 for _, d in self.grafo.nodes(data=True) if d.get("fuente") == "entidad")
        n_dep = sum(1 for *_, d in self.grafo.edges(data=True) if d.get("tipo") == "depende_de")
        n_men = sum(1 for *_, d in self.grafo.edges(data=True) if d.get("tipo") == "menciona")
        return (f"{n_sec} sección(es) de nodo, {n_ent} entidad(es), "
                f"{n_dep} arista(s) depende_de, {n_men} arista(s) menciona")


if __name__ == "__main__":
    import sys

    from solver.lector import Lector

    raiz = Path(__file__).resolve().parents[1]
    idx = Indexador()
    for nombre in (sys.argv[1:] or ["tarea-a-generativo-discriminativo"]):
        doc = Lector().leer(raiz / "enunciados" / f"{nombre}.pdf")
        idx.agregar_documento(doc, fuente_id=nombre.split("-")[0] + "-" + nombre.split("-")[1])
    idx.agregar_notas_curso(raiz.parent / "notas-teoricas")
    print(idx.resumen())
    print()
    print("Aristas depende_de encontradas:")
    for o, d, datos in idx.grafo.edges(data=True):
        if datos.get("tipo") == "depende_de":
            print(f"  {o} --{datos['via']}--> {d}")
