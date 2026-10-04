#!/usr/bin/env python3
"""generar_enunciados.py — los enunciados de práctica, de Markdown a PDF.

    python generar_enunciados.py            # enunciados/*.md → enunciados/*.pdf

El solver del Taller 03 v2 recibe **un PDF**, que es como llega una tarea de verdad. Los
tres enunciados de práctica se escriben en Markdown —para poder revisarlos en un diff— y
este script los imprime con PyMuPDF (`fitz.Story`), sin navegador ni LaTeX. Las fórmulas
`$…$` se imprimen como texto plano: el solver tiene que leerlas así, igual que leería
el PDF de un profesor que no las compuso.
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf as fitz
from markdown_it import MarkdownIt

AQUI = Path(__file__).resolve().parent
CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.35; }
h1 { font-size: 17pt; margin-bottom: 6pt; }
h2 { font-size: 13pt; margin-top: 12pt; margin-bottom: 4pt; }
p { margin-top: 3pt; margin-bottom: 5pt; text-align: justify; }
table { border-collapse: collapse; margin: 6pt 0; }
th, td { border: 0.6pt solid #888; padding: 2pt 4pt; font-size: 9.5pt; }
th { background-color: #eeeeee; }
code { font-family: monospace; font-size: 9.5pt; }
"""


def md_a_html(md: str) -> str:
    md = re.sub(r"\$([^$\n]+)\$", r"\1", md)          # las fórmulas, en texto plano
    md = md.replace("\\to", "→").replace("\\infty", "∞").replace("\\sum_j", "Σ_j")
    md = md.replace("\\exp", "exp")
    return MarkdownIt("commonmark").enable("table").render(md)


def imprimir(fuente: Path) -> Path:
    destino = fuente.with_suffix(".pdf")
    historia = fitz.Story(html=md_a_html(fuente.read_text(encoding="utf-8")),
                          user_css=CSS)
    escritor = fitz.DocumentWriter(str(destino))
    a4 = fitz.paper_rect("a4")
    caja = a4 + (56, 56, -56, -56)
    mas = True
    while mas:
        dispositivo = escritor.begin_page(a4)
        mas, _ = historia.place(caja)
        historia.draw(dispositivo)
        escritor.end_page()
    escritor.close()
    return destino


if __name__ == "__main__":
    for md in sorted((AQUI / "enunciados").glob("*.md")):
        pdf = imprimir(md)
        print(f"{md.name} → {pdf.name} ({fitz.open(pdf).page_count} p.)")
