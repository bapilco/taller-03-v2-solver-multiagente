# Lab 03 · solver-v2 — el kit del Taller 03 v2

> **2026-10-02.** Material del Taller 03 v2 (opcional): un solver multiagente con GraphRAG que
> resuelve una tarea a partir de su enunciado en PDF. Ver
> `curso/talleres/taller-03-v2-solver-multiagente.md`. Esta carpeta **no toca** `agent.py` ni
> `tools.py` del laboratorio.

```bash
pip install -r requirements.txt
python generar_enunciados.py            # enunciados/*.md → enunciados/*.pdf (ya vienen hechos)
python parte0/b_rag_plano.py            # Parte 0.b, sin red
python parte0/c_exit_cero.py            # Parte 0.c, sin red
python parte0/a_llm_sin_ejecutar.py     # Parte 0.a, con la H200 (VPN)
python verdades.py                      # las métricas de referencia de la Tarea C
python evaluar_solver.py --solver mi_solver:Solver --ruta ../mi-solver   # Parte 2.b
```

## Qué hay

```
solver-v2/
├── enunciados/              # tareas de práctica: .md (fuente) y .pdf (lo que lee el solver), y un paquete
│   ├── tarea-a-generativo-discriminativo   NB contra regresión logística → reporte.md
│   ├── tarea-b-temperatura-top-p           softmax, top-p, muestreo → notebook ejecutado
│   ├── tarea-c-recuperacion-lexica-lsa     TF-IDF contra LSA → reporte.pdf de 2 páginas
│   └── hackathon3_student/                 un PAQUETE: notebook a completar en su sitio, contrato con
│                                           hashes, módulo de apoyo y verificador propio (Tarea D)
├── generar_enunciados.py    # .md → .pdf con PyMuPDF, sin navegador
├── h200.py                  # cliente de la H200: LLM (12555) y embeddings bge-m3 (11434), stdlib
├── parte0/                  # las tres fallas que no fallan
│   ├── a_llm_sin_ejecutar.py   el LLM escribe el reporte sin ejecutar nada; se compara con lo medido
│   ├── b_rag_plano.py          un RAG por similitud pierde la referencia «Parte 1»; una arista la recupera
│   └── c_exit_cero.py          un experimento con fuga termina con código 0 y el evaluador ingenuo lo aprueba
├── golden_tareas.json       # 4 tareas (A–C en PDF, D un paquete), 33 comprobaciones; la verdad de cada cifra es código (verdad_py)
├── verdades.py              # el corpus de la Tarea C y sus métricas de referencia
├── evaluar_solver.py        # corre un solver sobre el golden set → resultados_solver.csv, resumen_solver.csv
└── requirements.txt
```

## Las comprobaciones del golden set

| Tipo | Aprueba si |
|---|---|
| `archivo` | existe el patrón (con `minimo`); con `o_imagen_en_ipynb`, vale una imagen embebida |
| `secciones` | están todos los encabezados, y en orden si `en_orden` |
| `palabras_max` / `paginas_max` | el entregable no pasa del límite |
| `cifra` | el valor de `verdad_py`, dentro de `tolerancia`, aparece en los 160 caracteres que siguen a `etiqueta` |
| `cifra_presente` | ese valor aparece en algún lugar del entregable |
| `contiene` | todas las cadenas de `lista` están |
| `ipynb_ejecutado` | toda celda de código tiene `execution_count` y ninguna salida es un error |
| `celdas_intactas` | ids, orden y hashes de las celdas protegidas, contra el `contrato` |
| `celdas_completas` | ninguna celda editable conserva `...` o el marcador, salvo las de `excepto` |
| `verificador_externo` | el verificador del paquete, corrido en una copia con el entregable (con los `reemplazos` declarados, solo en celdas editables), imprime `espera` |
| `procedencia` | la fracción de cifras con dos o más decimales del entregable que aparece también en algo que escribió una ejecución (JSON, CSV, logs, salidas de notebook) es al menos `minimo`; las cifras del enunciado no cuentan |

`verdad_py` es código que deja el valor en `verdad`; corre en el evaluador, no en el solver. Una
tarea entra como archivo (`pdf`) o como carpeta (`entrada`); en la D, la verdad de la pérdida
final la calcula el propio motor del paquete con las funciones de referencia.
**La verdad no se escribe a mano: se calcula.** Si el enunciado cambia, la verdad cambia con él.

## Lo que midió la Parte 0 el 2026-10-02

- **0.a** (H200, `zai-org/GLM-5.3-Flash`): 30 cifras en el reporte, ninguna de una ejecución.
  NB 0,9415 contra 0,9474 medido; regresión logística al 5 % del entrenamiento 0,7251 contra
  0,9298 medido. Sin la instrucción «no tienes intérprete», el modelo razonó 40 000 tokens y
  devolvió un `content` vacío.
- **0.b**: RAG plano, 0 de 3 hechos; con un salto por `depende_de`, 3 de 3.
- **0.c**: evaluador ingenuo APROBADO; con dos comprobaciones de código, RECHAZADO.
