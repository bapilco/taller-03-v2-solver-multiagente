# Tarea C — Recuperación léxica contra semántica latente sobre un corpus mínimo

**MMIA 6013 · Tarea de práctica para el solver del Taller 03 v2**
**Entrega:** un reporte en **PDF** (`reporte.pdf`) de dos páginas como máximo, con el código.

El corpus son diez documentos cortos y las consultas son seis, cada una con **un** documento
relevante. Los datos están completos en este enunciado: no hace falta descargar nada.

**Corpus**

| id | texto |
|---|---|
| d01 | El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión. |
| d02 | Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización. |
| d03 | La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo. |
| d04 | BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento. |
| d05 | Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación. |
| d06 | Un agente con herramientas decide en cada paso qué función llamar y observa el resultado. |
| d07 | ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos. |
| d08 | El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales. |
| d09 | La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad. |
| d10 | La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria. |

**Consultas y juicios de relevancia**

| id | consulta | relevante |
|---|---|---|
| q1 | ¿Qué función de ranking léxica pondera la frecuencia de términos? | d04 |
| q2 | ¿Cómo se añaden fragmentos recuperados al prompt? | d03 |
| q3 | ¿Qué técnica entrena matrices de bajo rango? | d08 |
| q4 | ¿Cómo se compara la orientación de dos vectores de texto? | d05 |
| q5 | ¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra? | d09 |
| q6 | ¿Qué arquitectura combina autoatención con capas feed-forward? | d02 |

## Parte 1 — Línea base léxica

Implementa la recuperación con TF-IDF (`TfidfVectorizer` de scikit-learn con sus parámetros
por defecto) y similitud coseno. Para cada consulta, ordena los diez documentos. Reporta
**Hit@1**, **Hit@3** y **MRR** sobre las seis consultas.

## Parte 2 — Semántica latente

Con las mismas consultas y los mismos juicios, proyecta la matriz TF-IDF a 4 dimensiones con
`TruncatedSVD(n_components=4, random_state=0)` (LSA) y repite la evaluación. Reporta las tres
métricas de los dos métodos en una sola tabla.

## Parte 3 — Análisis por consulta

Identifica en qué consultas falla cada método (el relevante fuera del top 3) y explica por qué,
mirando qué términos comparten la consulta y su documento relevante.

## Parte 4 — Pregunta conceptual

¿Por qué un recuperador denso (embeddings de un modelo entrenado) resolvería la consulta q5
mejor que los dos métodos de esta tarea? ¿Qué costo tiene frente a TF-IDF?

## El reporte

En PDF, dos páginas como máximo, con las secciones **Objetivo**, **Método**, **Resultados**
(la tabla de la Parte 2 y la de la Parte 3) y **Discusión**. Toda cifra sale de la ejecución.
