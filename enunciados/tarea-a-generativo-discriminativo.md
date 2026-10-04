# Tarea A — Generativo contra discriminativo con pocos datos

**MMIA 6013 · Tarea de práctica para el solver del Taller 03 v2**
**Entrega:** un reporte en Markdown (`reporte.md`), con el código que lo produce.

Ng y Jordan (2002) mostraron que un clasificador generativo puede alcanzar su error asintótico
con menos ejemplos que su par discriminativo, aunque ese error asintótico sea mayor. Esta tarea
lo comprueba sobre datos reales.

## Parte 1 — Datos

Carga el conjunto *Breast Cancer Wisconsin (Diagnostic)* que trae scikit-learn
(`sklearn.datasets.load_breast_cancer`). Reporta cuántos ejemplos, cuántos atributos y cuántos
casos de cada clase tiene. Divide los datos en entrenamiento y prueba, 70 % y 30 %,
estratificando por la clase, con `random_state=42`. Esa división es la única que se usa en toda
la tarea: el conjunto de prueba no se toca para ajustar nada.

## Parte 2 — Dos clasificadores

Entrena un **Naive Bayes gaussiano** y una **regresión logística** (con los atributos
estandarizados; el escalador se ajusta solo con el entrenamiento). Reporta la exactitud
(*accuracy*) y el F1 macro de cada uno sobre el conjunto de prueba, en una tabla.

## Parte 3 — La curva de aprendizaje

Sobre la misma división de la Parte 1, entrena los dos modelos con el 5 %, 10 %, 25 %, 50 % y
100 % del entrenamiento (submuestras estratificadas, `random_state=42`) y evalúa cada uno
siempre sobre el mismo conjunto de prueba. Dibuja la exactitud contra el número de ejemplos de
entrenamiento, con una curva por modelo, y guarda la figura como PNG.

## Parte 4 — Discusión

Explica, con tus cifras, si se observa el cruce que predicen Ng y Jordan: en qué tamaño de
entrenamiento conviene cada modelo y por qué. Relaciónalo con el sesgo y la varianza de un
modelo generativo frente a uno discriminativo.

## El reporte

Un `reporte.md` con estas secciones, en este orden: **Introducción**, **Metodología**,
**Resultados** (con la tabla de la Parte 2 y la figura de la Parte 3), **Discusión** y
**Conclusiones**. Máximo **1 200 palabras**. Toda cifra del reporte tiene que salir de la
ejecución del código entregado.
