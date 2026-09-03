# Previsión de Ventas Retail

**¿Cuánto le cuesta a un negocio no ver venir una huelga, una guerra o una promoción?**

Una aplicación interactiva que cuenta, paso a paso, cómo construí un modelo de
previsión de ventas capaz de tener en cuenta lo que pasa *fuera* del negocio (no solo
el histórico de ventas) y cuánto mejora eso la predicción.

No hace falta saber nada de Machine Learning para seguirla: empieza por el problema,
sigue por los datos, y termina dejándote probar el modelo tú mismo con tus propios
escenarios ("¿y si hay una huelga la semana que viene?").

## Ver la app

🔗 **[Abrir la app](https://forecast-ventas-retail-app-w5msdvcsbuxas94qjcdore.streamlit.app)**

## De qué trata, en dos frases

Una cadena retail necesita saber cuánto va a vender cada semana para planificar
inventario y personal. Un modelo que solo mira el pasado se equivoca más de lo
necesario porque no sabe que hubo una huelga, una promoción o un conflicto que movió
las cifras. Así que construí **dos modelos** (uno que solo mira el histórico y otro
que además tiene en cuenta esos eventos) y comparé cuál acertaba más.

**El resultado:** tener en cuenta el contexto reduce el error de predicción un 32%.

## Qué te vas a encontrar al recorrerla

1. **El problema** — por qué prever ventas a ciegas sale caro
2. **Los datos** — 156 semanas de ventas y 5 señales externas (huelgas, promociones, logística...)
3. **Antes de modelar** — qué se ve a simple vista antes de tocar ningún algoritmo
4. **El camino hasta el modelo** — cómo se llegó, paso a paso, sin tecnicismos
5. **SARIMA vs SARIMAX** — el modelo que solo mira el pasado, contra el que tiene en cuenta el contexto
6. **Por qué predice eso** — qué peso tiene cada evento en la predicción
7. **Playground** — mueve tú los controles y mira cómo cambia la predicción en directo
8. **Resultados y decisiones** — qué se descubrió y qué haría un negocio con ello

## Cómo está hecho

Python + [Streamlit](https://streamlit.io) para la aplicación, y
[statsmodels](https://www.statsmodels.org/) (SARIMAX) para el modelo de series
temporales. El análisis completo, en formato notebook, está en el
[repositorio de portfolio](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/04-series-temporales/arima/03-forecast-ventas-retail).

Calculé todos los números que aparecen en la app (el error del modelo, el peso de cada
evento, los resultados del backtesting...) una vez en `model/train.py` y los guardé
como datos: nada está escrito a mano, así que si cambian los datos de entrada, basta
con volver a ejecutar ese script para que toda la narrativa se actualice sola.

## Ejecutarla en tu ordenador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá sola en el navegador. Los resultados del modelo ya vienen calculados en
`model/artifacts/`, así que no hace falta reentrenar nada para verla funcionar.

Solo si cambias el dataset (`data/SALES_FORECASTING_VARIABLES_EXOGENAS.xlsx`) necesitas
regenerar esos resultados:

```bash
python model/train.py    # tarda unos 10-15 minutos (entrena 15 modelos en total)
```

<details>
<summary>Estructura del proyecto, para quien quiera curiosear el código</summary>

```
app.py                    la aplicación — toda la narrativa, sección a sección
components/
  ui.py                    bloques visuales reutilizables (tarjetas, títulos, callouts)
  charts.py                 gráficos, con la paleta de colores del proyecto
utils/
  data_loader.py             carga de datos y resultados (con cache de Streamlit)
  forecasting.py             genera predicciones en vivo para el playground
model/
  train.py                    entrena los modelos y calcula todos los resultados
  artifacts/                   resultados ya calculados (métricas, coeficientes...)
data/                      el dataset original
assets/style.css           el sistema visual de la app
```

Reconstruyo el modelo final al vuelo a partir de sus parámetros ya ajustados
(`model/artifacts/final_params.json`) en vez de cargar el modelo completo desde disco:
es instantáneo y evita guardar un archivo de cientos de MB.
</details>

---

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borja-mora-mendez/) · [GitHub](https://github.com/BORJAMOME)
