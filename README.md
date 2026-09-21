# Previsión de Ventas Retail

**¿Cuánto mejora una previsión de ventas cuando sabe qué estaba pasando *fuera* de la tienda?**

Una aplicación interactiva, con formato de reportaje de datos, que compara dos modelos de previsión
semanal de ventas: uno que solo mira el histórico (SARIMA) y otro que, además, conoce si esa semana hubo
promoción, huelga, conflicto o problemas logísticos (SARIMAX). La pregunta no es solo cuál acierta más,
sino **cuánto de esa mejora es fiable**.

No hace falta saber nada de Machine Learning para seguirla: empieza por el problema, sigue por los datos,
explica el método sin tecnicismos (el detalle técnico va aparte, en notas al margen) y termina dejándote
probar el modelo con tus propios escenarios.

## Ver la app

🔗 **[Abrir la app](https://forecast-ventas-retail-app-w5msdvcsbuxas94qjcdore.streamlit.app)**

## De qué trata, en dos frases

Una cadena retail necesita saber cuánto va a vender cada semana para decidir cuánto stock pedir y cuánto
personal programar. Un modelo que solo mira el pasado no puede saber que esa semana hubo una huelga, así que
comparé dos modelos sobre 156 semanas de ventas y medí no solo cuánto mejora el que conoce el contexto, sino
**de dónde sale esa mejora y cuánto se puede confiar en ella**.

**El resultado, con matices:**

- En el examen de 12 semanas, el error baja del 7,8 % al 5,3 % (un 32 % menos).
- Esa mejora se concentra en **2 semanas con incidencia**; en las otras 10, los dos modelos rinden casi igual.
- Repetido en 13 ventanas del pasado, el error medio del SARIMAX es del **7,3 %** (mediana 5,7 %, máximo
  14,5 %): el 5,3 % del examen fue de los favorables.
- Solo 3 de las 5 señales (promoción, guerra y problema logístico) tienen un efecto que el modelo distingue
  de cero; la guerra, además, procede de un único episodio.

## Qué te vas a encontrar al recorrerla

Trece capítulos, cada uno responde a una pregunta:

1. **El problema** — por qué prever ventas a ciegas sale caro
2. **Los datos** — 156 semanas, 5 señales y cuántos *episodios* hay de cada una
3. **Antes de modelar** — qué se ve a simple vista
4. **El camino hasta el modelo** — un método pensado para no engañarse
5. **Los dos modelos** — el que mira el pasado contra el que mira el contexto
6. **¿De dónde viene la mejora?** — semana a semana, con y sin incidencias
7. **¿Cuánto podemos fiarnos?** — backtesting en 13 ventanas, dónde falla y diagnóstico de residuos
8. **¿Cuánto pesa cada señal?** — qué efectos merecen confianza y cuáles no
9. **Ponlo a prueba** — Playground con avisos cuando el escenario se sale de lo que el histórico ha visto
10. **¿Qué hemos descubierto?** — el resultado, con sus matices
11. **¿Qué podría hacer el negocio?** — hipótesis, no conclusiones
12. **Lo que sabemos y lo que no** — limitaciones y siguientes pasos
13. **Del dato a la decisión** — conclusión

## Cómo está hecho

Python + [Streamlit](https://streamlit.io) para la aplicación y [statsmodels](https://www.statsmodels.org/)
(SARIMAX) para el modelo de series temporales. El análisis completo, en formato notebook, está en el
[repositorio de portfolio](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/04-series-temporales/arima/03-forecast-ventas-retail).

Ninguna cifra del texto está escrita a mano. Los modelos, el backtesting y el diagnóstico de residuos se
calculan una vez en `model/train.py` y se guardan como datos; el resto de cifras (episodios de cada señal,
desglose del examen por tipo de semana, referencia sin modelo, contexto de cada ventana del backtesting…) se
derivan de esos artefactos y de los datos en `utils/audit.py`, sin reentrenar nada. Si cambian los datos,
cambia el texto.

Algunas decisiones de rigor que están en la app y conviene conocer:

- **Cuánto de la mejora es real.** El −32 % se descompone por tipo de semana y se compara con una referencia
  sin modelo (repetir la última semana). Con 12 semanas de examen, la diferencia no es estadísticamente
  concluyente (Wilcoxon, p ≈ 0,20).
- **El examen conoce los eventos.** Tanto el examen como el backtesting usan los eventos reales de cada
  semana, no eventos previstos: el −32 % es lo que se gana *si se sabe qué va a pasar*.
- **Episodios, no semanas.** Un evento que ocurre en un solo episodio (la guerra) no permite separar su efecto
  de todo lo demás que pasó a la vez, aunque el p-valor sea pequeño.
- **Diagnóstico de residuos sin el arranque del filtro.** Con d=1 y D=1 (s=52), las primeras 53 semanas son
  inicialización, no error del modelo, y contaminan cualquier test. Se excluyen (y una semana más que arrastra
  su eco), y siempre se informa de cuántas.
- **Sin causalidad.** Son asociaciones dentro del modelo; la promoción, además, es una decisión del negocio.

## Ejecutarla en tu ordenador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá sola en el navegador. Los resultados del modelo ya vienen calculados en `model/artifacts/`, así que
no hace falta reentrenar nada para verla funcionar.

Solo si cambias el dataset (`data/SALES_FORECASTING_VARIABLES_EXOGENAS.xlsx`) necesitas regenerar esos
resultados:

```bash
python model/train.py    # tarda unos 10-15 minutos (entrena 15 modelos en total)
```

<details>
<summary>Estructura del proyecto, para quien quiera curiosear el código</summary>

```
app.py                    la aplicación — toda la narrativa, capítulo a capítulo
components/
  editorial.py              sistema editorial: un componente por función narrativa (lede, beat, figure…)
  charts.py                 gráficos, con la paleta de colores del proyecto
utils/
  data_loader.py             carga de datos y resultados (con cache de Streamlit)
  audit.py                    cifras derivadas de los artefactos y los datos (episodios, desglose del examen…)
  forecasting.py              genera predicciones en vivo para el Playground
model/
  train.py                    entrena los modelos y calcula todos los resultados
  artifacts/                   resultados ya calculados (métricas, coeficientes, backtesting, residuos…)
data/                      el dataset original
assets/editorial.css       sistema de composición editorial (rejilla, niveles de ancho, ritmo, tokens) — reutilizable
assets/style.css           identidad de este proyecto (paleta y familias tipográficas)
```

Reconstruyo el modelo final al vuelo a partir de sus parámetros ya ajustados
(`model/artifacts/final_params.json`) en vez de cargar el modelo completo desde disco: es instantáneo y evita
guardar un archivo de cientos de MB.
</details>

---

### Sistema editorial (reutilizable en otros proyectos de ML)

`assets/editorial.css` convierte el contenedor de Streamlit en una rejilla con cinco niveles de ancho
(`--canvas-width` / `--wide-width` / `--story-width` / `--reading-width` / `--aside-width`), una escala de
espacio (`--space-xs` … `--space-3xl`) y una escala tipográfica editorial. `components/editorial.py`
expone un componente por función narrativa. Para reutilizarlo: copiar ambos archivos y definir la paleta
en el `style.css` del proyecto. Requiere `streamlit==1.58.0` (usa `st.container(key=...)` y el DOM de esa
versión).

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)
