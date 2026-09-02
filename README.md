# Previsión de Ventas Retail — SARIMA vs SARIMAX (Streamlit)

Case study interactivo que convierte el notebook [`forecasting_ventas_retail.ipynb`](../Data-Analytics-Portfolio/03-Machine-Learning/04-series-temporales/arima/03-forecast-ventas-retail/forecasting_ventas_retail.ipynb)
en una experiencia narrativa: del problema de negocio a la decisión, pasando por los datos, el
modelo, su explicabilidad y un playground para simular escenarios en vivo.

## Qué resuelve

Una cadena retail necesita prever sus ventas semanales. Un modelo puramente histórico (SARIMA)
no anticipa el impacto de huelgas, conflictos, problemas logísticos o promociones. Este proyecto
compara ese baseline contra un SARIMAX con 5 variables exógenas y demuestra que el contexto
externo reduce el error de predicción de forma sustancial — y permite simular "¿qué pasa si...?"

## Cómo ejecutarlo

```bash
pip install -r requirements.txt

# 1. Entrenar el modelo y generar artefactos (una sola vez, o si cambian los datos)
python model/train.py

# 2. Lanzar la app
streamlit run app.py
```

## Estructura

```
app.py                 punto de entrada — toda la narrativa, sección a sección
components/
  ui.py                 bloques HTML reutilizables (kpis, cards, callouts, nav)
  charts.py              figuras Plotly con la paleta del proyecto
utils/
  data_loader.py          carga de datos y artefactos (con cache de Streamlit)
  forecasting.py          predicción de escenarios en vivo sobre el modelo ya entrenado
model/
  train.py                 entrena SARIMA + SARIMAX, backtesting, diagnóstico de residuos
  artifacts/                metricas, coeficientes, backtesting y modelo final (.pickle)
data/                     dataset original (xlsx)
assets/style.css           sistema visual editorial
```

## Notas

- El modelo final (`model/artifacts/final_model.pickle`) está entrenado con **todo** el histórico;
  el playground reutiliza ese modelo ya ajustado y solo cambia los valores exógenos futuros —
  no reentrena en cada interacción, por eso la simulación es instantánea.
- Todos los números que se muestran en la app (MAE, MAPE, RMSE, coeficientes, backtesting) se
  calculan en `model/train.py`, no están escritos a mano — cambiar el dataset y re-ejecutar el
  entrenamiento actualiza toda la narrativa.

---

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borja-mora-mendez/) · [GitHub](https://github.com/BORJAMOME)
