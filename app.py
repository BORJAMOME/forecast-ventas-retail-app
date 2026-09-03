"""
Previsión de Ventas Retail — SARIMA vs SARIMAX
Case study interactivo en Streamlit: de la pregunta de negocio a la
decisión, pasando por los datos, el modelo y su explicabilidad.

Autor: Borja Mora Méndez
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from components import charts, ui
from utils.data_loader import (EVENT_META, EXOG_COLS, artifacts_ready, load_csv,
                                load_data, load_final_model, load_json)
from utils.forecasting import predict_scenario, scenario_from_inputs

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Previsión de Ventas Retail · SARIMA vs SARIMAX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with open(ROOT / "assets" / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if not artifacts_ready():
    st.error(
        "Los artefactos del modelo todavía no se han generado. "
        "Ejecuta `py -3.10 model/train.py` desde la raíz del proyecto y recarga esta página."
    )
    st.stop()

df = load_data()
metrics = load_json("metrics.json")
coefficients = load_json("coefficients.json")
model_config = load_json("model_config.json")
adf_results = load_json("adf_results.json")
residual_diag = load_json("residual_diagnostics.json")
forecast_df = load_csv("test_forecast.csv")
backtest_df = load_csv("backtest.csv")
final_model = load_final_model()

last_date = df["date"].max()
n_weeks = len(df)
n_years = round((df["date"].max() - df["date"].min()).days / 365, 1)
cv_pct = df["sales"].std() / df["sales"].mean() * 100

# `promotion` es continua (0 a PROMO_MAX ~0.18) mientras que el resto de variables
# son binarias (0/1). El coeficiente crudo de coefficients.json representa el impacto
# por 1.0 unidad de intensidad — un descuento del 100%, que nunca ocurre en el histórico.
# Para no mostrar un porcentaje irreal, aquí se reexpresa "por 1 punto porcentual de
# descuento" (0.01) y se calcula además el impacto concreto al máximo histórico.
PROMO_MAX = float(df["promotion"].max())


def pct_impact_for(beta: float, delta: float) -> float:
    """Impacto porcentual real (no lineal) de mover una variable en `delta` unidades."""
    return (np.exp(beta * delta) - 1) * 100


def eur(value: float, sign: bool = False) -> str:
    """Formatea en euros con separador de miles europeo (punto), sin tocar el
    resto de la frase — evita el bug de usar .replace(",", ".") sobre texto
    que también contiene comas gramaticales."""
    spec = f"{value:+,.0f}" if sign else f"{value:,.0f}"
    return spec.replace(",", ".") + " €"


coef_map = {c["variable"]: c for c in coefficients}
coefficients_display = []
for c in coefficients:
    d = dict(c)
    if c["variable"] == "promotion":
        # Para comparar visualmente con las binarias (impacto "mientras dura el evento"),
        # se muestra el impacto de la promoción a su intensidad máxima histórica, no por
        # unidad de 1.0 (=100% de descuento, que nunca ocurre en los datos).
        d["pct_impact"] = pct_impact_for(c["beta"], PROMO_MAX)
    coefficients_display.append(d)

ui.nav()
ui.install_smooth_scroll()

# ============================================================ HERO ==
# Todo el hero va en UNA sola llamada a st.markdown: cada llamada por
# separado crea su propio contenedor en el DOM, así que abrir <div class
# ="hero-wrap"> en una y cerrarlo en otra produce un div vacío (0 hijos)
# en vez de envolver el contenido — eso fue lo que generaba la línea
# suelta encima del eyebrow (el border-bottom de ese div fantasma).
st.markdown(
    f"""
    <div id="top" class="hero-wrap">
      <p class="hero-kicker">Machine Learning Case Study · Series Temporales</p>
      <h1 class="hero-title">¿Cuánto le cuesta a un negocio retail <em>no ver venir</em> una huelga, una guerra o una promoción?</h1>
      <p class="hero-sub">Un modelo de series temporales aplicado a la previsión semanal de ventas de una cadena retail, comparando qué ocurre cuando el modelo solo mira el histórico frente a cuando también sabe qué está pasando en el mundo.</p>
      <div class="hero-meta">
        <span class="hero-pill">Borja Mora Méndez</span>
        <span class="hero-pill">Python · statsmodels (SARIMAX)</span>
        <span class="hero-pill">Streamlit</span>
        <span class="hero-pill">{n_weeks} semanas de histórico</span>
      </div>
      <div class="hero-scroll-row">
        <a href="#contexto" class="hero-scroll">explorar el caso &#8595;</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================ CONTEXTO ==
ui.section_open("contexto")
ui.eyebrow("Contexto")
ui.h2("El problema")
ui.lead(
    "Prever ventas es crítico para planificar inventario, personal y promociones. "
    "Pero un modelo que solo mira el histórico de ventas no puede anticipar lo que "
    "pasa fuera del negocio: una huelga de transporte, un conflicto geopolítico, un "
    "problema logístico. El equipo necesita saber cuánto pesa cada uno de esos "
    "eventos para poder planificar con ellos, no a pesar de ellos."
)
ui.kpi_grid([
    {"num": f"{n_weeks}", "label": "semanas de histórico"},
    {"num": f"{n_years}", "label": "años de operación (2023–2025)"},
    {"num": f"±{cv_pct:.0f}%", "label": "variación semana a semana"},
    {"num": "5", "label": "tipos de eventos exógenos registrados"},
])
st.write("")
ui.question_block(
    "La pregunta de negocio",
    '¿Podemos anticipar el impacto de una huelga, una guerra o una promoción '
    '<span class="accent">antes de que ocurra</span>, y usarlo para planificar?',
    "No se trata solo de predecir mejor: se trata de convertir el modelo en una "
    "herramienta de simulación — \"¿qué pasa si...?\" — para decisiones de inventario, "
    "personal y campañas.",
)
ui.section_close()

# ============================================================ DATOS ==
ui.section_open("datos")
ui.eyebrow("Materia prima")
ui.h2("Los datos")
ui.lead(
    "156 observaciones semanales de ventas en euros, con 5 señales externas registradas "
    "cada semana: intensidad de promoción, huelgas, conflicto bélico, problemas logísticos "
    "y otras incidencias. Sin huecos, sin duplicados — el reto no está en limpiar los datos, "
    "está en decidir si esas 5 señales realmente ayudan a predecir mejor."
)
ui.kpi_grid([
    {"num": f"{n_weeks}", "label": "observaciones semanales"},
    {"num": "5", "label": "variables exógenas"},
    {"num": "1", "label": "variable objetivo: ventas (€)"},
    {"num": "2023–25", "label": "periodo cubierto"},
])
st.write("")
ui.pipeline(["Datos crudos", "Transformación log", "Test de estacionariedad (ADF)",
             "Train / Test temporal", "SARIMA vs SARIMAX", "Escenarios"])

cols = st.columns(len(EXOG_COLS))
counts = {"promotion": int((df["promotion"] > 0).sum()), "strike": int(df["strike"].sum()),
          "war": int(df["war"].sum()), "logistics": int(df["logistics"].sum()),
          "other_incident": int(df["other_incident"].sum())}
for c, col in zip(cols, EXOG_COLS):
    meta = EVENT_META[col]
    with c:
        pct = counts[col] / n_weeks * 100
        st.markdown(
            f'<div class="co-card" style="text-align:center; padding:1.2rem .8rem;">'
            f'<div class="kpi-num" style="font-size:1.6rem;">{counts[col]}</div>'
            f'<div class="kpi-label">{meta["label"]}<br>{pct:.1f}% de las semanas</div></div>',
            unsafe_allow_html=True,
        )
st.write("")
ui.finding(
    "En el histórico no hay ninguna semana con más de un evento simultáneo — "
    "cada incidente aparece aislado. Por eso el <b>Playground</b> más abajo permite combinarlos: "
    "así se simula lo que el histórico nunca mostró."
)
ui.section_close()

# ============================================================ EXPLORACIÓN ==
ui.section_open("exploracion")
ui.eyebrow("Antes de modelar")
ui.h2("¿Qué nos dicen los datos?")
ui.lead("Tres preguntas simples antes de tocar ningún algoritmo: ¿cómo se mueven las ventas?, "
        "¿se nota el efecto de los eventos a simple vista?, ¿las promociones realmente venden más?")

h3_1, h3_2 = st.columns([1, 1])
with h3_1:
    ui.h3("Evolución de las ventas")
with h3_2:
    pass
st.plotly_chart(charts.sales_evolution(df), use_container_width=True, config={"displayModeBar": False})
ui.finding(
    f"Venta media semanal de <b>{eur(df['sales'].mean())}</b> "
    f"(rango {eur(df['sales'].min())} – {eur(df['sales'].max())}), con una tendencia de fondo "
    "al alza que la media móvil de 12 semanas deja ver con claridad. Hay estacionalidad, pero también ruido — "
    "ahí es donde entran los eventos externos.",
)

ui.h3("Ventas con incidencias superpuestas")
st.plotly_chart(charts.sales_with_events(df), use_container_width=True, config={"displayModeBar": False})
ui.finding(
    "Las caídas más pronunciadas de la serie coinciden visualmente con huelgas y problemas logísticos, "
    "y los picos más altos suelen coincidir con promociones. Es una pista, no una prueba — "
    "para cuantificarla hace falta un modelo."
)

ui.h3("¿Las promociones venden más?")
st.plotly_chart(charts.promotion_boxplot(df), use_container_width=True, config={"displayModeBar": False})
diff_pct = (df.loc[df["promotion"] > 0, "sales"].mean() / df.loc[df["promotion"] == 0, "sales"].mean() - 1) * 100
ui.finding(
    f"Las semanas con promoción activa venden de media un {ui.stat(f'{diff_pct:.1f}% más', 'pos')} que las "
    "semanas sin ella. Es una diferencia visible incluso antes de modelar — buena señal de que la variable "
    "va a aportar."
)
ui.section_close()

# ============================================================ METODOLOGÍA ==
ui.section_open("metodologia")
ui.eyebrow("Cómo se llegó al modelo")
ui.h2("El camino hasta el modelo")
ui.lead(
    "En series temporales no se puede barajar el tiempo: hay que respetar el orden en que "
    "ocurrieron las cosas. Así fue el recorrido, paso a paso."
)
ui.story_steps([
    ("Miramos los datos",
     "Analizamos tres años de ventas para entender el patrón de fondo antes de tocar ningún "
     "modelo: estacionalidad, tendencia, ruido."),
    ("Probamos con solo el pasado",
     "Entrenamos un primer modelo — SARIMA — que únicamente mira el histórico de ventas. Es el "
     "punto de partida honesto: si lo que viene después no lo mejora, el contexto no vale nada."),
    ("Añadimos el contexto",
     "Incorporamos huelgas, guerra, logística y promociones a ese mismo modelo (SARIMAX) para "
     "comprobar si esa información realmente cambiaba algo."),
    (f"Validamos sin hacer trampas",
     f"Reservamos las últimas {model_config['test_weeks']} semanas para comprobar cada modelo con datos que "
     "nunca había visto, y repetimos el experimento en 13 ventanas distintas del pasado para no "
     "fiarnos de un golpe de suerte."),
])

with st.expander("Para quien quiera el detalle técnico — estacionariedad, ACF/PACF, órdenes del modelo"):
    st.markdown(
        "Los modelos ARIMA necesitan una serie **estacionaria** (media y varianza constantes). "
        "Se aplicó el test de Dickey-Fuller aumentado (ADF) sobre la serie original y transformada:"
    )
    adf_cols = st.columns(len(adf_results))
    for c, res in zip(adf_cols, adf_results):
        with c:
            st.metric(res["name"], "Estacionaria" if res["is_stationary"] else "No estacionaria",
                      f'p={res["p_value"]:.4f}')
    st.markdown(
        "La transformación logarítmica estabiliza la varianza y permite leer los coeficientes del "
        "SARIMAX como **cambios porcentuales aproximados** — clave para la sección de explicabilidad. "
        "Con la serie diferenciada y en log, el ACF y PACF confirmaron un modelo "
        f"`SARIMA{model_config['order']}{model_config['seasonal_order']}` con estacionalidad anual (52 semanas)."
    )

st.write("")
st.plotly_chart(charts.train_test_split(df, model_config["test_weeks"]), use_container_width=True,
                 config={"displayModeBar": False})
ui.finding(
    f"Así quedan divididas las {n_weeks} semanas: {n_weeks - model_config['test_weeks']} para entrenar, "
    f"las últimas {model_config['test_weeks']} para comprobar honestamente si el modelo acierta — nunca al revés."
)
ui.section_close()

# ============================================================ MODELO ==
ui.section_open("modelo")
ui.eyebrow("¿Cómo intenta resolverlo?")
ui.h2("SARIMA vs SARIMAX")
ui.lead(
    "El SARIMA baseline solo ve el histórico de ventas. El SARIMAX ve lo mismo, "
    "más las 5 señales externas. La pregunta que responde esta sección es simple: "
    "¿de verdad importa saber que hubo una huelga?"
)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("MAE — error medio", eur(metrics["sarimax"]["mae"]),
              f'-{metrics["improvement_pct"]["mae"]:.1f}% vs SARIMA', delta_color="normal")
with m2:
    st.metric("MAPE — error %", f'{metrics["sarimax"]["mape"]:.2f}%',
              f'-{metrics["improvement_pct"]["mape"]:.1f}% vs SARIMA', delta_color="normal")
with m3:
    st.metric("RMSE", eur(metrics["sarimax"]["rmse"]),
              f'-{metrics["improvement_pct"]["rmse"]:.1f}% vs SARIMA', delta_color="normal")

with st.expander("¿Qué significan MAE, MAPE y RMSE?"):
    st.markdown(
        "- **MAE** — de media, en cuántos euros se equivoca el modelo. Fácil de leer: "
        f"el SARIMAX se equivoca {eur(metrics['sarimax']['mae'])} de media por semana.\n"
        "- **MAPE** — el mismo error, pero en porcentaje. Permite comparar entre negocios de "
        "distinto tamaño. Por debajo del 10% se considera un pronóstico bueno en retail.\n"
        "- **RMSE** — como el MAE, pero penaliza más los errores grandes. Si sube mucho más que "
        "el MAE, es señal de que hay semanas con fallos puntuales grandes."
    )

st.write("")
ui.h3("Predicción vs realidad — últimas semanas de test")
st.plotly_chart(charts.model_comparison(forecast_df), use_container_width=True, config={"displayModeBar": False})
mape_improvement = metrics["improvement_pct"]["mape"]
ui.finding(
    f"Añadir las 5 variables exógenas reduce el error un {ui.stat(f'{mape_improvement:.0f}%', 'pos')} "
    f"en las tres métricas a la vez (MAE, MAPE y RMSE). La serie histórica sola no basta cuando hay "
    "eventos que mueven las cifras — el SARIMAX no solo predice mejor, sabe *por qué* se equivoca menos."
)
ui.section_close()

# ============================================================ EXPLICABILIDAD ==
ui.section_open("explicabilidad")
ui.eyebrow("¿Por qué predice eso?")
ui.h2("Explicabilidad del modelo")
ui.lead(
    "El SARIMAX no es una caja negra: cada variable exógena tiene un coeficiente que se traduce "
    "directamente en un impacto porcentual sobre las ventas, manteniendo todo lo demás constante."
)
st.plotly_chart(charts.coefficient_impact(coefficients_display), use_container_width=True,
                 config={"displayModeBar": False})

labels = {"promotion": "Promoción", "strike": "Huelga", "war": "Guerra",
          "logistics": "Problema logístico", "other_incident": "Otra incidencia"}
cols_c = st.columns(len(coefficients_display))
for c, coef in zip(cols_c, sorted(coefficients_display, key=lambda x: -abs(x["pct_impact"]))):
    with c:
        tone = "pos" if coef["pct_impact"] > 0 else "neg"
        sig = "significativo" if coef["significant"] else "no significativo"
        unit = f"al {PROMO_MAX*100:.0f}% desc." if coef["variable"] == "promotion" else "mientras dura"
        pct_txt = ui.stat(f'{coef["pct_impact"]:+.2f}% en ventas', tone)
        ui.finding(
            f'<b>{labels.get(coef["variable"], coef["variable"])}</b><br>'
            f'{pct_txt} <span style="font-size:.7rem; opacity:.7;">({unit})</span><br>'
            f'<span style="font-size:.75rem; opacity:.75;">p={coef["p_value"]:.3f} · {sig}</span>'
        )
promo_beta = coef_map["promotion"]["beta"]
promo_per_pp = pct_impact_for(promo_beta, 0.01)
promo_at_max = pct_impact_for(promo_beta, PROMO_MAX)
ui.finding(
    f"La promoción es continua (0–{PROMO_MAX*100:.0f}% de descuento en el histórico), así que su coeficiente "
    f"se lee <b>por punto porcentual</b>: +{promo_per_pp:.2f}% de ventas por cada 1pp adicional de descuento. "
    f"Al nivel máximo histórico ({PROMO_MAX*100:.0f}%), eso equivale a un {ui.stat(f'{promo_at_max:+.1f}%', 'pos')} "
    f"en ventas — no al {coef_map['promotion']['pct_impact']:.0f}% que saldría de leer el coeficiente como si "
    "el descuento pudiera llegar al 100%."
)
ui.body(
    "Importante: son efectos estimados *ceteris paribus* (aislando cada variable) y asumen que el "
    "efecto de cada evento es constante en el tiempo — no implican causalidad directa sin un análisis más profundo."
)
ui.section_close()

# ============================================================ PLAYGROUND ==
ui.section_open("playground")
ui.eyebrow("Pruébalo tú mismo")
ui.h2("Playground — simula un escenario")
ui.lead(
    "Mueve las variables como si fueras a planificar las próximas semanas: ¿qué pasa si lanzamos "
    "una promoción agresiva? ¿Y si coincide con una huelga? El modelo genera la predicción en vivo."
)

pg_left, pg_right = st.columns([1, 1.4], gap="large")
with pg_left:
    horizon = st.slider("Horizonte de predicción (semanas)", 1, 8, 4)
    promotion = st.select_slider(
        "Intensidad de promoción", options=[0.0, 0.05, 0.08, 0.12, 0.18],
        value=0.0, format_func=lambda v: "Sin promoción" if v == 0 else f"{v*100:.0f}% descuento",
    )
    st.markdown("**Eventos activos durante todo el horizonte**")
    c1, c2 = st.columns(2)
    with c1:
        strike = st.checkbox("Huelga")
        war = st.checkbox("Guerra / conflicto")
    with c2:
        logistics = st.checkbox("Problema logístico")
        other_incident = st.checkbox("Otra incidencia")

scenario_df = scenario_from_inputs(last_date, horizon, promotion, strike, war, logistics, other_incident)
base_df = scenario_from_inputs(last_date, horizon, 0.0, False, False, False, False)

scenario_result = predict_scenario(final_model, scenario_df)
base_result = predict_scenario(final_model, base_df)

scenario_total = scenario_result["forecast"].sum()
base_total = base_result["forecast"].sum()
delta_pct = (scenario_total / base_total - 1) * 100

with pg_right:
    st.plotly_chart(charts.playground_forecast(df, scenario_result), use_container_width=True,
                     config={"displayModeBar": False})

st.write("")
r1, r2, r3 = st.columns(3)
with r1:
    st.metric(f"Escenario base ({horizon} sem.)", eur(base_total))
with r2:
    st.metric("Tu escenario", eur(scenario_total), f"{delta_pct:+.1f}% vs base")
with r3:
    st.metric("Diferencia en euros", eur(scenario_total - base_total, sign=True))

active = []
if promotion > 0:
    active.append(("promotion", promotion))
if strike:
    active.append(("strike", 1))
if war:
    active.append(("war", 1))
if logistics:
    active.append(("logistics", 1))
if other_incident:
    active.append(("other_incident", 1))

if active:
    ui.h3("¿Por qué esta predicción?")
    drivers = []
    for var, val in active:
        c = coef_map.get(var)
        if not c:
            continue
        impact = pct_impact_for(c["beta"], val)
        sign = "+" if impact > 0 else ""
        drivers.append(f"{sign}{impact:.1f}% por {labels.get(var, var).lower()}")
    driver_text = " · ".join(drivers)
    tone = "pos" if delta_pct > 0 else ("neg" if delta_pct < 0 else "")
    delta_txt = ui.stat(f"{delta_pct:+.1f}%", tone)
    ui.finding(
        f"Respecto al escenario base, tu combinación de eventos mueve la predicción un "
        f"{delta_txt} ({eur(scenario_total - base_total, sign=True)}). "
        f"Los factores en juego: {driver_text}."
    )
else:
    ui.finding("Activa alguna variable arriba para ver cómo cambia la predicción respecto al escenario base.")
ui.section_close()

# ============================================================ RESULTADOS ==
ui.section_open("resultados")
ui.eyebrow("¿Funciona de verdad?")
ui.h2("Resultados")
ui.h3("Performance técnica")
perf_cols = st.columns(3)
perf_rows = [("MAE", "mae", "€"), ("MAPE", "mape", "%"), ("RMSE", "rmse", "€")]
for c, (label, key, unit) in zip(perf_cols, perf_rows):
    with c:
        sarima_v = metrics["sarima"][key]
        sarimax_v = metrics["sarimax"][key]
        fmt = (lambda v: f"{v:,.0f} {unit}".replace(",", ".")) if unit == "€" else (lambda v: f"{v:.2f}{unit}")
        st.markdown(
            f'<div class="co-card" style="text-align:center;">'
            f'<div style="font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:.6rem;">{label}</div>'
            f'<div style="display:flex; justify-content:center; gap:1.6rem;">'
            f'<div><div class="kpi-num" style="font-size:1.3rem; color:var(--red);">{fmt(sarima_v)}</div>'
            f'<div class="kpi-label">SARIMA</div></div>'
            f'<div><div class="kpi-num" style="font-size:1.3rem; color:var(--green);">{fmt(sarimax_v)}</div>'
            f'<div class="kpi-label">SARIMAX</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
sarimax_mape = metrics["sarimax"]["mape"]
ui.finding(
    f"<b>¿Qué significa?</b> Con SARIMAX, el error porcentual medio baja a {ui.stat(f'{sarimax_mape:.2f}%', 'pos')} "
    "— dentro del rango \"bueno\" para forecasting de retail (por debajo del 10%). En la práctica: de cada "
    f"100 € previstos, el modelo se desvía en menos de {sarimax_mape:.0f} €."
)

ui.h3("Backtesting temporal")
ui.body(
    "Un único split train/test puede salir favorable por casualidad. El backtesting repite el "
    "experimento en múltiples ventanas del pasado: entrena hasta la semana T, predice las 4 "
    "siguientes, avanza y repite."
)
st.plotly_chart(charts.backtest_mape(backtest_df), use_container_width=True, config={"displayModeBar": False})
mean_mape_bt = backtest_df["MAPE"].mean()
quality = "excelente" if mean_mape_bt < 5 else ("buena" if mean_mape_bt < 10 else "aceptable")
ui.finding(
    f"MAPE medio del backtesting: <b>{mean_mape_bt:.2f}%</b> — calidad {quality}, estable a lo largo "
    f"de {len(backtest_df)} ventanas distintas. El modelo generaliza, no depende de un periodo de test favorable.",
)

with st.expander("Diagnóstico de residuos — ¿el modelo dejó información sin capturar?"):
    lb_ok = all(r["p_value"] > 0.05 for r in residual_diag["ljung_box"])
    sw_ok = residual_diag["shapiro"]["p_value"] > 0.05
    _mean_abs = abs(residual_diag["mean"])
    _mean_note = ("cercana a 0, correcto." if _mean_abs < 0.01 else
                  "leve sesgo, dentro de lo esperable en datos reales." if _mean_abs < 0.05 else
                  "alejada de 0, revisar.")
    st.markdown(
        f"- **Media de residuos**: {residual_diag['mean']:.5f} — {_mean_note}\n"
        f"- **Test de Ljung-Box** (autocorrelación): {'sin autocorrelación significativa en los lags evaluados — el modelo captura la estructura temporal.' if lb_ok else 'hay autocorrelación residual en algún lag — margen de mejora.'}\n"
        f"- **Test de Shapiro-Wilk** (normalidad): p={residual_diag['shapiro']['p_value']:.4f} — "
        f"{'no se rechaza la normalidad.' if sw_ok else 'los residuos no son perfectamente normales, algo habitual en datos reales y que no invalida el modelo si el resto de diagnósticos es correcto.'}"
    )
ui.section_close()

# ============================================================ IMPACTO ==
ui.section_open("impacto", tight=True)
ui.impact_banner(
    f'Ignorar el contexto cuesta un <span class="accent-neg">{metrics["improvement_pct"]["mape"]:.0f}% más de error.</span> '
    f'Incluirlo lo convierte en un <span class="accent-pos">simulador de decisiones.</span>',
    quote='"El modelo ya no solo predice — permite preguntar qué pasaría antes de que pase."',
)
ui.section_close()

# ============================================================ INSIGHTS ==
ui.section_open("insights")
ui.eyebrow("De los resultados a las conclusiones")
ui.h2("Insights")
ui.insight_card("01", "El contexto importa más que el histórico puro",
                 f"Las 5 variables exógenas reducen el error un {metrics['improvement_pct']['mape']:.0f}% "
                 "frente a un modelo que solo mira el pasado.",
                 "Cualquier forecast de retail debería incorporar un calendario de eventos, no solo ventas históricas.")
ui.insight_card("02", "Las promociones son la palanca de mayor impacto positivo",
                 f"Cada punto porcentual de descuento se asocia a un +{promo_per_pp:.2f}% en ventas "
                 f"({'estadísticamente significativo' if coef_map['promotion']['significant'] else 'no significativo con el histórico actual'}) "
                 f"— hasta un {promo_at_max:+.1f}% al nivel máximo histórico ({PROMO_MAX*100:.0f}%).",
                 "El coeficiente permite estimar si el descuento compensa su coste antes de lanzarlo.")
negative_events = [c for c in coefficients if c["variable"] != "promotion"]
significant_negative = [c for c in negative_events if c["significant"]]
worst_negative = min(significant_negative or negative_events, key=lambda c: c["pct_impact"])
biggest_point_estimate = min(negative_events, key=lambda c: c["pct_impact"])
caveat = ""
if biggest_point_estimate["variable"] != worst_negative["variable"]:
    caveat = (f' <i>{labels.get(biggest_point_estimate["variable"])} muestra un punto estimado aún mayor '
              f'({biggest_point_estimate["pct_impact"]:.1f}%), pero con solo pocas semanas de histórico su '
              f'coeficiente no es estadísticamente significativo (p={biggest_point_estimate["p_value"]:.2f}) — '
              'no hay evidencia suficiente para fiarse del número exacto.</i>')
ui.insight_card("03", f"{labels.get(worst_negative['variable'])} es el evento adverso más fiable de cuantificar",
                 f"Impacto estimado de {worst_negative['pct_impact']:.1f}% en ventas mientras dura "
                 f"(estadísticamente significativo, p={worst_negative['p_value']:.3f}).{caveat}",
                 "Dimensiona el presupuesto de contingencia por tipo de evento, no de forma genérica.")
ui.insight_card("04", "El modelo generaliza, no memoriza un split favorable",
                 f"El backtesting sobre {len(backtest_df)} ventanas temporales distintas confirma un "
                 f"MAPE medio de {mean_mape_bt:.2f}%, sin picos erráticos.",
                 "Da confianza para usarlo en producción, no solo como demostración puntual.")
ui.insight_card("05", "Los residuos no dejan señal sin explotar",
                 "El test de Ljung-Box no detecta autocorrelación relevante en los residuos del modelo.",
                 "No hay patrón obvio pendiente de capturar con los datos disponibles hoy.")
ui.section_close()

# ============================================================ DECISIONES ==
ui.section_open("decisiones")
ui.eyebrow("¿Qué haríamos con esto?")
ui.h2("Decisiones que habilita")
ui.decision_flow("El histórico solo no basta para prever inventario",
                  "Ajustar pedidos según el escenario más probable, no solo la media histórica",
                  "Reducir roturas de stock y sobreinventario",
                  "Error de previsión (MAPE)")
st.write("")
ui.decision_flow(f"{labels.get(worst_negative['variable'])} cuesta un {abs(worst_negative['pct_impact']):.1f}% en ventas",
                  "Cuantificar el impacto financiero antes de que ocurra el evento",
                  "Dimensionar el plan de contingencia por tipo de riesgo",
                  "€ en riesgo por escenario")
st.write("")
ui.decision_flow(f"Las promociones suman +{promo_per_pp:.2f}% por punto porcentual de descuento",
                  "Usar el coeficiente para estimar el retorno de cada nivel de descuento",
                  "Optimizar la intensidad de la promoción, no solo lanzarla",
                  "ROI de la promoción")
ui.section_close()

# ============================================================ LIMITACIONES ==
ui.section_open("limitaciones")
ui.eyebrow("Honestidad ante todo")
ui.h2("Limitaciones")
lc1, lc2 = st.columns(2)
with lc1:
    st.markdown('<p class="limit-col-title can">Lo que el modelo SÍ puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        """<ul class="limit-list">
        <li>Predecir ventas semanales agregadas con un error medio por debajo del 6%.</li>
        <li>Cuantificar el impacto aislado de cada evento externo conocido.</li>
        <li>Simular escenarios individuales y combinados antes de que ocurran.</li>
        <li>Generalizar de forma estable, validado con backtesting en múltiples ventanas.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
with lc2:
    st.markdown('<p class="limit-col-title cannot">Lo que el modelo NO puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        """<ul class="limit-list">
        <li>Predecir eventos nuevos que no existen en el histórico (asume que ocurren, no cuándo).</li>
        <li>Bajar al detalle de tienda, categoría o SKU — trabaja a nivel agregado semanal.</li>
        <li>Garantizar que el efecto de cada variable se mantenga si el negocio cambia estructuralmente.</li>
        <li>Distinguir causalidad de correlación sin un diseño experimental adicional.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
ui.body(
    "Para mejorar el modelo ayudaría: granularidad diaria o por tienda, variables macro adicionales "
    "(precio de la competencia, clima real en vez de proxy), y más historial para separar el efecto de "
    "cada tipo de evento con mayor confianza estadística — hoy strike, war y other_incident tienen pocas "
    "observaciones (entre 4 y 6 semanas cada uno)."
)
ui.section_close()

# ============================================================ CONCLUSIÓN ==
ui.section_open("conclusion")
ui.eyebrow("Del dato a la decisión")
ui.h2("Conclusión")
ui.lead(
    f"Incorporar contexto externo mejora la previsión un {metrics['improvement_pct']['mape']:.0f}% frente al "
    "histórico puro, y convierte el modelo en un simulador de escenarios. El modelo no sustituye la "
    "decisión de negocio — la toma con mejor información."
)
ui.section_close()

ui.footer_minimal(
    name="Borja Mora Méndez",
    repo_url="https://github.com/BORJAMOME/forecast-ventas-retail-app",
    linkedin_url="https://www.linkedin.com/in/borja-mora-mendez/",
    email="borja.mora.mendez@gmail.com",
)
