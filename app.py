"""
Previsión de Ventas Retail — SARIMA vs SARIMAX
Reportaje digital de datos: del problema de negocio a la decisión, pasando por los datos, el método,
la evidencia y sus límites.

Narrativa (13 beats):
  LEDE → 01 El problema → 02 Los datos → 03 Antes de modelar → 04 El camino hasta el modelo →
  05 Los dos modelos → 06 ¿De dónde viene la mejora? → 07 ¿Cuánto podemos fiarnos? →
  08 ¿Cuánto pesa cada señal? → 09 Ponlo a prueba → 10 ¿Qué hemos descubierto? →
  11 ¿Qué podría hacer el negocio? → 12 Lo que sabemos y lo que no → 13 Del dato a la decisión

Regla de copy: primero se explica qué significa, después se pone el nombre técnico (en notas
«Detalle técnico» o entre paréntesis).

La composición vive en assets/editorial.css + components/editorial.py (sistema editorial
reutilizable); aquí solo hay contenido y datos. Toda cifra del texto sale de los artefactos de
model/artifacts (train.py) o se calcula de ellos y de los datos (utils/audit.py): nada está escrito a mano.

Autor: Borja Mora Méndez
"""
import importlib
from pathlib import Path

import numpy as np
import streamlit as st

from components import charts
from components import editorial as ed
from utils import audit
from utils.data_loader import artifacts_ready, load_csv, load_data, load_final_model, load_json
from utils.forecasting import predict_scenario, scenario_from_inputs

# Streamlit recarga app.py al detectar cambios, pero mantiene en memoria los módulos locales ya
# importados. Tras un despliegue que modifica components/*.py y app.py a la vez, eso deja un
# app.py nuevo llamando a un módulo antiguo (AttributeError). Recargarlos en cada ejecución lo
# evita; el coste es despreciable.
importlib.reload(charts)
importlib.reload(ed)
importlib.reload(audit)

ROOT = Path(__file__).resolve().parent
LECTURA = "16 min"
ACTUALIZADO = "Septiembre 2026"
FUENTE = "Elaboración propia"
PLOT = {"displayModeBar": False}

st.set_page_config(
    page_title="Previsión de Ventas Retail · SARIMA vs SARIMAX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Identidad del proyecto primero (paleta, tipografías); sistema editorial después
ed.load_css(ROOT, "style.css", "editorial.css")

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
final_params = load_json("final_params.json")
forecast_df = load_csv("test_forecast.csv")
backtest_df = load_csv("backtest.csv")
residuals_df = load_csv("residuals.csv")
final_model = load_final_model()


# ---------------------------------------------------------------- formato en español
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]


def es(value: float, decimals: int = 1) -> str:
    """Número con coma decimal, como se escribe en español."""
    return f"{value:.{decimals}f}".replace(".", ",")


def sgn(value: float, decimals: int = 1) -> str:
    """Número con signo y coma decimal; el menos tipográfico (−) para no confundirlo con un guion."""
    return f"{value:+.{decimals}f}".replace("-", "−").replace(".", ",")


def pct(value: float, decimals: int = 1) -> str:
    return f"{es(value, decimals)}%"


def miles(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def eur(value: float, sign: bool = False) -> str:
    text = f"{value:+,.0f}".replace("-", "−") if sign else f"{value:,.0f}"
    return text.replace(",", ".") + " €"


def fecha(ts, year: bool = True) -> str:
    return f"{ts.day} de {MESES[ts.month - 1]}" + (f" de {ts.year}" if year else "")


def pvalue(p: float) -> str:
    return "p &lt; 0,001" if p < 0.001 else f"p = {es(p, 3)}"


# ---------------------------------------------------------------- cifras del texto
n_weeks = len(df)
first_date, last_date = df["date"].min(), df["date"].max()
test_weeks = model_config["test_weeks"]
n_train = n_weeks - test_weeks
imp = metrics["improvement_pct"]
sarima, sarimax = metrics["sarima"], metrics["sarimax"]

ev = audit.event_summary(df)
prof = audit.sales_profile(df)
tb = audit.test_breakdown(forecast_df)
naive = audit.naive_baselines(df, test_weeks)
bt = audit.backtest_context(df, backtest_df)
rd = audit.clean_residuals(residuals_df)
betas = audit.final_betas(final_params)

LABELS = charts.EVENT_LABELS
coef_map = {c["variable"]: c for c in coefficients}
PROMO_MAX = float(df["promotion"].max())


def impact(beta: float, delta: float) -> float:
    """Cambio porcentual real (no lineal) de las ventas al mover una señal `delta` unidades."""
    return (np.exp(beta * delta) - 1) * 100


promo_per_point = impact(coef_map["promotion"]["beta"], 0.01)
promo_at_max = impact(coef_map["promotion"]["beta"], PROMO_MAX)
coefficients_display = [
    {**c, "pct_impact": promo_at_max} if c["variable"] == "promotion" else c for c in coefficients
]
n_significant = sum(c["significant"] for c in coefficients)
significant_names = [LABELS[c["variable"]].lower() for c in coefficients if c["significant"]]
uncertain_names = [LABELS[c["variable"]].lower() for c in coefficients if not c["significant"]]
uncertain_txt = " y ".join(uncertain_names)
sigma = float(np.sqrt(dict(zip(final_params["names"], final_params["values"]))["sigma2"]))

bt_mean, bt_median, bt_max, bt_min = bt["MAPE"].mean(), bt["MAPE"].median(), bt["MAPE"].max(), bt["MAPE"].min()
bt_under_10 = int((bt["MAPE"] < 10).sum())
RANGO = f"entre el {es(bt_median)}% y el {es(bt_mean)}%"
worst = bt.nlargest(2, "MAPE").sort_values("iteration")
worst_events = sorted({e for evs in worst["events"] for e in evs})
worst_seen = [sum(seen.values()) for seen in worst["seen"]]      # semanas del evento ya vistas al entrenar

weekly_change = prof["weekly_change_median"]
yearly = prof["yearly_mean"]
years = sorted(yearly)
growth = prof["yearly_growth"]
promo_diff = (prof["promo_mean"] / prof["no_promo_mean"] - 1) * 100
test_start, test_end = forecast_df["date"].min(), forecast_df["date"].max()
war_dates = df.loc[df["war"] == 1, "date"]
ma_l1 = dict(zip(final_params["names"], final_params["values"]))["ma.L1"]
adf_raw = adf_results[0]

# ============================================================ LEDE ==
ed.skip_link("contexto")
ed.topbar(
    "Previsión de Ventas · Retail",
    [("contexto", "Problema"), ("datos", "Datos"), ("metodo", "Método"), ("modelos", "Modelos"),
     ("validacion", "Validación"), ("efectos", "Efectos"), ("playground", "Playground"),
     ("resultado", "Resultado"), ("conclusion", "Conclusión")],
    back_url="https://borjamora.es/",
)
ed.anchor_scroll()
ed.lede(
    kicker="Machine Learning Case Study · Series temporales",
    headline="¿Cuánto mejora una previsión de ventas cuando sabe qué estaba pasando <em>fuera</em> de la tienda?",
    deck=(
        "Una cadena retail necesita saber cuánto va a vender cada semana para decidir cuánto stock pedir y cuánto "
        f"personal programar. Comparé dos modelos sobre {n_weeks} semanas de ventas: uno que solo mira el histórico "
        "y otro que, además, sabe si esa semana hubo promoción, huelga, conflicto o problemas logísticos. La "
        "pregunta es <b>cuál acierta más</b> y, sobre todo, <b>cuánto de esa mejora es fiable</b>."
    ),
    meta=[
        ("Autor", "Borja Mora Méndez"),
        ("Stack", "Python · statsmodels (SARIMA y SARIMAX) · Streamlit"),
        ("Datos", f"{n_weeks} semanas · {MESES[first_date.month - 1][:3]} {first_date.year} – "
                  f"{MESES[last_date.month - 1][:3]} {last_date.year}"),
        ("Actualizado", ACTUALIZADO),
        ("Lectura", LECTURA),
    ],
)

# ============================================================ 01 · EL PROBLEMA ==
ed.beat(
    "contexto", "01", "Problema", "El problema",
    deck=(
        "Prever ventas es lo que decide cuánto stock pedir, cuánta gente programar y cuándo lanzar una promoción. "
        "Un modelo que solo mira el histórico aprende la tendencia y las repeticiones del calendario, pero <b>no "
        "puede saber que esa semana hubo una huelga</b>. Y cuando la previsión falla, el coste aparece en dos "
        "sitios: estanterías vacías o un almacén lleno."
    ),
)
ed.band(
    "La pregunta de negocio",
    '¿Puede una previsión tener en cuenta <span class="accent">lo que ocurre fuera del histórico</span> y, además, '
    "servir para preguntarse «¿y si…?» antes de decidir?",
    "No quería solo un predictor: quería una herramienta para simular escenarios de inventario, personal y "
    "campañas. Pero antes de simular hay que responder a otra pregunta: cuánto se puede confiar en el modelo.",
)

# ============================================================ 02 · LOS DATOS ==
ed.beat(
    "datos", "02", "Datos", "Los datos",
    deck=(
        f"Trabajamos con <b>{n_weeks} observaciones semanales</b> de ventas, en euros, entre "
        f"{MESES[first_date.month - 1]} de {first_date.year} y {MESES[last_date.month - 1]} de {last_date.year}. "
        "Cada semana viene acompañada de cinco señales que indican qué estaba pasando: la intensidad de la "
        "promoción, y si hubo huelga, conflicto bélico, problemas logísticos u otras incidencias. No hay huecos ni "
        "semanas duplicadas, así que el trabajo estuvo en decidir si esas cinco señales ayudaban de verdad a prever "
        "mejor."
    ),
)
ed.provenance([
    ("Dataset", "SALES_FORECASTING_VARIABLES_EXOGENAS.xlsx"),
    ("Observaciones", f"{n_weeks} semanas, del {fecha(first_date)} al {fecha(last_date)}"),
    ("Variable a prever", "Ventas semanales (€)"),
    ("Señales usadas", "5: promoción, huelga, guerra, logística y otras incidencias"),
    ("Señales sin usar", "Índice de precios y temperatura (están en el fichero)"),
    ("Valores vacíos", f"{int(df.drop(columns=['ma_12', 'log_sales']).isna().sum().sum())} · sin huecos ni semanas duplicadas"),
    ("Elaboración", FUENTE),
])

ed.subhead("¿Con qué frecuencia ocurre cada cosa?", level="wide")
with ed.figure("senales"):
    ed.cols([
        {"tag": f"{ev['promotion']['weeks']} semanas · {ev['promotion']['episodes']} episodios",
         "title": "Promoción",
         "text": f"Con una intensidad de entre el {df.loc[df['promotion'] > 0, 'promotion'].min() * 100:.0f}% y el "
                 f"{PROMO_MAX * 100:.0f}%. Es lo único que decide el propio negocio."},
        {"tag": f"{ev['strike']['weeks']} semanas · {ev['strike']['episodes']} episodios",
         "title": "Huelga", "text": f"Rachas de hasta {ev['strike']['max_run']} semanas seguidas."},
        {"tag": f"{ev['war']['weeks']} semanas · {ev['war']['episodes']} episodio" + ("s" if ev["war"]["episodes"] > 1 else ""),
         "title": "Guerra",
         "text": f"Una sola racha, del {war_dates.min().day} de {MESES[war_dates.min().month - 1]} al "
                 f"{war_dates.max().day} de {MESES[war_dates.max().month - 1]} de {war_dates.max().year}."},
        {"tag": f"{ev['logistics']['weeks']} semanas · {ev['logistics']['episodes']} episodios",
         "title": "Problema logístico", "text": f"Rachas de hasta {ev['logistics']['max_run']} semanas seguidas."},
        {"tag": f"{ev['other_incident']['weeks']} semanas · {ev['other_incident']['episodes']} episodios",
         "title": "Otra incidencia", "text": f"Rachas de hasta {ev['other_incident']['max_run']} semanas seguidas."},
    ], count=5)
ed.insight(
    "Más que el porcentaje de semanas, importa el número de <b>episodios</b>. Cuando un evento aparece "
    "pocas veces, el modelo puede medir cuánto cambiaron las ventas esas semanas, pero <b>no puede separar su "
    "efecto de todo lo demás que pasó a la vez</b>. La guerra, por ejemplo, es un único episodio.",
    aside=(
        f"Los cuatro eventos adversos no coinciden nunca entre sí ({ev['adverse_weeks']} semanas adversas, "
        f"{ev['adverse_overlap']} con más de un evento). Solo {ev['promo_with_adverse']} semanas combinan una "
        "promoción con un evento adverso. Por eso combinarlos en el Playground es <i>extrapolar</i>, no repetir "
        "algo que el histórico ya haya mostrado."
    ),
)

# ============================================================ 03 · ANTES DE MODELAR ==
ed.beat(
    "exploracion", "03", "Antes de modelar", "Antes de modelar, mirar los datos",
    deck=(
        "Antes de tocar ningún modelo me hice tres preguntas sencillas: <b>¿cómo se mueven las ventas?</b> ¿Se "
        "nota el efecto de los eventos a simple vista? Y ¿venden más las semanas con promoción?"
    ),
)

ed.subhead("¿Cómo se mueven las ventas?", level="wide")
with ed.split("evolucion", "8-4") as (viz, txt):
    with viz:
        st.plotly_chart(charts.sales_evolution(df), use_container_width=True, config=PLOT)
        ed.caption("FIG. 01", "Ventas semanales (€) y su media móvil de 12 semanas.", FUENTE)
    with txt:
        ed.insight(
            f"Las ventas medias por semana pasan de {eur(yearly[years[0]])} en {years[0]} a "
            f"{eur(yearly[years[1]])} en {years[1]} y {eur(yearly[years[2]])} en {years[2]}: un {pct(growth[0], 0)} "
            f"y un {pct(growth[1], 0)} más cada año. Sobre esa tendencia hay altibajos: una semana típica cambia "
            f"un {pct(weekly_change)} respecto a la anterior. Ahí es donde pueden entrar los eventos externos."
        )

ed.subhead("¿Se nota el efecto de los eventos a simple vista?", level="wide")
with ed.figure("eventos"):
    st.plotly_chart(charts.sales_with_events(df), use_container_width=True, config=PLOT)
    ed.caption("FIG. 02", "Ventas semanales con las semanas de evento señaladas. El tamaño de la burbuja verde es la "
                          "intensidad de la promoción.", FUENTE)
adv = prof["adverse_vs_local"]
ed.insight(
    f"Las semanas de huelga venden de media un {es(abs(adv['strike']), 0)}% menos que su entorno; las de guerra, un "
    f"{es(abs(adv['war']), 0)}%; las de problemas logísticos, un {es(abs(adv['logistics']), 0)}%, y las de otras "
    f"incidencias, un {es(abs(adv['other_incident']), 0)}%. Es una <b>pista, no una prueba</b>: muchas cosas cambian "
    "a la vez y hace falta un modelo para separarlas.",
    aside=(
        "«Entorno» = media móvil centrada de 12 semanas, que incluye la propia semana del evento, así que las "
        "caídas reales son algo mayores que las indicadas."
    ),
)

ed.subhead("¿Venden más las semanas con promoción?", level="wide")
with ed.split("promocion", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            f"Las semanas con promoción venden de media {eur(prof['promo_mean'])}, frente a "
            f"{eur(prof['no_promo_mean'])} sin ella: un {pct(promo_diff)} más. Pero esa diferencia bruta mezcla "
            f"varias cosas. Las promociones se concentran en los años recientes ({prof['promo_per_year'][years[0]]} "
            f"semanas en {years[0]}, {prof['promo_per_year'][years[1]]} en {years[1]} y "
            f"{prof['promo_per_year'][years[2]]} en {years[2]}), cuando las ventas base ya eran más altas, y "
            "además <b>la promoción la decide el negocio</b>: no ocurre al azar. Para separar todo eso hace falta "
            "un modelo."
        )
    with viz:
        st.plotly_chart(charts.promotion_boxplot(df), use_container_width=True, config=PLOT)
        ed.caption("FIG. 03", "Distribución de las ventas semanales con y sin promoción. El rombo marca la media.", FUENTE)

# ============================================================ 04 · EL CAMINO HASTA EL MODELO ==
ed.beat(
    "metodo", "04", "Método", "El camino hasta el modelo",
    deck=(
        "En series temporales no se puede barajar el tiempo: cada semana se parece a las anteriores, y mezclar "
        "pasado y futuro sería hacer trampa. Todo el método está pensado para <b>no engañarnos</b>: empezar por "
        "una referencia honesta, examinar con datos que el modelo nunca ha visto y repetir el examen más de una vez."
    ),
)
ed.route(["Ventas semanales", "Escala logarítmica", "Corte temporal", "SARIMA", "SARIMAX", "Backtesting", "Escenarios"])
ed.steps([
    ("Puse las ventas en una escala comparable",
     "Trabajé con el logaritmo de las ventas: así una subida del 10% pesa igual en una semana floja que en una "
     "fuerte, y los coeficientes del modelo se leen como <b>cambios porcentuales</b>."),
    ("Empecé solo con el pasado (SARIMA)",
     "Un modelo que únicamente usa el histórico de ventas: su tendencia y su calendario anual. Es el punto de "
     "partida honesto: <b>si lo que viene después no lo mejora, el contexto no aporta nada</b>."),
    ("Añadí el contexto (SARIMAX)",
     "El mismo modelo, más las cinco señales de cada semana. Así puedo comprobar si saber qué estaba pasando "
     "cambia algo."),
    ("Examiné sin hacer trampas",
     f"Entrené con las primeras {n_train} semanas y reservé las últimas {test_weeks} como examen: datos que el "
     "modelo no había visto. Nunca al revés."),
    ("Repetí el examen en 13 ventanas",
     "Un solo examen puede salir bien por casualidad. Repetí el ejercicio 13 veces: entrenar hasta una semana, "
     "predecir las 4 siguientes, avanzar y volver a empezar."),
])
ed.note(
    f"<b>Detalle técnico.</b> SARIMA({','.join(map(str, model_config['order']))})"
    f"({','.join(map(str, model_config['seasonal_order']))}) ajustado con "
    f"statsmodels sobre el logaritmo de las ventas. El test ADF ya da la serie original como estacionaria "
    f"({pvalue(adf_raw['p_value'])}), pero el modelo diferencia igualmente (d=1, D=1) por la convención del notebook "
    f"original, y el coeficiente MA(1) queda en {sgn(ma_l1, 3)}, un síntoma habitual de sobrediferenciación. Los errores "
    "fuera de muestra se miden igual, pero probar una especificación con menos diferenciación es un paso pendiente."
)

ed.subhead("¿Cómo dividimos el tiempo?", level="wide")
with ed.figure("corte"):
    st.plotly_chart(charts.train_test_split(df, test_weeks), use_container_width=True, config=PLOT)
    ed.caption("FIG. 04", f"Las {n_train} primeras semanas para entrenar y las {test_weeks} últimas como examen.", FUENTE)
ed.insight(
    f"El examen cubre del {fecha(test_start, False)} al {fecha(test_end)}. Un detalle que conviene tener presente al "
    f"leer el resultado: en esas {test_weeks} semanas solo hubo <b>{tb['n_event']} con incidencia</b> y "
    f"{'ninguna' if tb['n_promo'] == 0 else tb['n_promo']} con promoción."
)

# ============================================================ 05 · LOS DOS MODELOS ==
ed.beat(
    "modelos", "05", "Modelos", "Los dos modelos: el que mira el pasado y el que mira el contexto", weight="major",
    deck=(
        "El <b>SARIMA</b> solo ve el histórico de ventas. El <b>SARIMAX</b> ve lo mismo y, además, las cinco señales "
        "de cada semana. La pregunta es sencilla: <b>¿de verdad importa saber qué estaba pasando?</b>"
    ),
)
ed.subhead("¿Qué predijo cada modelo en las semanas de examen?", level="wide")
with ed.figure("comparacion"):
    st.plotly_chart(charts.model_comparison(forecast_df), use_container_width=True, config=PLOT)
    ed.caption("FIG. 05", "Ventas reales y predicción de cada modelo en las semanas de examen. La banda es el intervalo "
                          "de predicción al 95% del SARIMAX; los rombos rojos marcan las semanas con incidencia.",
               "Modelos entrenados en este proyecto")
ed.subhead("En resumen")
ed.metrics([
    ("Error medio (MAE)", eur(sarimax["mae"]),
     f"Con el SARIMAX, la previsión de una semana se desvía de media {eur(sarimax['mae'])}; con el SARIMA, "
     f"{eur(sarima['mae'])}."),
    ("Error porcentual (MAPE)", pct(sarimax["mape"]),
     f"De cada 100 € que se venden, el SARIMAX se equivoca de media en {es(sarimax['mape'])} €; el SARIMA, en "
     f"{es(sarima['mape'])} €."),
    ("Fallos grandes (RMSE)", eur(sarimax["rmse"]),
     f"Como el error medio, pero castiga más las semanas en que el modelo falla por mucho. SARIMA: "
     f"{eur(sarima['rmse'])}."),
    ("Referencia sin modelo", pct(naive["last_value"]),
     f"Repetir las ventas de la última semana conocida se equivoca un {es(naive['last_value'])}%. El SARIMA apenas "
     f"mejora eso ({pct(sarima['mape'])}); el SARIMAX, sí."),
])
ed.insight(
    f"Con las cinco señales, el error baja entre un {min(imp.values()):.0f}% y un {max(imp.values()):.0f}% según la "
    "métrica: el SARIMAX gana en las tres. Es un buen resultado, pero <b>es el resultado de "
    f"{test_weeks} semanas concretas</b>. La siguiente pregunta es de dónde sale esa mejora.",
    aside=(
        f"El intervalo al 95% del SARIMAX contiene el dato real en {tb['coverage']} de {tb['n']} semanas y mide de "
        f"media el {es(tb['width']['sarimax'], 0)}% de las ventas de esa semana; el del SARIMA, el "
        f"{es(tb['width']['sarima'], 0)}%. Los intervalos son condicionales: suponen que se conocen las señales de "
        "cada semana."
    ),
)

# ============================================================ 06 · ¿DE DÓNDE VIENE LA MEJORA? ==
ed.beat(
    "origen", "06", "Origen de la mejora", "¿De dónde viene la mejora?",
    deck=(
        f"Un {imp['mape']:.0f}% menos de error suena a que el contexto mejora la previsión <b>todas las semanas</b>. "
        f"Separar las {test_weeks} semanas del examen cuenta otra historia."
    ),
)
ed.subhead("Semana a semana", level="wide")
with ed.figure("error-semanal"):
    st.plotly_chart(charts.weekly_error(tb["weekly"]), use_container_width=True, config=PLOT)
    ed.caption("FIG. 06", "Error absoluto de cada modelo en cada semana de examen, como porcentaje de las ventas de esa "
                          "semana. Sombreadas, las semanas con incidencia.", "Modelos entrenados en este proyecto")
ed.metrics([
    (f"Semanas con incidencia ({tb['n_event']})", pct(tb["event_ape"]["sarimax"]),
     f"El SARIMAX se equivoca un {es(tb['event_ape']['sarimax'])}% de media en las semanas con incidencia; el SARIMA, "
     f"un {es(tb['event_ape']['sarima'])}%. Aquí saber lo que pasaba lo cambia casi todo."),
    (f"Semanas sin eventos ({tb['n_quiet']})", pct(tb["quiet_ape"]["sarimax"]),
     f"En las semanas normales, el SARIMAX se equivoca un {es(tb['quiet_ape']['sarimax'])}% y el SARIMA un "
     f"{es(tb['quiet_ape']['sarima'])}%: una diferencia pequeña."),
    ("Semanas en que gana el SARIMAX", f"{tb['wins']} de {tb['n']}",
     f"Pierde en las otras {tb['n'] - tb['wins']}. Añadir contexto no mejora todas las semanas."),
])
ed.insight(
    f"La mayor parte de la mejora viene de <b>{tb['n_event']} semanas</b>. Cuando no ocurre nada especial, ambos "
    "modelos rinden casi igual: el contexto protege sobre todo de los sustos."
)
ed.subhead("Dos advertencias sobre el examen")
ed.passage(
    f"Con {tb['n']} semanas no se puede afirmar que la diferencia sea concluyente: una prueba estadística sobre los "
    f"errores de cada semana da {pvalue(tb['wilcoxon_p'])}, lejos del 0,05 que suele exigirse. Y hay una segunda "
    "trampa: en este examen <b>el modelo conoce de antemano qué eventos ocurrirán esas semanas</b>. En la vida "
    "real, una promoción se conoce porque se planifica; una huelga o un conflicto, no. "
    f"El {imp['mape']:.0f}% es lo que se gana <b>si se sabe qué va a pasar</b>.",
    aside=(
        "Prueba de Wilcoxon de rangos con signo (dos colas) sobre el error porcentual absoluto de cada semana "
        f"(n={tb['n']}). "
        + ("Además, ninguna de las semanas de examen tiene promoción: el examen no dice nada sobre lo bien que se "
           "prevén las promociones." if tb["n_promo"] == 0 else "")
    ),
)

# ============================================================ 07 · ¿CUÁNTO PODEMOS FIARNOS? ==
ed.beat(
    "validacion", "07", "Validación", "¿Cuánto podemos fiarnos?", weight="major",
    deck=(
        "Un solo examen puede salir bien por casualidad. Por eso repetí el experimento en 13 ventanas del pasado: "
        "entrenar hasta una semana, predecir las 4 siguientes, avanzar y volver a empezar. Así se ve <b>cuánto "
        "varía el error</b> de un momento a otro."
    ),
)
ed.subhead("¿Es estable el error?", level="wide")
with ed.figure("backtest"):
    st.plotly_chart(charts.backtest_mape(bt, sarimax["mape"]), use_container_width=True, config=PLOT)
    ed.caption("FIG. 07", "Error porcentual (MAPE) del SARIMAX en cada una de las 13 ventanas de 4 semanas.",
               "Modelo entrenado en este proyecto")
ed.metrics([
    ("Error medio", pct(bt_mean),
     f"Media de las 13 ventanas. Es más alto que el {es(sarimax['mape'])}% del examen: aquel examen fue de los "
     "favorables."),
    ("Error típico", pct(bt_median), "Mediana: la mitad de las ventanas queda por debajo. Las peores arrastran la media."),
    ("Peor ventana", pct(bt_max),
     f"{int((bt['MAPE'] > 13).sum())} de las 13 ventanas superan el 13%; {bt_under_10} de 13 quedan por debajo del 10%."),
])
ed.insight(
    f"El {es(sarimax['mape'])}% del examen fue mejor que el error medio de las 13 ventanas ({es(bt_mean)}%). Si hubiera "
    f"que quedarse con un solo número para planificar, sería <b>{RANGO}, con semanas malas de hasta el "
    f"{es(bt_max)}%</b>."
)

ed.subhead("¿Dónde falla?")
worst_names = " y ".join(LABELS[e].lower() for e in worst_events)
ed.insight(
    f"Las dos peores ventanas coinciden con {'la ' if worst_events == ['war'] else ''}{worst_names}. En la primera, "
    f"el modelo había visto {worst_seen[0]} semanas de ese tipo de evento al entrenar; en la segunda, "
    f"{worst_seen[1]}. <b>No puede cuantificar lo que nunca ha visto</b>, y esa es la limitación más importante "
    "para usarlo con eventos nuevos.",
    aside=(
        "El backtesting solo se ejecutó con el SARIMAX, no con el SARIMA: mide cuánto varía el error del modelo con "
        "contexto, no si sigue siendo mejor que el base en cada ventana. Como en el examen, cada ventana usa los "
        "eventos reales de esas 4 semanas."
    ),
)

ed.subhead("¿Ha dejado el modelo información sin capturar?")
ed.insight(
    f"Revisé los errores del modelo sobre los datos de entrenamiento (los residuos). Para hacerlo bien hay que "
    f"apartar las primeras {rd['n_excluded'] - 1} semanas (el modelo todavía no ha «arrancado» y su error llega a "
    f"{es(rd['first_residual'])} en escala logarítmica, frente a {es(sigma, 2)} habitual) y una más "
    f"({fecha(rd['echo_date'])}) que arrastra ese mismo "
    f"eco. Con las {rd['n_kept']} restantes no aparece autocorrelación clara ({pvalue(rd['lb'][10])} y "
    f"{pvalue(rd['lb'][20])}) y los residuos son compatibles con una distribución normal ({pvalue(rd['shapiro_p'])}).",
    aside=(
        f"Ljung-Box a 10 y 20 retardos y Shapiro-Wilk sobre {rd['n_kept']} de {rd['n_total']} residuos. Con d=1 y D=1 "
        "(s=52) las primeras 53 observaciones son inicialización difusa, no innovaciones del modelo. La semana del "
        f"{fecha(rd['echo_date'])} queda 52 semanas después de la 53 y su residuo ({sgn(rd['echo_value'])}) es "
        "compatible con un eco de esa inicialización. El p de 20 retardos queda cerca del umbral habitual: no hay "
        "evidencia de autocorrelación, pero tampoco un margen amplio."
    ),
)

# ============================================================ 08 · ¿CUÁNTO PESA CADA SEÑAL? ==
ed.beat(
    "efectos", "08", "Explicabilidad", "¿Cuánto pesa cada señal?",
    deck=(
        "El SARIMAX no es una caja negra: cada señal tiene un coeficiente que se traduce en un cambio porcentual de "
        "las ventas <b>manteniendo todo lo demás igual</b>. Lo que importa aquí es <b>cuáles de esos números "
        "merecen confianza</b>."
    ),
)
ed.subhead("¿Cuánto cambian las ventas con cada señal?", level="wide")
with ed.figure("coeficientes"):
    st.plotly_chart(charts.coefficient_impact(coefficients_display, PROMO_MAX), use_container_width=True, config=PLOT)
    ed.caption("FIG. 08", "Cambio estimado de las ventas cuando la señal está activa. Las barras claras son "
                          "estimaciones inciertas (el modelo no las distingue de cero).", "Modelo entrenado en este proyecto")

card_detail = {
    "promotion": (f"{sgn(promo_at_max)}%", f"a intensidad máxima ({PROMO_MAX * 100:.0f}%)",
                  f"Cada punto de intensidad se asocia a un {sgn(promo_per_point, 2)}% de ventas. Es una decisión del "
                  "negocio, no un experimento: no permite hablar de causa."),
    "strike": (None, "mientras dura",
               f"Estimación llamativa, apoyada en solo {ev['strike']['episodes']} episodios de hasta "
               f"{ev['strike']['max_run']} semanas: no hay evidencia suficiente para fiarse del número exacto."),
    "war": (None, "mientras dura",
            f"Estadísticamente clara, pero sale de {'un único episodio' if ev['war']['episodes'] == 1 else str(ev['war']['episodes']) + ' episodios'} "
            f"de {ev['war']['weeks']} semanas seguidas: no se puede separar de todo lo demás que pasó entonces."),
    "logistics": (None, "mientras dura",
                  f"{ev['logistics']['episodes']} episodios de hasta {ev['logistics']['max_run']} semanas: la "
                  "estimación más repetida entre los eventos adversos."),
    "other_incident": (None, "mientras dura",
                       f"Efecto pequeño y poco claro, con {ev['other_incident']['episodes']} episodios. Aun así, dos "
                       "de las semanas de examen fueron de este tipo."),
}
cards = []
for c in sorted(coefficients_display, key=lambda c: (not c["significant"], -abs(c["pct_impact"]))):
    var = c["variable"]
    stat, label, text = card_detail[var]
    color = ("var(--positive)" if c["pct_impact"] > 0 else "var(--negative)") if c["significant"] else "var(--muted)"
    cards.append({
        "tag": f"{'Efecto fiable' if c['significant'] else 'Efecto incierto'} · "
               f"<span style=\"text-transform:none;letter-spacing:0\">{pvalue(c['p_value'])}</span>",
        "title": LABELS[var], "color": color,
        "stat": stat or f"{sgn(c['pct_impact'])}%", "stat_label": label, "text": text,
    })
with ed.figure("efectos-cards"):
    ed.cards(cards, count=len(cards))
ed.insight(
    f"Solo <b>{n_significant} de las {len(coefficients)} señales</b> tienen un efecto que el modelo distingue de cero: "
    f"{', '.join(significant_names[:-1])} y {significant_names[-1]}. {uncertain_txt.capitalize()} tienen estimaciones "
    "llamativas pero inciertas. Y la guerra, aunque clara en los números, descansa en un único episodio.",
    aside=(
        f"Coeficientes del SARIMAX entrenado con las {n_train} semanas de entrenamiento; impacto = exp(β) − 1. El "
        f"modelo final del Playground se reajustó con las {n_weeks} semanas y sus valores cambian algo (promoción a "
        f"intensidad máxima: {sgn(impact(betas['promotion'], PROMO_MAX))}% frente a {sgn(promo_at_max)}%). La "
        "promoción es continua (de 0 a 0,18): se lee por punto de intensidad, que aquí interpretamos como porcentaje "
        "de descuento. Con tan pocos episodios conviene leer los p-valores con cautela."
    ),
)
ed.insight(
    "Todo esto son <b>asociaciones dentro del modelo, no causas</b>: no hay un experimento detrás, y los eventos "
    "coinciden con otras cosas (calendario, estacionalidad). Y la promoción, en concreto, es una decisión del "
    "negocio, no algo que ocurra al azar.",
    tight=True,
)

# ============================================================ 09 · PONLO A PRUEBA ==
ed.beat(
    "playground", "09", "Playground", "Ponlo a prueba", weight="major",
    deck=(
        "Mueve las señales como si fueras a planificar las próximas semanas: ¿qué pasa si lanzo una promoción? ¿Y "
        "si coincide con un problema logístico? El modelo final genera la predicción en vivo. Es un <b>simulador de "
        "escenarios, no una bola de cristal</b>: el resultado depende de que el evento ocurra tal como lo defines."
    ),
)
ed.note(
    "<b>Detalle técnico.</b> El escenario base supone que no hay ninguna señal activa. Las señales que elijas se "
    f"mantienen activas todas las semanas del horizonte. Modelo final entrenado con las {n_weeks} semanas."
)

with ed.split("playground", "4-8") as (pg_left, pg_right):
    with pg_left:
        horizon = st.slider("Horizonte de predicción (semanas)", 1, 8, 4)
        promotion = st.select_slider(
            "Intensidad de promoción", options=[0.0, 0.05, 0.08, 0.12, 0.18], value=0.0,
            format_func=lambda v: "Sin promoción" if v == 0 else f"{v * 100:.0f}% de descuento",
        )
        st.markdown("**Eventos activos**")
        strike = st.checkbox("Huelga")
        war = st.checkbox("Guerra / conflicto")
        logistics = st.checkbox("Problema logístico")
        other_incident = st.checkbox("Otra incidencia")

    scenario_df = scenario_from_inputs(last_date, horizon, promotion, strike, war, logistics, other_incident)
    base_df = scenario_from_inputs(last_date, horizon, 0.0, False, False, False, False)
    scenario_result = predict_scenario(final_model, scenario_df)
    base_result = predict_scenario(final_model, base_df)
    scenario_total, base_total = scenario_result["forecast"].sum(), base_result["forecast"].sum()
    delta_pct = (scenario_total / base_total - 1) * 100

    with pg_right:
        st.plotly_chart(charts.playground_forecast(df, scenario_result), use_container_width=True, config=PLOT)
        ed.caption("FIG. 09", "Últimas 20 semanas reales y predicción de tu escenario. La banda es el intervalo al 95%, "
                              "válido solo si el escenario ocurre tal como lo defines.", "Modelo final entrenado en este proyecto")

ed.metrics([
    (f"Escenario base ({horizon} sem.)", eur(base_total), "Ventas previstas si no ocurre nada especial."),
    ("Tu escenario", eur(scenario_total),
     f"{sgn(delta_pct)}% frente al escenario base." if any([promotion, strike, war, logistics, other_incident])
     else "Activa alguna señal para ver cómo cambia la previsión."),
    ("Diferencia", eur(scenario_total - base_total, sign=abs(scenario_total - base_total) >= 0.5),
     "Ventas que se ganan o se pierden en el horizonte elegido."),
])

active = [(k, v) for k, v in (("promotion", promotion), ("strike", int(strike)), ("war", int(war)),
                               ("logistics", int(logistics)), ("other_incident", int(other_incident))) if v]
if active:
    parts = " · ".join(f"{sgn(impact(betas[var], val))}% por {LABELS[var].lower()}" for var, val in active)
    ed.subhead("¿Por qué esta predicción?")
    ed.insight(
        f"Respecto al escenario base, tu combinación mueve la previsión un <b>{sgn(delta_pct)}%</b> "
        f"({eur(scenario_total - base_total, sign=True)}). Los factores en juego: {parts}. Se combinan "
        "multiplicando, no sumando.",
        tight=True,
    )
    uncertain, too_long = [], []
    for var, val in active:
        e, name = ev[var], LABELS[var].lower()
        if var != "promotion" and not coef_map[var]["significant"]:
            uncertain.append(f"<b>{name}</b> ({pvalue(coef_map[var]['p_value'])}, {e['episodes']} episodios)")
        elif var != "promotion" and e["episodes"] == 1:
            uncertain.append(f"<b>{name}</b> (un único episodio de {e['weeks']} semanas)")
        if horizon > e["max_run"]:
            too_long.append(f"{name}, {e['max_run']} semanas")
    warnings = []
    if uncertain:
        warnings.append("Estimaciones poco fiables en este escenario: " + ", ".join(uncertain) + ". Úsalas como orientación, "
                        "no como cifra.")
    if too_long:
        warnings.append("Duración máxima en el histórico: " + "; ".join(too_long) + f". Tu escenario las mantiene "
                        f"{horizon} semanas seguidas.")
    n_adverse = sum(1 for v, _ in active if v != "promotion")
    if n_adverse >= 2:
        warnings.append("El histórico no contiene ninguna semana con dos eventos adversos a la vez: esta combinación es una "
                        "extrapolación del modelo.")
    elif promotion > 0 and n_adverse:
        warnings.append(f"Solo {ev['promo_with_adverse']} semanas del histórico combinan promoción y evento adverso.")
    if promotion > 0:
        level_n = int((df["promotion"].round(2) == promotion).sum())
        warnings.append(f"La intensidad de promoción del {promotion * 100:.0f}% aparece solo en {level_n} semanas del histórico.")
    for w in warnings:
        ed.note(w)

# ============================================================ 10 · ¿QUÉ HEMOS DESCUBIERTO? ==
ed.beat(
    "resultado", "10", "Resultado", "¿Qué hemos descubierto?", weight="major",
    deck=[
        f"Sobre las {test_weeks} semanas de examen, incorporar el contexto reduce el error de previsión del "
        f"{es(sarima['mape'])}% al {es(sarimax['mape'])}%: un <b>{imp['mape']:.0f}% menos</b>. Es una mejora real, "
        "pero el resultado que conviene llevarse es más matizado.",
        "La mejora se concentra en las semanas con incidencia; en las demás, los dos modelos rinden parecido. Y el "
        "error que cabe esperar de verdad, según las 13 ventanas, es más alto que el del examen.",
    ],
)
ed.band(
    "El matiz",
    'El contexto ayuda sobre todo <span class="accent">cuando ocurre algo</span>. Cuando no ocurre nada, los dos '
    "modelos rinden casi igual.",
    "Añadir señales externas no mejora todas las semanas: protege de las semanas malas.",
)
ed.subhead("En resumen")
ed.metrics([
    ("Mejora en el examen", f"−{imp['mape']:.0f}%",
     f"El error (MAPE) baja de {es(sarima['mape'])}% a {es(sarimax['mape'])}% en {test_weeks} semanas. Es el mejor "
     "caso."),
    ("De dónde viene", f"{tb['n_event']} de {tb['n']}",
     f"La mejora se concentra en las {tb['n_event']} semanas con incidencia; en las otras {tb['n_quiet']} la "
     f"diferencia es pequeña ({es(tb['quiet_ape']['sarima'])}% frente a {es(tb['quiet_ape']['sarimax'])}%)."),
    ("Error esperable", pct(bt_mean),
     f"Error medio del SARIMAX en 13 ventanas distintas (mediana {es(bt_median)}%, máximo {es(bt_max)}%)."),
    ("Efectos fiables", f"{n_significant} de {len(coefficients)}",
     f"Señales cuyo efecto el modelo distingue de cero: {', '.join(significant_names[:-1])} y "
     f"{significant_names[-1]}."),
])
ed.insight(
    "No hemos demostrado que el contexto mejore <b>siempre</b> la previsión. Hemos comprobado que ayuda cuando "
    "ocurre algo, y hemos medido cuánto puede fallar cuando lo que ocurre es nuevo."
)

# ============================================================ 11 · ¿QUÉ PODRÍA HACER EL NEGOCIO? ==
ed.beat(
    "implicaciones", "11", "Implicaciones", "¿Qué podría hacer el negocio con esto?",
    deck=(
        "El modelo no decide por sí solo. Lo que aporta es una forma de <b>plantear la decisión con números y con "
        "márgenes de error</b>."
    ),
)
with ed.figure("decisiones"):
    ed.cols([
        {"tag": "Planificar pedidos", "title": "Pedir con margen, no con un número",
         "text": f"Usar el error esperable ({RANGO}) como colchón habitual, y el "
                 "simulador para ajustar cuando ya se conoce una promoción o una incidencia. Seguimiento: el error de "
                 "previsión (MAPE) de cada semana."},
        {"tag": "Presupuestar riesgos", "title": "Estimar solo lo que se puede estimar",
         "text": f"Los problemas logísticos ({sgn(coef_map['logistics']['pct_impact'], 0)}%) y la guerra "
                 f"({sgn(coef_map['war']['pct_impact'], 0)}%) tienen un efecto medible. Huelga y otras incidencias aún "
                 "no tienen episodios suficientes para dimensionar una contingencia: habría que acumular más histórico."},
        {"tag": "Medir promociones", "title": "Probar antes de optimizar",
         "text": f"El modelo asocia cada punto de intensidad con un {sgn(promo_per_point, 2)}% de ventas, pero la promoción "
                 "es una decisión del negocio. Para saber si un descuento compensa su coste habría que probarlo con "
                 "semanas o tiendas de control y medir el resultado."},
    ])
ed.subhead("Una advertencia importante")
ed.insight(
    f"Son <b>hipótesis de negocio</b>, no conclusiones del modelo. El modelo describe cómo se comportaron las ventas "
    f"en estos {len(years)} años; decidir qué hacer requiere probar la acción y medir su efecto.",
    tight=True,
)

# ============================================================ 12 · LO QUE SABEMOS Y LO QUE NO ==
ed.beat("limitaciones", "12", "Limitaciones", "Lo que sabemos y lo que no sabemos")
ed.cards([
    {"title": "Lo que el modelo sí puede hacer", "color": "var(--positive)", "points": [
        f"Prever las ventas semanales agregadas con un <b>error {RANGO}</b> de media, con semanas de hasta "
        f"el {es(bt_max)}%.",
        f"Cuantificar, con incertidumbre, el efecto de <b>{', '.join(significant_names[:-1])} y "
        f"{significant_names[-1]}</b>.",
        "Simular <b>un evento cada vez</b> y ver cómo cambia la previsión respecto a un escenario base.",
        f"Mantener el error por debajo del 10% en <b>{bt_under_10} de 13 ventanas</b> de 4 semanas.",
    ]},
    {"title": "Lo que el modelo no puede hacer", "color": "var(--negative)", "points": [
        "<b>Predecir cuándo</b> ocurrirá una huelga, un conflicto o una incidencia: solo responde si se le dice "
        "que ocurre.",
        f"Estimar con fiabilidad el impacto de <b>{uncertain_txt}</b>: hay muy pocos episodios.",
        f"Prever bien un <b>evento que apenas ha visto</b>: las dos peores ventanas coinciden con la guerra, cuando el "
        f"modelo había visto {worst_seen[0]} y {worst_seen[1]} semanas de ese evento.",
        "Bajar al detalle de <b>tienda, categoría o producto</b>: trabaja con ventas semanales agregadas.",
        "Distinguir <b>causa de correlación</b>: no hay experimento detrás, y la promoción es una decisión del negocio.",
        "Evaluar <b>combinaciones</b> de eventos adversos: en el histórico nunca coinciden dos.",
    ]},
])
ed.subhead("Y los límites del propio análisis")
with ed.figure("limites-analisis"):
    ed.lists([
        {"title": "Cómo está hecho", "points": [
            "El examen y el backtesting usan los eventos <b>reales</b> de cada semana, no eventos previstos.",
            f"Solo hay {len(years)} años de datos: la estacionalidad anual (52 semanas) se estima con {len(years)} "
            "ciclos, y uno se pierde al diferenciar.",
            "El backtesting solo se ejecutó con el SARIMAX: no se comparó con el SARIMA ventana a ventana.",
            "El intervalo de predicción supone que se conocen las señales de cada semana.",
        ]},
        {"title": "Qué probaría a continuación", "points": [
            "Medir <b>cuánto aporta cada señal por separado</b>, en vez de las cinco a la vez.",
            "Probar una especificación <b>con menos diferenciación</b> y comparar el error.",
            "Incorporar el <b>índice de precios y la temperatura</b>, que ya están en el fichero.",
            "Comparar con otros modelos de referencia (suavizado exponencial, modelos de árboles) y repetir el "
            "backtesting para los dos modelos.",
        ]},
    ])

# ============================================================ 13 · DEL DATO A LA DECISIÓN ==
ed.beat(
    "conclusion", "13", "Conclusión", "Del dato a la decisión", weight="major",
    deck=[
        f"Con cinco señales de contexto, el SARIMAX reduce el error de previsión un <b>{imp['mape']:.0f}%</b> frente al "
        f"SARIMA en el examen de {test_weeks} semanas (del {es(sarima['mape'])}% al {es(sarimax['mape'])}%). La mejora "
        "existe, pero se concentra en las semanas en que ocurre algo; en las demás, los dos modelos rinden casi igual.",
        f"Más útil que ese {imp['mape']:.0f}% es saber <b>cuánto se puede confiar en él</b>: un error "
        f"{RANGO} con semanas de hasta el {es(bt_max)}%, efectos fiables solo "
        f"para {n_significant} de las {len(coefficients)} señales y ninguna garantía ante eventos que el modelo no ha "
        "visto.",
        "El siguiente paso no sería un modelo más complejo. Sería preguntarse: <b>¿qué eventos podemos conocer con "
        "antelación y cuáles no?</b> Solo los primeros mejoran la previsión; los segundos sirven para simular "
        "escenarios.",
    ],
)
ed.colophon("Borja Mora Méndez", [
    ("Repositorio del proyecto", "https://github.com/BORJAMOME/forecast-ventas-retail-app"),
    ("Portfolio", "https://borjamora.es/"),
    ("LinkedIn", "https://www.linkedin.com/in/borjamoramendez/"),
    ("Contacto", "mailto:borja.mora.mendez@gmail.com"),
])
