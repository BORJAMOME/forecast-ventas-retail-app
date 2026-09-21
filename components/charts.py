"""Figuras Plotly reutilizadas en toda la narrativa. Sistema de color consistente:
  - Serie real / principal        → azul marino oscuro (INK / NAVY2)
  - Referencia o modelo base      → gris (MUTED): «lo que hay que superar»
  - Predicción, test, escenario   → color de apoyo (SUPPORT, ámbar): «esto no es el dato real»
  - Efecto positivo               → verde (POSITIVE)
  - Efecto negativo / riesgo      → rojo (NEGATIVE)
Rojo y verde SOLO significan malo/bueno: un modelo no es «bueno» o «malo» por ser uno u otro, así
que SARIMA y SARIMAX no los usan. Los cuatro eventos adversos comparten el rojo y se distinguen por
la FORMA del marcador: son la misma cosa (algo que perjudica a las ventas) contada cuatro veces."""
import pandas as pd
import plotly.graph_objects as go

INK = "#1D2638"
NAVY2 = "#273A5F"
NAVY3 = "#4A628E"
NAVY4 = "#B9C5D6"
MUTED = "#6B7280"
LINE = "#E3DFD5"
POSITIVE = "#6E7F5B"
NEGATIVE = "#C2412E"
NEGATIVE_SOFT = "rgba(194,65,46,0.10)"
SUPPORT = "#B8783C"
SUPPORT_SOFT = "rgba(184,120,60,0.16)"
NAVY_SOFT = "rgba(39,58,95,0.10)"
FONT = "Arial, Helvetica, sans-serif"

EVENT_STYLE = {
    "strike": {"label": "Huelga", "color": NEGATIVE, "symbol": "triangle-down"},
    "war": {"label": "Guerra", "color": NEGATIVE, "symbol": "x"},
    "logistics": {"label": "Problema logístico", "color": NEGATIVE, "symbol": "square"},
    "other_incident": {"label": "Otra incidencia", "color": NEGATIVE, "symbol": "diamond"},
}
EVENT_LABELS = {"promotion": "Promoción", **{k: v["label"] for k, v in EVENT_STYLE.items()}}


def _base_layout(fig, height=420, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12.5),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11.5), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def sales_evolution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["sales"], name="Ventas semanales",
                             line=dict(color=NAVY2, width=1.8), fill="tozeroy", fillcolor=NAVY_SOFT))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma_12"], name="Media móvil de 12 semanas",
                             line=dict(color=SUPPORT, width=2.2, dash="dash")))
    fig.update_yaxes(title_text="Ventas (€)", rangemode="tozero")
    return _base_layout(fig)


def sales_with_events(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["sales"], name="Ventas semanales",
                             line=dict(color=MUTED, width=1.6), opacity=0.8))
    for col, meta in EVENT_STYLE.items():
        mask = df[col] == 1
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df.loc[mask, "date"], y=df.loc[mask, "sales"], mode="markers",
                name=f'{meta["label"]} ({int(mask.sum())} sem.)',
                marker=dict(color=meta["color"], size=11, symbol=meta["symbol"], line=dict(color=INK, width=0.6)),
            ))
    promo = df["promotion"] > 0
    if promo.any():
        fig.add_trace(go.Scatter(
            x=df.loc[promo, "date"], y=df.loc[promo, "sales"], mode="markers",
            name=f"Promoción ({int(promo.sum())} sem.)",
            marker=dict(color=POSITIVE, size=10 + df.loc[promo, "promotion"] * 160, opacity=0.55,
                        line=dict(color=INK, width=0.6)),
        ))
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=440)


def promotion_boxplot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Box(y=df.loc[df["promotion"] == 0, "sales"], name="Sin promoción", marker_color=MUTED, boxmean=True))
    fig.add_trace(go.Box(y=df.loc[df["promotion"] > 0, "sales"], name="Con promoción", marker_color=POSITIVE, boxmean=True))
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=380, legend=False)


def train_test_split(df: pd.DataFrame, test_weeks: int) -> go.Figure:
    train, test = df.iloc[:-test_weeks], df.iloc[-test_weeks:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["date"], y=train["sales"], name="Entrenamiento (el modelo aprende)",
                             line=dict(color=NAVY2, width=1.8)))
    fig.add_trace(go.Scatter(x=test["date"], y=test["sales"], name="Examen (el modelo no lo ve)",
                             line=dict(color=SUPPORT, width=2.6)))
    fig.add_vline(x=test["date"].iloc[0], line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=360)


def model_comparison(forecast_df: pd.DataFrame) -> go.Figure:
    f = forecast_df
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([f["date"], f["date"][::-1]]), y=pd.concat([f["sarimax_upper"], f["sarimax_lower"][::-1]]),
        fill="toself", fillcolor=SUPPORT_SOFT, line=dict(color="rgba(0,0,0,0)"),
        name="Intervalo al 95% del SARIMAX", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["date"], y=f["actual"], name="Ventas reales",
                             line=dict(color=INK, width=2.6), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=f["date"], y=f["sarima_forecast"], name="SARIMA (solo histórico)",
                             line=dict(color=MUTED, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=f["date"], y=f["sarimax_forecast"], name="SARIMAX (con contexto)",
                             line=dict(color=SUPPORT, width=2.4, dash="dash")))
    event = f[[c for c in EVENT_STYLE]].sum(axis=1) > 0
    if event.any():
        fig.add_trace(go.Scatter(x=f.loc[event, "date"], y=f.loc[event, "actual"], mode="markers",
                                 name="Semana con incidencia", marker=dict(color=NEGATIVE, size=13, symbol="diamond",
                                                                            line=dict(color=INK, width=0.8))))
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=440)


def weekly_error(weekly: pd.DataFrame) -> go.Figure:
    """Error absoluto de cada modelo, semana a semana; las semanas con incidencia van sombreadas."""
    fig = go.Figure()
    for _, row in weekly[weekly["event"]].iterrows():
        fig.add_vrect(x0=row["date"] - pd.Timedelta(days=3.5), x1=row["date"] + pd.Timedelta(days=3.5),
                      fillcolor=NEGATIVE_SOFT, line_width=0, layer="below")
    fig.add_trace(go.Bar(x=weekly["date"], y=weekly["sarima_ape"], name="SARIMA (solo histórico)", marker_color=MUTED))
    fig.add_trace(go.Bar(x=weekly["date"], y=weekly["sarimax_ape"], name="SARIMAX (con contexto)", marker_color=SUPPORT))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Semana con incidencia",
                             marker=dict(color=NEGATIVE_SOFT, size=14, symbol="square", line=dict(color=NEGATIVE, width=1))))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title_text="Error (% de las ventas)", ticksuffix="%")
    fig.update_xaxes(tickformat="%d %b", dtick=7 * 24 * 3600 * 1000 * 2)
    return _base_layout(fig, height=380)


def coefficient_impact(coeffs: list, promo_max: float) -> go.Figure:
    """Las estimaciones inciertas se marcan en la etiqueta del eje (no junto a la barra: en móvil chocaba)."""
    labels = {"promotion": f"Promoción<br>(a intensidad máx. {promo_max * 100:.0f}%)", "strike": "Huelga",
              "war": "Guerra", "logistics": "Problema<br>logístico", "other_incident": "Otra<br>incidencia"}
    items = sorted(coeffs, key=lambda c: c["pct_impact"])
    colors = [POSITIVE if c["pct_impact"] > 0 else NEGATIVE for c in items]
    ylabels = [labels[c["variable"]] + ("" if c["significant"] else "<br><i>(incierta)</i>") for c in items]
    text = [f'{c["pct_impact"]:+.1f}%'.replace("-", "−").replace(".", ",") for c in items]
    fig = go.Figure(go.Bar(
        x=[c["pct_impact"] for c in items], y=ylabels, orientation="h",
        marker=dict(color=colors, opacity=[0.9 if c["significant"] else 0.4 for c in items],
                    line=dict(color=colors, width=1.2)),
        text=text, textposition="outside", cliponaxis=False, hoverinfo="skip"))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(title_text="Cambio en las ventas (%)", showgrid=True, gridcolor=LINE, range=[-74, 30])
    fig.update_yaxes(showgrid=False, automargin=True)
    return _base_layout(fig, height=380, legend=False)


def backtest_mape(ctx: pd.DataFrame, test_mape: float) -> go.Figure:
    """MAPE de cada ventana; oscuras las que contienen algún evento adverso. Las dos referencias van en la
    leyenda (no como anotaciones sobre las barras: en móvil se solapaban con ellas)."""
    fig = go.Figure()
    for has_event, name, color in ((False, "Ventana sin eventos adversos", NAVY4), (True, "Ventana con algún evento adverso", NAVY2)):
        part = ctx[ctx["has_event"] == has_event]
        fig.add_trace(go.Bar(x=part["iteration"], y=part["MAPE"], name=name, marker_color=color,
                             text=[f"{v:.1f}".replace(".", ",") for v in part["MAPE"]], textposition="outside",
                             textfont=dict(size=11), cliponaxis=False, hoverinfo="skip"))
    xs = [ctx["iteration"].min() - 0.5, ctx["iteration"].max() + 0.5]
    for y, name, color, dash in ((ctx["MAPE"].mean(), "Media de las 13 ventanas", INK, "dash"),
                                 (test_mape, "Examen de 12 semanas", SUPPORT, "dot")):
        fig.add_trace(go.Scatter(x=xs, y=[y, y], mode="lines", line=dict(color=color, width=2.2, dash=dash),
                                 name=f"{name}: {y:.1f}%".replace(".", ","), hoverinfo="skip"))
    fig.update_xaxes(title_text="Ventana (de más antigua a más reciente)", dtick=1, range=xs)
    fig.update_yaxes(title_text="Error (MAPE, %)", range=[0, ctx["MAPE"].max() * 1.22])
    fig.update_layout(bargap=0.3)
    return _base_layout(fig, height=430)


def playground_forecast(history: pd.DataFrame, scenario: pd.DataFrame, tail_weeks: int = 20) -> go.Figure:
    hist = history.tail(tail_weeks)
    last_date, last_sales = hist["date"].iloc[-1], hist["sales"].iloc[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[last_date] + list(scenario["date"]) + list(scenario["date"][::-1]) + [last_date],
        y=[last_sales] + list(scenario["upper95"]) + list(scenario["lower95"][::-1]) + [last_sales],
        fill="toself", fillcolor=SUPPORT_SOFT, line=dict(color="rgba(0,0,0,0)"),
        name="Intervalo al 95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["sales"], name="Histórico", line=dict(color=INK, width=2)))
    fig.add_trace(go.Scatter(x=[last_date] + list(scenario["date"]), y=[last_sales] + list(scenario["forecast"]),
                             name="Predicción de tu escenario", line=dict(color=SUPPORT, width=2.6, dash="dash"),
                             mode="lines+markers"))
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=420)
