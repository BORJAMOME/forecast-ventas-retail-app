"""Figuras Plotly reutilizadas en toda la narrativa. Sistema de color
consistente en toda la app:
  - Serie principal        → azul marino oscuro (INK / NAVY2)
  - Serie secundaria       → azul marino medio (NAVY3), misma familia
  - Escenario / simulación / test → color de apoyo (SUPPORT, ámbar),
    deliberadamente fuera de la familia azul para que se lea de un
    vistazo como "esto no es el dato real, es una proyección"
  - Resultado positivo     → verde (POSITIVE)
  - Resultado negativo/riesgo → rojo (NEGATIVE)
Los cuatro tipos de evento adverso comparten el mismo rojo y se
distinguen por la FORMA del marcador, no por el color: no son
categorías con significado propio, son la misma cosa (un evento que
perjudica las ventas) contada cuatro veces."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

INK = "#1D2638"
NAVY2 = "#273A5F"
NAVY3 = "#4A628E"
NAVY4 = "#B9C5D6"
MUTED = "#6B7280"
LINE = "#E3DFD5"
POSITIVE = "#6E7F5B"
POSITIVE_SOFT = "rgba(110,127,91,0.14)"
NEGATIVE = "#C2412E"
NEGATIVE_SOFT = "rgba(194,65,46,0.12)"
SUPPORT = "#B8783C"
SUPPORT_SOFT = "rgba(184,120,60,0.14)"
NAVY_SOFT = "rgba(39,58,95,0.10)"
FONT = "Arial, Helvetica, sans-serif"

EVENT_STYLE = {
    "strike": {"label": "Huelga", "color": NEGATIVE, "symbol": "triangle-down"},
    "war": {"label": "Guerra", "color": NEGATIVE, "symbol": "x"},
    "logistics": {"label": "Problema logístico", "color": NEGATIVE, "symbol": "square"},
    "other_incident": {"label": "Otra incidencia", "color": NEGATIVE, "symbol": "diamond"},
}


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
                     font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def sales_evolution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["sales"], name="Ventas semanales",
                              line=dict(color=NAVY2, width=1.8), fill="tozeroy", fillcolor=NAVY_SOFT))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma_12"], name="Media móvil 12 semanas",
                              line=dict(color=NAVY3, width=2, dash="dash")))
    fig.update_yaxes(title_text="Ventas (€)", rangemode="tozero")
    return _base_layout(fig)


def sales_with_events(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["sales"], name="Ventas semanales",
                              line=dict(color=MUTED, width=1.6), opacity=0.75))
    for col, meta in EVENT_STYLE.items():
        mask = df[col] == 1
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df.loc[mask, "date"], y=df.loc[mask, "sales"], mode="markers",
                name=f'{meta["label"]} ({int(mask.sum())} sem.)',
                marker=dict(color=meta["color"], size=11, symbol=meta["symbol"],
                            line=dict(color=INK, width=0.6)),
            ))
    promo_mask = df["promotion"] > 0
    if promo_mask.any():
        sizes = 10 + df.loc[promo_mask, "promotion"] * 160
        fig.add_trace(go.Scatter(
            x=df.loc[promo_mask, "date"], y=df.loc[promo_mask, "sales"], mode="markers",
            name=f"Promoción ({int(promo_mask.sum())} sem.)",
            marker=dict(color=POSITIVE, size=sizes, opacity=0.55, line=dict(color=INK, width=0.6)),
        ))
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=440)


def promotion_boxplot(df: pd.DataFrame) -> go.Figure:
    promo_yes = df.loc[df["promotion"] > 0, "sales"]
    promo_no = df.loc[df["promotion"] == 0, "sales"]
    fig = go.Figure()
    fig.add_trace(go.Box(y=promo_no, name="Sin promoción", marker_color=MUTED, boxmean=True))
    fig.add_trace(go.Box(y=promo_yes, name="Con promoción", marker_color=POSITIVE, boxmean=True))
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=360, legend=False)


def train_test_split(df: pd.DataFrame, test_weeks: int) -> go.Figure:
    train = df.iloc[:-test_weeks]
    test = df.iloc[-test_weeks:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["date"], y=train["sales"], name="Train", line=dict(color=NAVY2, width=1.8)))
    fig.add_trace(go.Scatter(x=test["date"], y=test["sales"], name="Test", line=dict(color=SUPPORT, width=2.4)))
    fig.add_vline(x=test["date"].iloc[0], line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=360)


def model_comparison(forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["actual"], name="Ventas reales",
                              line=dict(color=INK, width=2.4), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["sarima_forecast"], name="SARIMA (baseline)",
                              line=dict(color=NEGATIVE, width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["sarimax_upper"], forecast_df["sarimax_lower"][::-1]]),
        fill="toself", fillcolor=POSITIVE_SOFT, line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["sarimax_forecast"], name="SARIMAX (con eventos)",
                              line=dict(color=POSITIVE, width=2.2, dash="dash")))
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=440)


def coefficient_impact(coeffs: list) -> go.Figure:
    labels = {"promotion": "Promoción (al 18% desc., máx. histórico)", "strike": "Huelga", "war": "Guerra",
              "logistics": "Problema logístico", "other_incident": "Otra incidencia"}
    colors = {"promotion": POSITIVE, "strike": NEGATIVE, "war": NEGATIVE,
              "logistics": NEGATIVE, "other_incident": NEGATIVE}
    items = sorted(coeffs, key=lambda c: c["pct_impact"])
    y = [labels.get(c["variable"], c["variable"]) for c in items]
    x = [c["pct_impact"] for c in items]
    bar_colors = [colors.get(c["variable"], MUTED) for c in items]
    text = [f'{c["pct_impact"]:+.1f}%{"" if c["significant"] else "  (n.s.)"}' for c in items]
    fig = go.Figure(go.Bar(x=x, y=y, orientation="h", marker_color=bar_colors, text=text,
                            textposition="outside", opacity=0.9))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(title_text="Impacto estimado en ventas (%)", showgrid=True, gridcolor=LINE)
    fig.update_yaxes(showgrid=False)
    return _base_layout(fig, height=320, legend=False)


def backtest_mape(bt_df: pd.DataFrame) -> go.Figure:
    mean_mape = bt_df["MAPE"].mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bt_df["iteration"], y=bt_df["MAPE"], marker_color=NAVY2, opacity=0.85,
                          name="MAPE por ventana"))
    fig.add_hline(y=mean_mape, line_dash="dash", line_color=INK,
                  annotation_text=f"Media {mean_mape:.2f}%", annotation_font_color=INK)
    fig.update_xaxes(title_text="Ventana de backtesting")
    fig.update_yaxes(title_text="MAPE (%)")
    return _base_layout(fig, height=340, legend=False)


def playground_forecast(history: pd.DataFrame, scenario: pd.DataFrame, tail_weeks: int = 20) -> go.Figure:
    hist_tail = history.tail(tail_weeks)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_tail["date"], y=hist_tail["sales"], name="Histórico",
                              line=dict(color=INK, width=2)))
    bridge_x = [hist_tail["date"].iloc[-1]] + list(scenario["date"])
    bridge_y = [hist_tail["sales"].iloc[-1]] + list(scenario["forecast"])
    fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, name="Predicción (tu escenario)",
                              line=dict(color=SUPPORT, width=2.6, dash="dash"), mode="lines+markers"))
    band_x = [hist_tail["date"].iloc[-1]] + list(scenario["date"]) + list(scenario["date"][::-1]) + [hist_tail["date"].iloc[-1]]
    band_y = ([hist_tail["sales"].iloc[-1]] + list(scenario["upper95"]) +
              list(scenario["lower95"][::-1]) + [hist_tail["sales"].iloc[-1]])
    fig.add_trace(go.Scatter(x=band_x, y=band_y, fill="toself", fillcolor=SUPPORT_SOFT,
                              line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig.update_yaxes(title_text="Ventas (€)")
    return _base_layout(fig, height=420)


def scenario_bars(rows: list) -> go.Figure:
    """rows: list of dicts {name, total, is_base}"""
    base_val = next((r["total"] for r in rows if r["is_base"]), rows[0]["total"])
    rows_sorted = sorted(rows, key=lambda r: r["total"])
    colors = []
    for r in rows_sorted:
        if r["is_base"]:
            colors.append(MUTED)
        elif r["total"] > base_val * 1.005:
            colors.append(POSITIVE)
        elif r["total"] < base_val * 0.995:
            colors.append(NEGATIVE)
        else:
            colors.append(MUTED)
    fig = go.Figure(go.Bar(
        x=[r["total"] for r in rows_sorted], y=[r["name"] for r in rows_sorted],
        orientation="h", marker_color=colors,
        text=[f'{r["total"]:,.0f} €'.replace(",", ".") for r in rows_sorted], textposition="outside",
    ))
    fig.add_vline(x=base_val, line_dash="dash", line_color=INK)
    fig.update_xaxes(title_text="Ventas estimadas (€)")
    return _base_layout(fig, height=140 + 46 * len(rows_sorted), legend=False)
