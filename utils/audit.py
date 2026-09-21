"""Cifras derivadas que la narrativa necesita para leer el resultado con rigor.

Todo se calcula a partir de los datos y de los artefactos que ya genera `model/train.py`
(test_forecast.csv, backtest.csv, residuals.csv, final_params.json). No se reentrena nada y
no se escribe ningún número a mano: si cambian los datos, cambia el texto.
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import shapiro, wilcoxon
from statsmodels.stats.diagnostic import acorr_ljungbox

from utils.data_loader import EXOG_COLS

ADVERSE = [c for c in EXOG_COLS if c != "promotion"]


def _runs(active: pd.Series) -> list:
    """Longitud de cada racha de semanas consecutivas con el evento activo."""
    runs, current = [], 0
    for value in active.astype(bool):
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


@st.cache_data(show_spinner=False)
def event_summary(df: pd.DataFrame) -> dict:
    """Semanas, episodios y racha más larga de cada señal, más los solapes entre señales."""
    out = {}
    for col in EXOG_COLS:
        runs = _runs(df[col] > 0)
        out[col] = {"weeks": int(sum(runs)), "episodes": len(runs), "max_run": max(runs) if runs else 0,
                    "pct": sum(runs) / len(df) * 100}
    adverse_any = df[ADVERSE].sum(axis=1)
    out["adverse_weeks"] = int((adverse_any > 0).sum())
    out["adverse_overlap"] = int((adverse_any > 1).sum())
    out["promo_with_adverse"] = int(((df["promotion"] > 0) & (adverse_any > 0)).sum())
    out["quiet_weeks"] = int(((df["promotion"] == 0) & (adverse_any == 0)).sum())
    return out


@st.cache_data(show_spinner=False)
def sales_profile(df: pd.DataFrame) -> dict:
    """Nivel, crecimiento anual, variación semanal y contexto de las semanas con evento."""
    yearly = df.groupby(df["date"].dt.year)["sales"].mean()
    local_level = df["sales"].rolling(12, center=True, min_periods=6).mean()
    relative = df["sales"] / local_level - 1
    promo = df["promotion"] > 0
    return {
        "yearly_mean": {int(y): float(v) for y, v in yearly.items()},
        "yearly_growth": [float(v) * 100 for v in yearly.pct_change().dropna()],
        "weekly_change_median": float(df["sales"].pct_change().abs().median() * 100),
        "adverse_vs_local": {c: float(relative[df[c] == 1].mean() * 100) for c in ADVERSE},
        "promo_mean": float(df.loc[promo, "sales"].mean()),
        "no_promo_mean": float(df.loc[~promo, "sales"].mean()),
        "promo_per_year": {int(y): int(n) for y, n in df[promo].groupby(df["date"].dt.year).size().items()},
        "promo_level_mean": float(df.loc[promo, "promotion"].mean()),
        "max_sales": (df.loc[df["sales"].idxmax(), "date"], float(df["sales"].max())),
    }


@st.cache_data(show_spinner=False)
def test_breakdown(forecast_df: pd.DataFrame) -> dict:
    """De dónde sale la mejora en las 12 semanas de examen: semanas con incidencia frente a normales."""
    f = forecast_df.copy()
    f["event"] = f[ADVERSE].sum(axis=1) > 0
    for m in ("sarima", "sarimax"):
        f[f"{m}_ape"] = (f[f"{m}_forecast"] - f["actual"]).abs() / f["actual"] * 100
        f[f"{m}_bias"] = (f[f"{m}_forecast"] - f["actual"]) / f["actual"] * 100
    ev, qu = f[f["event"]], f[~f["event"]]
    covered = ((f["actual"] >= f["sarimax_lower"]) & (f["actual"] <= f["sarimax_upper"])).sum()
    width = lambda m: float(((f[f"{m}_upper"] - f[f"{m}_lower"]) / f["actual"]).mean() * 100)
    return {
        "n": len(f), "n_event": int(f["event"].sum()), "n_quiet": int((~f["event"]).sum()),
        "event_weeks": list(ev["date"]), "n_promo": int((f["promotion"] > 0).sum()),
        "event_ape": {m: float(ev[f"{m}_ape"].mean()) for m in ("sarima", "sarimax")},
        "quiet_ape": {m: float(qu[f"{m}_ape"].mean()) for m in ("sarima", "sarimax")},
        "wins": int((f["sarimax_ape"] < f["sarima_ape"]).sum()),
        "bias": {m: float(f[f"{m}_bias"].mean()) for m in ("sarima", "sarimax")},
        "coverage": int(covered), "width": {m: width(m) for m in ("sarima", "sarimax")},
        "wilcoxon_p": float(wilcoxon(f["sarima_ape"], f["sarimax_ape"]).pvalue),
        "weekly": f[["date", "event", "sarima_ape", "sarimax_ape"]].reset_index(drop=True),
    }


@st.cache_data(show_spinner=False)
def naive_baselines(df: pd.DataFrame, test_weeks: int) -> dict:
    """Referencias sin modelo, sobre las mismas semanas de examen: si un modelo no las mejora, no aporta."""
    test = df.iloc[-test_weeks:]
    last_value = df["sales"].iloc[-test_weeks - 1]
    seasonal = df["sales"].shift(52).iloc[-test_weeks:]
    mape = lambda forecast: float((abs(test["sales"] - forecast) / test["sales"]).mean() * 100)
    return {"last_value": mape(last_value), "seasonal": mape(seasonal)}


@st.cache_data(show_spinner=False)
def backtest_context(df: pd.DataFrame, backtest_df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """Qué eventos caen en cada ventana y cuántas semanas de ese evento había visto ya el modelo."""
    rows = []
    for _, r in backtest_df.iterrows():
        start = pd.Timestamp(r["test_start"])
        window = df[(df["date"] >= start) & (df["date"] < start + pd.Timedelta(weeks=horizon))]
        seen = df[df["date"] <= pd.Timestamp(r["train_end"])]
        present = [c for c in ADVERSE if (window[c] > 0).any()]
        rows.append({
            "iteration": int(r["iteration"]), "test_start": start, "MAPE": float(r["MAPE"]),
            "has_event": bool(present), "events": present,
            "seen": {c: int((seen[c] > 0).sum()) for c in present},
            "promo_in_window": bool((window["promotion"] > 0).any()),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def clean_residuals(residuals_df: pd.DataFrame, burn_in: int = 54) -> dict:
    """Diagnóstico de residuos sin las semanas de arranque del filtro.

    Con d=1 y D=1 (s=52) los primeros 1 + 52 valores son inicialización difusa, no innovaciones
    del modelo: su residuo llega a 11,5 en escala log frente a una desviación típica de 0,06 y
    contamina cualquier test. La semana 105 (30/12/2024) queda 52 semanas después de la 53 y
    arrastra el mismo eco, así que también se aparta. Se informa siempre de cuántas se excluyen.
    """
    r = residuals_df.reset_index(drop=True)["residual"]
    echo = [104] if len(r) > 104 and abs(r.iloc[104]) > 1 else []
    kept = r.drop(index=list(range(burn_in)) + echo)
    lb = acorr_ljungbox(kept, lags=[10, 20], return_df=True)
    sw = shapiro(kept)
    return {
        "n_total": len(r), "n_kept": len(kept), "n_excluded": len(r) - len(kept),
        "mean": float(kept.mean()), "sd": float(kept.std()),
        "lb": {10: float(lb["lb_pvalue"].iloc[0]), 20: float(lb["lb_pvalue"].iloc[1])},
        "shapiro_w": float(sw[0]), "shapiro_p": float(sw[1]),
        "first_residual": float(r.iloc[0]), "echo_date": residuals_df["date"].iloc[104] if echo else None,
        "echo_value": float(r.iloc[104]) if echo else None,
    }


def final_betas(final_params: dict) -> dict:
    """Coeficientes del modelo final (156 semanas), los mismos que generan la predicción del Playground."""
    values = dict(zip(final_params["names"], final_params["values"]))
    return {c: float(values[c]) for c in EXOG_COLS}
