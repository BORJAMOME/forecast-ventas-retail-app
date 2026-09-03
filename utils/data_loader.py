"""Carga de datos y artefactos del modelo, con cache de Streamlit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "SALES_FORECASTING_VARIABLES_EXOGENAS.xlsx"
ARTIFACTS = ROOT / "model" / "artifacts"

EXOG_COLS = ["promotion", "strike", "war", "logistics", "other_incident"]
# El color solo se usa donde hay significado semántico real (positivo/negativo).
EVENT_META = {
    "promotion": {"label": "Promoción", "color": "#6E7F5B", "kind": "continuous"},
    "strike": {"label": "Huelga", "color": "#C2412E", "kind": "binary"},
    "war": {"label": "Guerra", "color": "#C2412E", "kind": "binary"},
    "logistics": {"label": "Problema logístico", "color": "#C2412E", "kind": "binary"},
    "other_incident": {"label": "Otra incidencia", "color": "#C2412E", "kind": "binary"},
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    df["log_sales"] = np.log(df["sales"])
    df["ma_12"] = df["sales"].rolling(window=12, center=True).mean()
    return df


@st.cache_data(show_spinner=False)
def load_json(name: str):
    with open(ARTIFACTS / name, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(ARTIFACTS / name, parse_dates=["date"] if "date" in open(ARTIFACTS / name, encoding="utf-8").readline() else None)
    return df


@st.cache_resource(show_spinner=False)
def load_final_model():
    """Reconstruye el modelo final aplicando los parámetros ya optimizados
    (`final_params.json`) con `.filter()` — instantáneo, sin reentrenar.
    Evita cargar un pickle de cientos de MB: el estado del filtro de Kalman
    para un SARIMAX estacional (s=52) es lo que ocupa espacio, no los datos."""
    df = load_data()
    config = load_json("model_config.json")
    params = load_json("final_params.json")
    model = SARIMAX(
        df["log_sales"], exog=df[EXOG_COLS], order=tuple(config["order"]),
        seasonal_order=tuple(config["seasonal_order"]),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.filter(params["values"])


def artifacts_ready() -> bool:
    return (ARTIFACTS / "final_params.json").exists() and (ARTIFACTS / "metrics.json").exists()
