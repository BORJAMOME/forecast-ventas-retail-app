"""
Entrena SARIMA (baseline) y SARIMAX (con variables exogenas) replicando
la metodologia del notebook original y guarda los artefactos que consume
la app de Streamlit: metricas, coeficientes, backtesting, residuos y los
resultados ajustados (para poder generar escenarios en vivo).

Ejecutar una sola vez (o cuando cambien los datos):
    py -3.10 model/train.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "SALES_FORECASTING_VARIABLES_EXOGENAS.xlsx"
ARTIFACTS = ROOT / "model" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

EXOG_COLS = ["promotion", "strike", "war", "logistics", "other_incident"]
SARIMA_ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 52)
TEST_WEEKS = 12


def mape(y_real, y_pred):
    y_real, y_pred = np.array(y_real), np.array(y_pred)
    mask = y_real != 0
    return float(np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100)


def adf_summary(series, name):
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "name": name,
        "adf_stat": float(result[0]),
        "p_value": float(result[1]),
        "n_lags": int(result[2]),
        "n_obs": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "is_stationary": bool(result[1] < 0.05),
    }


def main():
    print("Cargando dataset...")
    df = pd.read_excel(DATA_PATH)
    df["log_sales"] = np.log(df["sales"])
    df["ma_12"] = df["sales"].rolling(window=12, center=True).mean()

    # -- Estacionariedad ---------------------------------------------------
    adf_results = [
        adf_summary(df["sales"], "Ventas originales"),
        adf_summary(df["sales"].diff(), "Ventas diferenciadas (d=1)"),
        adf_summary(np.log(df["sales"]), "Log(Ventas)"),
        adf_summary(np.log(df["sales"]).diff(), "Delta Log(Ventas)"),
    ]

    # -- Train / Test split (temporal) --------------------------------------
    train = df.iloc[:-TEST_WEEKS].copy()
    test = df.iloc[-TEST_WEEKS:].copy()

    # -- Modelo 1: SARIMA baseline -------------------------------------------
    print("Entrenando SARIMA baseline...")
    sarima_model = SARIMAX(
        train["log_sales"], order=SARIMA_ORDER, seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    sarima_result = sarima_model.fit(disp=False)

    sarima_fc_log = sarima_result.get_forecast(steps=TEST_WEEKS)
    sarima_forecast = np.exp(sarima_fc_log.predicted_mean)
    sarima_ci = sarima_fc_log.conf_int()
    sarima_lower = np.exp(sarima_ci.iloc[:, 0])
    sarima_upper = np.exp(sarima_ci.iloc[:, 1])

    sarima_mae = mean_absolute_error(test["sales"], sarima_forecast)
    sarima_mape = mape(test["sales"], sarima_forecast)
    sarima_rmse = float(np.sqrt(mean_squared_error(test["sales"], sarima_forecast)))

    # -- Modelo 2: SARIMAX con variables exogenas ----------------------------
    print("Entrenando SARIMAX con variables exogenas...")
    sarimax_model = SARIMAX(
        train["log_sales"], exog=train[EXOG_COLS], order=SARIMA_ORDER,
        seasonal_order=SEASONAL_ORDER, enforce_stationarity=False, enforce_invertibility=False,
    )
    sarimax_result = sarimax_model.fit(disp=False)

    sarimax_fc_log = sarimax_result.get_forecast(steps=TEST_WEEKS, exog=test[EXOG_COLS])
    sarimax_forecast = np.exp(sarimax_fc_log.predicted_mean)
    sarimax_ci = sarimax_fc_log.conf_int()
    sarimax_lower = np.exp(sarimax_ci.iloc[:, 0])
    sarimax_upper = np.exp(sarimax_ci.iloc[:, 1])

    sarimax_mae = mean_absolute_error(test["sales"], sarimax_forecast)
    sarimax_mape = mape(test["sales"], sarimax_forecast)
    sarimax_rmse = float(np.sqrt(mean_squared_error(test["sales"], sarimax_forecast)))

    metrics = {
        "test_weeks": TEST_WEEKS,
        "sarima": {"mae": float(sarima_mae), "mape": float(sarima_mape), "rmse": sarima_rmse},
        "sarimax": {"mae": float(sarimax_mae), "mape": float(sarimax_mape), "rmse": sarimax_rmse},
        "improvement_pct": {
            "mae": float((1 - sarimax_mae / sarima_mae) * 100),
            "mape": float((1 - sarimax_mape / sarima_mape) * 100),
            "rmse": float((1 - sarimax_rmse / sarima_rmse) * 100),
        },
    }

    # -- Predicciones train/test para graficar -------------------------------
    forecast_df = pd.DataFrame({
        "date": test["date"].values,
        "actual": test["sales"].values,
        "sarima_forecast": sarima_forecast.values,
        "sarima_lower": sarima_lower.values,
        "sarima_upper": sarima_upper.values,
        "sarimax_forecast": sarimax_forecast.values,
        "sarimax_lower": sarimax_lower.values,
        "sarimax_upper": sarimax_upper.values,
    })
    for col in EXOG_COLS:
        forecast_df[col] = test[col].values

    # -- Coeficientes SARIMAX --------------------------------------------------
    coefs = sarimax_result.params
    pvals = sarimax_result.pvalues
    coefficients = []
    for col in EXOG_COLS:
        if col in coefs.index:
            beta = float(coefs[col])
            pval = float(pvals[col])
            pct = float((np.exp(beta) - 1) * 100)
            coefficients.append({
                "variable": col, "beta": beta, "p_value": pval,
                "pct_impact": pct, "significant": bool(pval < 0.05),
            })

    # -- Backtesting temporal ---------------------------------------------------
    print("Backtesting temporal (esto tarda un poco)...")
    HORIZON, MIN_TRAIN, STEP = 4, 104, 4
    backtest_rows = []
    iteration = 0
    for start_test in range(MIN_TRAIN, len(df) - HORIZON + 1, STEP):
        iteration += 1
        bt_train = df.iloc[:start_test]
        bt_test = df.iloc[start_test:start_test + HORIZON]
        try:
            bt_model = SARIMAX(
                bt_train["log_sales"], exog=bt_train[EXOG_COLS], order=SARIMA_ORDER,
                seasonal_order=SEASONAL_ORDER, enforce_stationarity=False, enforce_invertibility=False,
            )
            bt_result = bt_model.fit(disp=False, maxiter=200)
            bt_fc_log = bt_result.get_forecast(steps=HORIZON, exog=bt_test[EXOG_COLS])
            bt_forecast = np.exp(bt_fc_log.predicted_mean)
            backtest_rows.append({
                "iteration": iteration,
                "train_end": bt_train["date"].iloc[-1].strftime("%Y-%m-%d"),
                "test_start": bt_test["date"].iloc[0].strftime("%Y-%m-%d"),
                "MAE": float(mean_absolute_error(bt_test["sales"], bt_forecast)),
                "MAPE": mape(bt_test["sales"], bt_forecast),
                "RMSE": float(np.sqrt(mean_squared_error(bt_test["sales"], bt_forecast))),
            })
            print(f"  ventana {iteration} OK (MAPE={backtest_rows[-1]['MAPE']:.2f}%)")
        except Exception as exc:  # pragma: no cover
            print(f"  ventana {iteration} fallo: {exc}")

    backtest_df = pd.DataFrame(backtest_rows)

    # -- Diagnostico de residuos (sobre SARIMAX de train) ------------------------
    residuals = sarimax_result.resid
    lb = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    stat_sw, p_sw = shapiro(residuals)
    residual_diag = {
        "mean": float(residuals.mean()),
        "ljung_box": [
            {"lag": int(idx), "stat": float(row["lb_stat"]), "p_value": float(row["lb_pvalue"])}
            for idx, row in lb.iterrows()
        ],
        "shapiro": {"stat": float(stat_sw), "p_value": float(p_sw)},
    }
    residuals_df = pd.DataFrame({
        "date": train["date"].values,
        "residual": residuals.values,
    })

    # -- Modelo final (entrenado con TODOS los datos) -----------------------------
    print("Entrenando modelo final con todos los datos...")
    final_model = SARIMAX(
        df["log_sales"], exog=df[EXOG_COLS], order=SARIMA_ORDER,
        seasonal_order=SEASONAL_ORDER, enforce_stationarity=False, enforce_invertibility=False,
    )
    final_result = final_model.fit(disp=False)

    # -- Guardar artefactos -----------------------------------------------------
    print("Guardando artefactos...")
    # Se guardan solo los parametros ya optimizados (no el objeto results completo):
    # para un SARIMAX estacional (s=52) el pickle de resultados pesa cientos de MB
    # por las matrices del filtro de Kalman. La app reconstruye el modelo con
    # `.filter(params)`, que aplica los parametros ya ajustados sin reoptimizar.
    with open(ARTIFACTS / "final_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "names": list(final_result.params.index),
            "values": [float(v) for v in final_result.params.values],
        }, f, ensure_ascii=False, indent=2)

    with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(ARTIFACTS / "coefficients.json", "w", encoding="utf-8") as f:
        json.dump(coefficients, f, ensure_ascii=False, indent=2)
    with open(ARTIFACTS / "adf_results.json", "w", encoding="utf-8") as f:
        json.dump(adf_results, f, ensure_ascii=False, indent=2)
    with open(ARTIFACTS / "residual_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(residual_diag, f, ensure_ascii=False, indent=2)
    with open(ARTIFACTS / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "exog_cols": EXOG_COLS, "order": SARIMA_ORDER, "seasonal_order": SEASONAL_ORDER,
            "test_weeks": TEST_WEEKS, "aic": float(final_result.aic), "bic": float(final_result.bic),
            "last_date": df["date"].max().strftime("%Y-%m-%d"),
            "n_obs": int(len(df)),
        }, f, ensure_ascii=False, indent=2)

    forecast_df.to_csv(ARTIFACTS / "test_forecast.csv", index=False)
    backtest_df.to_csv(ARTIFACTS / "backtest.csv", index=False)
    residuals_df.to_csv(ARTIFACTS / "residuals.csv", index=False)

    print("\nListo. Resumen:")
    print(f"  SARIMA  MAPE={sarima_mape:.2f}%  MAE={sarima_mae:,.0f} EUR  RMSE={sarima_rmse:,.0f} EUR")
    print(f"  SARIMAX MAPE={sarimax_mape:.2f}%  MAE={sarimax_mae:,.0f} EUR  RMSE={sarimax_rmse:,.0f} EUR")
    print(f"  Mejora MAPE: {metrics['improvement_pct']['mape']:.1f}%")


if __name__ == "__main__":
    main()
