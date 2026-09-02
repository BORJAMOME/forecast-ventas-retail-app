"""Genera predicciones de escenarios en vivo a partir del modelo SARIMAX
ya entrenado (final_model.pickle). Misma lógica que `predict_scenario`
del notebook original."""
import numpy as np
import pandas as pd

from utils.data_loader import EXOG_COLS


def build_future_frame(last_date, horizon_weeks: int) -> pd.DataFrame:
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon_weeks, freq="W-MON")
    frame = pd.DataFrame({"date": future_dates})
    for col in EXOG_COLS:
        frame[col] = 0.0
    return frame


def predict_scenario(final_result, scenario_df: pd.DataFrame) -> pd.DataFrame:
    horizon = len(scenario_df)
    pred = final_result.get_forecast(steps=horizon, exog=scenario_df[EXOG_COLS])
    mean_log = pred.predicted_mean
    ci_log = pred.conf_int(alpha=0.05)

    result = scenario_df[["date"]].copy()
    result["forecast"] = np.exp(mean_log).values
    result["lower95"] = np.exp(ci_log.iloc[:, 0]).values
    result["upper95"] = np.exp(ci_log.iloc[:, 1]).values
    return result


def scenario_from_inputs(last_date, horizon_weeks: int, promotion: float, strike: bool, war: bool,
                          logistics: bool, other_incident: bool) -> pd.DataFrame:
    frame = build_future_frame(last_date, horizon_weeks)
    frame["promotion"] = promotion
    frame["strike"] = int(strike)
    frame["war"] = int(war)
    frame["logistics"] = int(logistics)
    frame["other_incident"] = int(other_incident)
    return frame
