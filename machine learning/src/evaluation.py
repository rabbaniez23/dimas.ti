"""Metrik regresi per WPP, dihitung ulang dari file prediksi tersimpan."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mase_denominators(panel: pd.DataFrame, start: str, end: str) -> pd.Series:
    """mean |y_t - y_{t-12}| pada bulan target latih referensi [start, end] (hanya t yang punya y_{t-12}).
    Sama untuk semua model; data validasi/uji tidak masuk."""
    p = panel[(panel["target_month"] >= start) & (panel["target_month"] <= end)].dropna(subset=["lag_12", "n_det"])
    d = (p["n_det"] - p["lag_12"]).abs().groupby(p["wpp"]).agg(["mean", "size"])
    d.loc[d["mean"] == 0, "mean"] = np.nan  # undefined, bukan epsilon
    return d


def metrics_by_wpp(pred: pd.DataFrame, denom: pd.DataFrame) -> pd.DataFrame:
    p = pred.dropna(subset=["y_pred"]).assign(err=lambda d: d["y_pred"] - d["n_det"])
    g = p.groupby(["split", "model", "wpp"])
    m = g.agg(n_months=("err", "size"), MAE=("err", lambda e: e.abs().mean()),
              RMSE=("err", lambda e: np.sqrt((e ** 2).mean())), bias_mean_error=("err", "mean"),
              mean_actual=("n_det", "mean")).reset_index()
    m["MASE"] = m["MAE"] / m["wpp"].map(denom["mean"])
    m["MASE_denominator"] = m["wpp"].map(denom["mean"])
    m["MASE_denominator_n"] = m["wpp"].map(denom["size"])
    expected = pred.groupby(["split", "model", "wpp"]).size().rename("n_expected").reset_index()
    return m.merge(expected, on=["split", "model", "wpp"], how="right")


def summary_by_model(m: pd.DataFrame) -> pd.DataFrame:
    return (m.groupby(["split", "model"])
              .agg(n_wpp=("wpp", "nunique"), n_wpp_months=("n_months", "sum"), macro_MASE=("MASE", "mean"),
                   median_MASE=("MASE", "median"), macro_MAE=("MAE", "mean"), macro_RMSE=("RMSE", "mean"))
              .reset_index())


def paired_delta(pred: pd.DataFrame, a: str = "M2_RF_History", b: str = "M3_RF_Ocean", split: str = "test") -> pd.DataFrame:
    """delta MAE = MAE(a) - MAE(b) per WPP pada pasangan WPP-bulan yang sama; positif = b (oseanografi) lebih baik."""
    p = pred[pred["split"] == split]
    wa = p[p["model"] == a].set_index(["wpp", "target_month"])
    wb = p[p["model"] == b].set_index(["wpp", "target_month"])
    idx = wa.index.intersection(wb.index)
    ea = (wa.loc[idx, "y_pred"] - wa.loc[idx, "n_det"]).abs()
    eb = (wb.loc[idx, "y_pred"] - wb.loc[idx, "n_det"]).abs()
    d = pd.DataFrame({"abs_err_a": ea, "abs_err_b": eb}).reset_index()
    out = d.groupby("wpp").agg(n_months=("abs_err_a", "size"), MAE_M2=("abs_err_a", "mean"), MAE_M3=("abs_err_b", "mean"),
                               months_M3_better=("abs_err_a", lambda s: int((s > d.loc[s.index, "abs_err_b"]).sum())))
    out["delta_MAE_M2_minus_M3"] = out["MAE_M2"] - out["MAE_M3"]
    out["delta_rel_pct_of_M2"] = 100 * out["delta_MAE_M2_minus_M3"] / out["MAE_M2"]
    return out.reset_index()
