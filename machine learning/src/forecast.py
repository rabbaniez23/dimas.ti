"""Model peramalan satu langkah ke depan (rolling/expanding origin) untuk panel WPP-bulan.

M0 Seasonal Naive : y_hat(m) = y(m-12)
M1 ETS            : ETS musiman (periode 12) per WPP pada log1p(y); spesifikasi dipilih dengan AICc pada data latih
                    (target <= 2023-12), parameter di-refit pada setiap forecast origin.
M2 RF-History     : RandomForest panel (gabungan WPP) dengan fitur riwayat, kalender, one-hot WPP.
M3 RF-Ocean       : M2 + SST dan log10-klorofil bulan m-1 serta perubahan tahunannya (m-1 vs m-13).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HIST_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_12", "roll3_mean", "roll3_std", "moy_sin", "moy_cos"]
OCEAN_FEATURES = ["sst_lag1", "log10_chl_lag1", "sst_yoy_lag1", "log10_chl_yoy_lag1"]


def wpp_onehot(df: pd.DataFrame, codes) -> pd.DataFrame:
    """One-hot dengan daftar kategori tetap (tidak dipelajari dari data, jadi tidak ada kebocoran)."""
    return pd.DataFrame({f"wpp_{c}": (df["wpp"] == c).astype(np.int8) for c in codes}, index=df.index)


def ocean_lagged(ocean: pd.DataFrame) -> pd.DataFrame:
    """Nilai oseanografi yang tersedia pada origin bulan m: bulan m-1, dan selisih m-1 terhadap m-13."""
    o = ocean.sort_values(["wpp", "month"]).copy()
    g = o.groupby("wpp", sort=False)
    o["sst_yoy"] = o["sst"] - g["sst"].shift(12)
    o["log10_chl_yoy"] = o["log10_chl"] - g["log10_chl"].shift(12)
    o["target_month"] = o["month"] + pd.DateOffset(months=1)
    o["source_month"] = o["month"]
    keep = {"sst": "sst_lag1", "log10_chl": "log10_chl_lag1", "sst_yoy": "sst_yoy_lag1",
            "log10_chl_yoy": "log10_chl_yoy_lag1"}
    return o.rename(columns=keep)[["wpp", "target_month", "source_month", *keep.values()]]


def rf_grid(cfg) -> list[dict]:
    g = cfg["forecast"]["rf"]["grid"]
    return [dict(max_depth=a, min_samples_leaf=b, max_features=c)
            for a, b, c in itertools.product(g["max_depth"], g["min_samples_leaf"], g["max_features"])]


def fit_predict_rf(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], params: dict, cfg) -> np.ndarray:
    rf_cfg = cfg["forecast"]["rf"]
    m = RandomForestRegressor(n_estimators=rf_cfg["n_estimators"], random_state=rf_cfg["random_state"],
                              n_jobs=rf_cfg["n_jobs"], **params)
    m.fit(train[feats].to_numpy(), np.log1p(train["n_det"].to_numpy()))
    return np.expm1(m.predict(test[feats].to_numpy())), m


def rolling_rf(panel: pd.DataFrame, months: list[pd.Timestamp], feats: list[str], params: dict, cfg,
               train_start: str) -> pd.DataFrame:
    """Untuk setiap bulan target m: latih dengan baris target_month < m (label sudah tersedia), prediksi bulan m."""
    out = []
    usable = panel.dropna(subset=feats + ["n_det"])
    for m in months:
        tr = usable[(usable["target_month"] >= train_start) & (usable["target_month"] < m)]
        te = panel[panel["target_month"] == m]
        assert te[feats].notna().all().all(), f"fitur kosong pada {m}"
        assert tr["target_month"].max() < m
        yhat, _ = fit_predict_rf(tr, te, feats, params, cfg)
        out.append(te[["wpp", "target_month", "forecast_origin", "n_det"]].assign(y_pred=yhat, n_train_rows=len(tr),
                                                                                  train_last_target=tr["target_month"].max()))
    return pd.concat(out, ignore_index=True)


def seasonal_naive(panel: pd.DataFrame, months) -> pd.DataFrame:
    te = panel[panel["target_month"].isin(months)]
    return te[["wpp", "target_month", "forecast_origin", "n_det"]].assign(y_pred=te["lag_12"].to_numpy())


# ------------------------------------------------------------------ ETS
ETS_SPECS = [dict(trend=None, damped_trend=False), dict(trend="add", damped_trend=True)]


def _fit_ets(y: pd.Series, spec: dict):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ETSModel(np.log1p(y.astype(float)), error="add", seasonal="add", seasonal_periods=12, **spec)
        return mod.fit(disp=False, maxiter=2000)


def select_ets_spec(y_train: pd.Series) -> tuple[dict, list[dict]]:
    rows = []
    for spec in ETS_SPECS:
        try:
            r = _fit_ets(y_train, spec)
            rows.append({**spec, "aicc": float(r.aicc), "ok": True})
        except Exception as e:  # noqa: BLE001
            rows.append({**spec, "aicc": np.inf, "ok": False, "error": str(e)[:200]})
    best = min(rows, key=lambda r: r["aicc"])
    return {"trend": best["trend"], "damped_trend": best["damped_trend"]}, rows


def rolling_ets(panel: pd.DataFrame, wpp: int, months, spec: dict, hist_start: str) -> pd.DataFrame:
    s = panel[(panel["wpp"] == wpp)].set_index("target_month")["n_det"]
    s.index = pd.DatetimeIndex(s.index, freq="MS")
    out = []
    for m in months:
        y = s[(s.index >= hist_start) & (s.index < m)]
        try:
            r = _fit_ets(y, spec)
            pred = float(np.expm1(r.forecast(1).iloc[0]))
            status = "ok"
        except Exception as e:  # noqa: BLE001
            pred, status = np.nan, f"failed: {str(e)[:120]}"
        out.append({"wpp": wpp, "target_month": m, "forecast_origin": m - pd.offsets.MonthEnd(1),
                    "n_det": s.get(m, np.nan), "y_pred": pred, "fit_status": status, "n_train_rows": len(y)})
    return pd.DataFrame(out)
