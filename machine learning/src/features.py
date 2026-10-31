"""Dataset B (panel WPP-bulan) dan fitur peramalan kausal."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_wpp_month(cfg, con, analytical, monthly_avail: pd.DataFrame) -> pd.DataFrame:
    """Satu baris = satu WPP x satu bulan kalender dalam rentang data.

    count_status:
      observed_positive  : ada >=1 deteksi QF1 di WPP pada bulan tsb
      zero_no_detection  : input memuat rekaman (QF apa pun, seluruh bbox) pada bulan tsb tetapi tidak ada deteksi
                           QF1 di WPP -> dianggap nol teramati (dengan catatan: tanpa manifest cakupan satelit)
      not_available      : bulan di luar rentang data atau tanpa rekaman apa pun -> NaN, tidak diisi nol
    """
    cnt = con.sql(f"""
      SELECT wpp, make_date(year::BIGINT, month::BIGINT, 1) AS "month", count(*) AS n_det,
             count(DISTINCT date_utc) AS n_dates_det, median(rad_dnb) AS rad_median
      FROM read_parquet('{analytical.as_posix()}') GROUP BY ALL""").df()
    cnt["month"] = pd.to_datetime(cnt["month"])
    months = pd.to_datetime(monthly_avail["month"])
    idx = pd.MultiIndex.from_product([cfg["wpp_codes"], months], names=["wpp", "month"])
    panel = pd.DataFrame(index=idx).reset_index().merge(cnt, on=["wpp", "month"], how="left")
    av = monthly_avail.set_index(pd.to_datetime(monthly_avail["month"]))
    has_any = panel["month"].map(av["n_all_qf"].fillna(0) > 0).fillna(False)
    panel["n_dates_any_record"] = panel["month"].map(av["n_dates_any_record"])
    panel["count_status"] = np.where(panel["n_det"].notna(), "observed_positive",
                                     np.where(has_any, "zero_no_detection", "not_available"))
    panel.loc[panel["count_status"] == "zero_no_detection", "n_det"] = 0
    panel.loc[panel["count_status"] == "zero_no_detection", "n_dates_det"] = 0
    panel["year"] = panel["month"].dt.year
    panel["moy"] = panel["month"].dt.month
    panel["days_in_month"] = panel["month"].dt.days_in_month
    return panel.sort_values(["wpp", "month"]).reset_index(drop=True)


def add_history_features(panel: pd.DataFrame, lags=(1, 2, 3, 12), win: int = 3) -> pd.DataFrame:
    """Fitur kausal per WPP untuk target bulan m: semua dari bulan <= m-1.

    Sortir per WPP & waktu; shift dilakukan SEBELUM rolling; groupby mencegah rolling lintas WPP.
    Panel harus berupa indeks kalender reguler (tanpa bulan yang hilang di tengah).
    """
    df = panel.sort_values(["wpp", "month"]).copy()
    g = df.groupby("wpp", sort=False)["n_det"]
    for L in lags:
        df[f"lag_{L}"] = g.shift(L)
    shifted = g.shift(1)
    df[f"roll{win}_mean"] = shifted.groupby(df["wpp"]).transform(lambda s: s.rolling(win, min_periods=win).mean())
    df[f"roll{win}_std"] = shifted.groupby(df["wpp"]).transform(lambda s: s.rolling(win, min_periods=win).std())
    df["moy_sin"] = np.sin(2 * np.pi * df["moy"] / 12)
    df["moy_cos"] = np.cos(2 * np.pi * df["moy"] / 12)
    df["target_month"] = df["month"]
    df["forecast_origin"] = df["month"] - pd.offsets.MonthEnd(1)  # akhir bulan m-1
    return df


def add_ocean_features(df: pd.DataFrame, ocean: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Gabungkan SST/klorofil bulan m-lag (nilai bulanan aktual per WPP)."""
    o = ocean.copy()
    o["target_month"] = o["month"] + pd.DateOffset(months=lag)
    o = o.drop(columns=["month"]).rename(columns={c: f"{c}_lag{lag}" for c in o.columns if c not in ("wpp", "target_month")})
    n0 = len(df)
    out = df.merge(o, on=["wpp", "target_month"], how="left", validate="one_to_one")
    assert len(out) == n0, "join oseanografi menggandakan baris"
    return out
