"""Tahap 5: metrik dari file prediksi tersimpan, ablasi oseanografi, asosiasi deskriptif, tabel profil lama."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src import evaluation as E  # noqa: E402
from src.audit import analytical_path  # noqa: E402
from src.common import Timer, duck, load_config, log_step, run_dir  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    data, tables = run_dir(cfg, "data"), run_dir(cfg, "tables")
    checks = {}
    with Timer() as t:
        # ---------------- metrik dihitung ulang dari CSV prediksi (bukan dari memori model)
        pred = pd.read_csv(tables / "T06_predictions.csv", parse_dates=["target_month", "forecast_origin"])
        pred = pred.rename(columns={"y_true": "n_det"})
        feat = pd.read_parquet(data / "forecast_features.parquet")
        denom = E.mase_denominators(feat, cfg["period"]["main_start"], cfg["forecast"]["train_target_end"] + "-01")
        denom.rename(columns={"mean": "mase_denominator", "size": "n_months"}).to_csv(tables / "T07c_mase_denominators.csv")
        m = E.metrics_by_wpp(pred, denom)
        m.to_csv(tables / "T07_metrics_by_wpp.csv", index=False)
        s = E.summary_by_model(m)
        s.to_csv(tables / "T07b_model_summary.csv", index=False)
        checks["n_test_rows_per_model"] = pred[pred["split"] == "test"].groupby("model").size().to_dict()
        checks["mase_denominator_defined_all_wpp"] = bool(denom["mean"].notna().all())

        ab = pd.concat([E.paired_delta(pred, split=sp).assign(split=sp) for sp in ["val", "test"]])
        ab.to_csv(tables / "T08_ocean_ablation_M2_vs_M3.csv", index=False)

        # ---------------- asosiasi deskriptif: anomali musiman (kurangi rata-rata bulan kalender 2013-2025)
        panel = pd.read_parquet(data / "wpp_month_panel.parquet")
        ocean = pd.read_parquet(data / "ocean_wpp_month.parquet")
        a = panel[(panel["month"] >= "2013-01-01") & (panel["month"] <= "2025-12-01")].merge(ocean, on=["wpp", "month"], how="left")
        assert len(a) == 11 * 156
        a["log_det"] = np.log1p(a["n_det"])
        for c in ["log_det", "sst", "log10_chl"]:
            a[f"{c}_anom"] = a[c] - a.groupby(["wpp", "moy"])[c].transform("mean")
        a = a.sort_values(["wpp", "month"])
        for c in ["sst_anom", "log10_chl_anom"]:
            a[f"{c}_prev"] = a.groupby("wpp")[c].shift(1)
        a.to_parquet(data / "ocean_association_frame.parquet", index=False)
        rows = []
        for w, g in a.groupby("wpp"):
            for var in ["sst", "log10_chl"]:
                for kind, x, y in [("raw_same_month", g[var], g["log_det"]),
                                   ("anomaly_same_month", g[f"{var}_anom"], g["log_det_anom"]),
                                   ("anomaly_ocean_prev_month", g[f"{var}_anom_prev"], g["log_det_anom"])]:
                    ok = x.notna() & y.notna()
                    rho = spearmanr(x[ok], y[ok]).statistic
                    rows.append({"wpp": w, "variable": var, "relation": kind, "spearman_rho": round(float(rho), 3),
                                 "n_months": int(ok.sum()), "period": "2013-01..2025-12",
                                 "note": "deskriptif; tanpa p-value (deret berautokorelasi)"})
        pd.DataFrame(rows).to_csv(tables / "T09_ocean_association_spearman.csv", index=False)

        # ---------------- tabel analisis lama yang dihitung ulang dari view analitis
        con = duck(cfg)
        ap = analytical_path(cfg).as_posix()
        prof = con.sql(f"""
          SELECT wpp, count(*) n_det, median(rad_dnb) rad_median,
                 avg((rad_dnb < 10)::INT) * 100 pct_low, avg((rad_dnb >= 10 AND rad_dnb <= 400)::INT) * 100 pct_medium,
                 avg((rad_dnb > 400)::INT) * 100 pct_high
          FROM read_parquet('{ap}') GROUP BY 1 ORDER BY 1""").df()
        prof["share_pct"] = 100 * prof["n_det"] / prof["n_det"].sum()
        prof["period"] = "2012-04-01..2026-05-31 (semua data, deskriptif)"
        prof.to_csv(tables / "S09_wpp_radiance_profile.csv", index=False)
        tot = con.sql(f"""SELECT count(*) n_det, median(rad_dnb) rad_median, avg((rad_dnb<10)::INT)*100 pct_low,
                          avg((rad_dnb>=10 AND rad_dnb<=400)::INT)*100 pct_medium, avg((rad_dnb>400)::INT)*100 pct_high,
                          sum((rad_dnb<=0)::INT) n_nonpositive FROM read_parquet('{ap}')""").df()
        tot.to_csv(tables / "S01_radiance_overall.csv", index=False)
        edges = np.logspace(-1, 5.5, 66)
        rad = con.sql(f"SELECT rad_dnb FROM read_parquet('{ap}') WHERE rad_dnb > 0").df()["rad_dnb"].to_numpy()
        h, _ = np.histogram(rad, bins=edges)
        pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": h}).to_csv(tables / "S01_radiance_hist.csv", index=False)
        del rad
        q = con.sql(f"""SELECT wpp, quantile_cont(rad_dnb, [0.05,0.25,0.5,0.75,0.95]) q, count(*) n
                        FROM read_parquet('{ap}') GROUP BY 1 ORDER BY 1""").df()
        q[["p05", "p25", "p50", "p75", "p95"]] = pd.DataFrame(q["q"].tolist(), index=q.index)
        q.drop(columns="q").to_csv(tables / "S02_radiance_quantiles_by_wpp.csv", index=False)

        ann = panel[(panel["year"] >= 2013) & (panel["year"] <= 2025)].groupby(["wpp", "year"])["n_det"].sum().reset_index()
        ann.to_csv(tables / "S07_annual_counts_2013_2025.csv", index=False)
        cv = ann.groupby("wpp")["n_det"].agg(mean="mean", sd="std", n_years="size").reset_index()
        cv["cv_pct"] = 100 * cv["sd"] / cv["mean"]
        cv["period"] = "2013..2025 (tahun penuh; 2012 dan 2026 parsial dikeluarkan)"
        cv.to_csv(tables / "S10_interannual_cv.csv", index=False)

        inv = pd.DataFrame([
            ("Kepadatan spasial (KDE)", "dihitung ulang (bentuk lain)", "Diganti histogram spasial grid 1 km EPSG:6933 (jumlah per sel), bukan KDE; dipakai di F03."),
            ("Distribusi radiansi", "dihitung ulang", "S01 histogram log dari view analitis saat ini; median dihitung ulang."),
            ("Profil WPP & kelas radiansi", "dihitung ulang; lampiran", "S09; kelas <10, 10-400, >400 nW/cm2/sr disebut kelas radiansi, bukan ukuran kapal. Angka per WPP berbeda dari Tabel 5 paper (lihat data_issues.md)."),
            ("Tren jangka panjang (MK/Sen)", "tidak dijalankan ulang; lampiran", "Hanya deret tahunan S07 ditampilkan. Uji MK tidak diulang karena n=13 tahun, autokorelasi dan 11 uji berganda belum ditangani; angka paper tidak disalin."),
            ("Variasi antartahun (CV)", "dihitung ulang; lampiran", "S10 memakai 2013-2025; cutoff CV paper bukan standar universal."),
            ("Pola musiman (STL)", "dihitung ulang (bentuk lain)", "F04 memakai indeks musiman median per bulan kalender; STL tidak diulang."),
            ("Forecasting (ETS/SARIMA)", "diganti protokol baru", "M0/M1/M2/M3 rolling 1 langkah 2025; angka ETS/SARIMA paper (horizon 12 bulan) tidak sebanding."),
        ], columns=["analysis_branch", "status", "reason"])
        inv.to_csv(tables / "legacy_analysis_inventory.csv", index=False)
    log_step(cfg, "05_evaluate", {"seconds": t.seconds, "checks": checks})
    print(s.to_string())
    print(ab[ab["split"] == "test"].round(2).to_string())
    print(json.dumps(checks, indent=1))
