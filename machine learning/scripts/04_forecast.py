"""Tahap 4: fitur kausal, validasi 2024 (pemilihan hyperparameter), uji akhir 2025 (parameter beku)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from src import evaluation as E  # noqa: E402
from src import forecast as F  # noqa: E402
from src.common import Timer, load_config, log_step, run_dir  # noqa: E402
from src.features import add_history_features  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    FC = cfg["forecast"]
    data, tables, models = run_dir(cfg, "data"), run_dir(cfg, "tables"), run_dir(cfg, "models")
    checks = {}
    with Timer() as t:
        panel = pd.read_parquet(data / "wpp_month_panel.parquet")
        panel = panel[panel["month"] <= cfg["period"]["main_end"]]  # 2026 parsial tidak dipakai
        feat = add_history_features(panel, FC["lags"], FC["rolling_window"])
        ocean = pd.read_parquet(data / "ocean_wpp_month.parquet")
        n0 = len(feat)
        feat = feat.merge(F.ocean_lagged(ocean), on=["wpp", "target_month"], how="left", validate="one_to_one")
        checks["ocean_join_no_duplication"] = len(feat) == n0
        oh = F.wpp_onehot(feat, cfg["wpp_codes"])
        feat = pd.concat([feat, oh], axis=1)
        wcols = list(oh.columns)
        feats = {"M2_RF_History": F.HIST_FEATURES + wcols,
                 "M3_RF_Ocean": F.HIST_FEATURES + F.OCEAN_FEATURES + wcols}

        # ---------------- pemeriksaan kebocoran
        f2 = feat.sort_values(["wpp", "target_month"])
        prev = f2.groupby("wpp")["n_det"].shift(1)
        checks["lag1_equals_prev_month_same_wpp"] = bool(np.allclose(f2["lag_1"].dropna(), prev.dropna()))
        checks["first_row_each_wpp_lag1_nan"] = bool(f2.groupby("wpp").head(1)["lag_1"].isna().all())
        checks["ocean_source_before_target"] = bool((feat["source_month"].dropna() < feat.loc[feat["source_month"].notna(), "target_month"]).all())
        r3 = f2.groupby("wpp")["n_det"].transform(lambda s: s.shift(1).rolling(3).mean())
        checks["roll3_uses_only_past"] = bool(np.allclose(f2["roll3_mean"].dropna(), r3.dropna()))

        val_m = list(pd.date_range(FC["val"][0] + "-01", FC["val"][1] + "-01", freq="MS"))
        test_m = list(pd.date_range(FC["test"][0] + "-01", FC["test"][1] + "-01", freq="MS"))
        train_start = cfg["period"]["main_start"]
        train_end = FC["train_target_end"] + "-01"
        checks["splits_disjoint"] = len(set(val_m) & set(test_m)) == 0 and max(val_m) < min(test_m) \
            and pd.Timestamp(train_end) < min(val_m)
        tr_rows = feat[(feat["target_month"] >= train_start) & (feat["target_month"] <= train_end)]
        checks["train_rows_dropped_missing_features"] = int(tr_rows[feats["M3_RF_Ocean"]].isna().any(axis=1).sum())
        feat.to_parquet(data / "forecast_features.parquet", index=False)

        # ---------------- validasi: pilih hyperparameter RF (anggaran sama untuk M2 & M3)
        val_rows, chosen = [], {}
        denom = E.mase_denominators(feat, train_start, train_end)
        for model, fs in feats.items():
            res = []
            for i, params in enumerate(F.rf_grid(cfg)):
                pr = F.rolling_rf(feat, val_m, fs, params, cfg, train_start).assign(split="val", model=model)
                mm = E.metrics_by_wpp(pr, denom)
                res.append({"model": model, "candidate": i, **{k: str(v) for k, v in params.items()},
                            "val_macro_MASE": mm["MASE"].mean(), "val_macro_MAE": mm["MAE"].mean(), "_pred": pr,
                            "_params": params})
                print(f"{model} cand {i} {params}: val macro-MASE={mm['MASE'].mean():.4f}", flush=True)
            best = min(res, key=lambda r: (r["val_macro_MASE"], r["candidate"]))
            chosen[model] = best["_params"]
            val_rows.append(best["_pred"])
            pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in res]).assign(
                selected=lambda d: d["candidate"] == best["candidate"]).to_csv(
                tables / f"T05b_rf_validation_{model}.csv", index=False)

        # ETS: spesifikasi dipilih dengan AICc pada target <= 2023-12, lalu dibekukan
        ets_specs, ets_sel = {}, []
        for w in cfg["wpp_codes"]:
            s = feat[(feat["wpp"] == w) & (feat["target_month"] <= train_end)].set_index("target_month")["n_det"]
            s.index = pd.DatetimeIndex(s.index, freq="MS")
            spec, rows = F.select_ets_spec(s)
            ets_specs[w] = spec
            ets_sel += [{"wpp": w, **r, "selected": (r["trend"] == spec["trend"])} for r in rows]
        pd.DataFrame(ets_sel).to_csv(tables / "T05c_ets_spec_selection.csv", index=False)

        hist_start = "2012-04-01"
        preds = []
        for split, months in [("val", val_m), ("test", test_m)]:
            preds.append(F.seasonal_naive(feat, months).assign(split=split, model="M0_SeasonalNaive"))
            preds.append(pd.concat([F.rolling_ets(feat, w, months, ets_specs[w], hist_start) for w in cfg["wpp_codes"]])
                         .assign(split=split, model="M1_ETS"))
        preds += val_rows
        for model, fs in feats.items():
            preds.append(F.rolling_rf(feat, test_m, fs, chosen[model], cfg, train_start).assign(split="test", model=model))
        pred = pd.concat(preds, ignore_index=True)
        pred["fit_status"] = pred.get("fit_status", pd.Series(index=pred.index)).fillna("ok")
        pred["model_params"] = pred["model"].map({**{k: json.dumps(v) for k, v in chosen.items()},
                                                  "M0_SeasonalNaive": "y(m-12)", "M1_ETS": "per-WPP AICc spec, log1p"})
        pred = pred.rename(columns={"n_det": "y_true"})
        pred.to_csv(tables / "T06_predictions.csv", index=False)
        pred.to_parquet(data / "predictions.parquet", index=False)

        # cakupan: semua model dinilai pada pasangan WPP-bulan yang sama
        keys = pred.dropna(subset=["y_pred"]).groupby(["split", "model"]).apply(
            lambda d: frozenset(zip(d["wpp"], d["target_month"])), include_groups=False)
        checks["same_wpp_month_pairs_across_models"] = {s: len(set(keys[s].tolist())) == 1 for s in ["val", "test"]}
        checks["n_failed_predictions"] = int(pred["y_pred"].isna().sum())

        # ---------------- permutation importance (validasi 2024), model latih s.d. 2023-12
        imp_rows = []
        usable = feat.dropna(subset=feats["M3_RF_Ocean"])
        tr = usable[(usable["target_month"] >= train_start) & (usable["target_month"] <= train_end)]
        va = feat[feat["target_month"].isin(val_m)]
        for model, fs in feats.items():
            _, m = F.fit_predict_rf(tr, va, fs, chosen[model], cfg)
            pi = permutation_importance(m, va[fs].to_numpy(), np.log1p(va["n_det"].to_numpy()),
                                        scoring="neg_mean_absolute_error", n_repeats=30, random_state=42, n_jobs=1)
            for f, mu, sd in zip(fs, pi.importances_mean, pi.importances_std):
                imp_rows.append({"model": model, "feature": f, "importance_mean": mu, "importance_std": sd})
        pd.DataFrame(imp_rows).to_csv(tables / "S08_permutation_importance_val.csv", index=False)

        (models / "forecast_selected_params.json").write_text(json.dumps(
            {"rf": chosen, "ets": {str(k): v for k, v in ets_specs.items()},
             "features": feats, "target_transform": "log1p (RF dan ETS), prediksi dikembalikan dengan expm1"},
            indent=2), encoding="utf-8")
        ft = pd.DataFrame([{"model": m, "feature": f} for m, fs in feats.items() for f in fs])
        ft["source_month_rule"] = ft["feature"].map(lambda f: "m-1" if f in ("lag_1", "sst_lag1", "log10_chl_lag1") else
                                                     "m-2" if f == "lag_2" else "m-3" if f == "lag_3" else "m-12" if f == "lag_12"
                                                     else "m-3..m-1" if f.startswith("roll3") else
                                                     "m-1 minus m-13" if "yoy" in f else "kalender bulan target" if f.startswith("moy")
                                                     else "kategori tetap")
        ft.to_csv(tables / "T05_features_and_splits.csv", index=False)
        splits = pd.DataFrame([
            {"stage": "train_targets", "start": train_start[:7], "end": FC["train_target_end"],
             "note": "baris tanpa lag_12 lengkap (2013-01..2013-03 tidak, karena data mulai 2012-04: lag_12 2013-01..03 kosong) dibuang dari latih"},
            {"stage": "validation_rolling", "start": FC["val"][0], "end": FC["val"][1], "note": "pemilihan hyperparameter RF; 12 origin bulanan"},
            {"stage": "test_rolling", "start": FC["test"][0], "end": FC["test"][1], "note": "parameter beku; label bulan uji yang sudah lewat menjadi riwayat"},
            {"stage": "warmup_history", "start": "2012-04", "end": "2012-12", "note": "hanya sumber lag; bukan tahun penuh"}])
        splits.to_csv(tables / "T05a_time_splits.csv", index=False)
    log_step(cfg, "04_forecast", {"seconds": t.seconds, "checks": checks, "rf_selected": chosen})
    print(json.dumps(checks, indent=1, default=str))
    print(chosen)
