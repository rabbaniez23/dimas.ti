"""Tahap 3: HDBSCAN per WPP atas sel 1 km aktif berulang (data latih 2013-2023) -> CFA referensi beku."""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import clustering as C  # noqa: E402
from src.common import Timer, duck, load_config, log_step, run_dir  # noqa: E402


def cluster_wpp(cfg, cells_all, wpp, min_years, size, grid_params, cache_dir, tag):
    cells = (cells_all[(cells_all["wpp"] == wpp) & (cells_all["n_years"] >= min_years)]
             .sort_values(["ix", "iy"]).reset_index(drop=True))
    runs = {}
    for mcs, ms in grid_params:
        f = cache_dir / f"{tag}_w{wpp}_mcs{mcs}_ms{ms}.pkl"
        if f.exists():
            runs[(mcs, ms)] = pickle.loads(f.read_bytes())
        else:
            r = C.run_one(cells, mcs, ms, cfg["hdbscan"]["cluster_selection_method"])
            f.write_bytes(pickle.dumps(r))
            runs[(mcs, ms)] = r
        print(f"  [{tag}] WPP {wpp} mcs={mcs} ms={ms}: k={runs[(mcs, ms)]['n_clusters']} "
              f"noise={runs[(mcs, ms)]['noise_frac']:.3f} ({runs[(mcs, ms)]['seconds']}s)", flush=True)
    return cells, runs


if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    H = cfg["hdbscan"]
    data, models, tables = run_dir(cfg, "data"), run_dir(cfg, "models"), run_dir(cfg, "tables")
    cache = run_dir(cfg, "models/hdbscan_runs")
    size = cfg["spatial"]["grid_size_m"]
    crs = cfg["spatial"]["grid_crs"]
    grid_params = [(a, b) for a in H["min_cluster_size"] for b in H["min_samples"]]
    train = pd.read_parquet(data / f"grid_train_{size}m.parquet")
    assert pd.to_datetime(train["last_date"]).max() <= pd.Timestamp(H["cfa_train_end"]), "data latih CFA melewati batas"

    wpps = [H["pilot_wpp"]] + [w for w in cfg["wpp_codes"] if w != H["pilot_wpp"]]
    if cfg["resources"]["mode"] == "pilot":
        wpps = [H["pilot_wpp"]]
    summaries, cats, geoms, cellmaps, done = [], [], [], [], []
    with Timer() as t:
        for w in wpps:
            cells, runs = cluster_wpp(cfg, train, w, H["min_distinct_years"], size, grid_params, cache, "main")
            s = C.summarize_runs(w, cells, runs)
            mcs, ms = C.choose_params(s)
            s["selected"] = (s["min_cluster_size"] == mcs) & (s["min_samples"] == ms)
            summaries.append(s)
            r = runs[(mcs, ms)]
            cat, g, cm = C.build_cfa(w, cells, r["labels"], r["persistence"], size, crs)
            cat["min_cluster_size"], cat["min_samples"] = mcs, ms
            cats.append(cat); geoms.append(g); cellmaps.append(cm); done.append(w)
            print(f"WPP {w}: terpilih mcs={mcs} ms={ms}, {len(cat)} CFA, luas {cat['area_km2'].sum():,.0f} km2", flush=True)

        summary = pd.concat(summaries, ignore_index=True)
        catalog = pd.concat(cats, ignore_index=True)
        cellmap = pd.concat(cellmaps, ignore_index=True)
        gdf = gpd.GeoDataFrame(pd.concat(geoms, ignore_index=True), crs=crs)
        summary.to_csv(tables / "T04a_hdbscan_param_runs.csv", index=False)
        catalog["period_train"] = f"{H['cfa_train_start']}..{H['cfa_train_end']}"
        catalog["grid_size_m"] = size
        catalog["selection_rule"] = f"n_years>={H['min_distinct_years']} (2013-2023)"
        catalog.to_csv(tables / "cfa_catalog.csv", index=False)
        cellmap.to_parquet(data / "cfa_cells.parquet", index=False)
        gdf.to_parquet(data / "cfa_reference_6933.parquet")
        out = gdf[["cfa_id", "wpp", "n_cells", "area_km2", "n_det_train", "persistence", "geometry"]].to_crs("EPSG:4326")
        out.to_file(data / "cfa_reference.geojson", driver="GeoJSON")

        # ---- hitungan bulanan dalam CFA beku (semua bulan; label periode)
        con = duck(cfg)
        con.register("cm", cellmap[["wpp", "ix", "iy", "cfa_id"]])
        mc = con.sql(f"""
          SELECT cm.cfa_id, cm.wpp, make_date(p.year::BIGINT, p.month::BIGINT, 1) AS "month", count(*) AS n_det
          FROM read_parquet('{(data / 'points_grid.parquet').as_posix()}') p
          JOIN cm ON p.wpp = cm.wpp AND p.ix_{size} = cm.ix AND p.iy_{size} = cm.iy
          GROUP BY ALL""").df()
        mc["month"] = pd.to_datetime(mc["month"])
        months = pd.date_range("2012-04-01", "2026-05-01", freq="MS")
        full = pd.MultiIndex.from_product([catalog["cfa_id"], months], names=["cfa_id", "month"]).to_frame(index=False)
        full = full.merge(catalog[["cfa_id", "wpp"]], on="cfa_id").merge(mc, on=["cfa_id", "wpp", "month"], how="left")
        full["n_det"] = full["n_det"].fillna(0).astype(int)  # setiap bulan memiliki rekaman input (lihat audit)
        full["period"] = np.select([full["month"] < "2013-01-01", full["month"] <= "2023-12-01",
                                    full["month"] <= "2024-12-01", full["month"] <= "2025-12-01"],
                                   ["warmup", "train", "val", "test"], "partial_2026")
        full.to_parquet(data / "cfa_monthly_counts.parquet", index=False)
        tr_check = full[full["period"] == "train"].groupby("cfa_id")["n_det"].sum()
        recon = bool((tr_check.reindex(catalog["cfa_id"]).to_numpy() == catalog["n_det_train"].to_numpy()).all())

        # ---- sensitivitas: seleksi >=3 tahun (1 km) dan grid 2 km (>=2 tahun), parameter terpilih per WPP
        sens = []
        sel = summary[summary["selected"]].set_index("wpp")
        train2 = pd.read_parquet(data / "grid_train_2000m.parquet")
        main_area = catalog.groupby("wpp")["area_km2"].sum()
        main_union = gdf.dissolve("wpp")
        for w in done:
            mcs, ms = int(sel.loc[w, "min_cluster_size"]), int(sel.loc[w, "min_samples"])
            for tag, src, yrs, sz in [("ge3years_1km", train, 3, size), ("ge2years_2km", train2, 2, 2000)]:
                cells, runs = cluster_wpp(cfg, src, w, yrs, sz, [(mcs, ms)], cache, tag)
                r = runs[(mcs, ms)]
                cat2, g2, _ = C.build_cfa(w, cells, r["labels"], r["persistence"], sz, crs)
                a2 = float(cat2["area_km2"].sum()) if len(cat2) else 0.0
                inter = 0.0
                if len(g2):
                    u2 = g2.union_all()
                    inter = main_union.loc[w, "geometry"].intersection(u2).area / 1e6
                union_area = main_area.get(w, 0) + a2 - inter
                sens.append({"wpp": w, "variant": tag, "min_cluster_size": mcs, "min_samples": ms,
                             "n_cells_input": len(cells), "n_clusters": r["n_clusters"],
                             "noise_frac_cells": round(r["noise_frac"], 4), "cfa_area_km2": round(a2, 1),
                             "main_cfa_area_km2": round(float(main_area.get(w, 0)), 1),
                             "area_jaccard_vs_main": round(inter / union_area, 4) if union_area > 0 else np.nan})
        pd.DataFrame(sens).to_csv(tables / "T04b_cfa_sensitivity.csv", index=False)

    checks = {"wpp_done": done, "cfa_monthly_train_reconciles_catalog": recon,
              "cfa_train_max_date": str(pd.to_datetime(train["last_date"]).max().date()),
              "n_cfa": int(len(catalog)), "total_cfa_area_km2": float(catalog["area_km2"].sum()),
              "geometry_crs": str(gdf.crs)}
    log_step(cfg, "03_cfa", {"seconds": t.seconds, "checks": checks})
    print(json.dumps(checks, indent=1))
