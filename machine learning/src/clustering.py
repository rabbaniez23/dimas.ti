"""HDBSCAN (paket scikit-learn-contrib `hdbscan`) atas pusat sel grid aktif berulang -> kandidat CFA.

- Input: sel grid dari data LATIH saja (2013-01..2023-12), seleksi n_years >= min_distinct_years.
- Metrik haversine, urutan [lat, lon] dalam radian. Waktu/oseanografi TIDAK masuk jarak.
- Frekuensi deteksi tidak dipakai sebagai bobot; hanya atribut katalog.
- Geometri CFA = union sel grid anggota (bukan convex hull); luas dihitung di EPSG:6933 (equal-area).
"""
from __future__ import annotations

import time
from itertools import combinations

import hdbscan
import numpy as np
import pandas as pd
import shapely
from sklearn.metrics import adjusted_rand_score

from .spatial import grid_cells_geoms

EARTH_R_KM = 6371.0088


def run_one(cells: pd.DataFrame, mcs: int, ms: int, method: str = "eom") -> dict:
    X = np.radians(cells[["center_lat", "center_lon"]].to_numpy())
    t0 = time.time()
    cl = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=int(ms), metric="haversine",
                         cluster_selection_method=method, core_dist_n_jobs=4)
    labels = cl.fit_predict(X)
    secs = time.time() - t0
    k = int(labels.max() + 1)
    pers = np.asarray(cl.cluster_persistence_) if k > 0 else np.array([])
    return {"labels": labels, "probabilities": cl.probabilities_, "persistence": pers, "seconds": round(secs, 1),
            "n_clusters": k, "noise_frac": float((labels < 0).mean())}


def summarize_runs(wpp: int, cells: pd.DataFrame, runs: dict) -> pd.DataFrame:
    """Metrik per kombinasi parameter + kestabilan (ARI rata-rata terhadap kombinasi lain)."""
    keys = list(runs)
    ari = {k: [] for k in keys}
    for a, b in combinations(keys, 2):
        s = adjusted_rand_score(runs[a]["labels"], runs[b]["labels"])
        ari[a].append(s)
        ari[b].append(s)
    rows = []
    det = cells["n_det"].to_numpy()
    for (mcs, ms), r in runs.items():
        lab = r["labels"]
        in_c = lab >= 0
        sizes = np.bincount(lab[in_c]) if in_c.any() else np.array([])
        rows.append({"wpp": wpp, "min_cluster_size": mcs, "min_samples": ms, "n_cells_input": len(cells),
                     "n_clusters": r["n_clusters"], "noise_frac_cells": round(r["noise_frac"], 4),
                     "det_share_in_clusters": round(float(det[in_c].sum() / det.sum()), 4),
                     "largest_cluster_cells": int(sizes.max()) if sizes.size else 0,
                     "median_cluster_cells": float(np.median(sizes)) if sizes.size else 0,
                     "mean_persistence": round(float(r["persistence"].mean()), 4) if r["persistence"].size else np.nan,
                     "median_persistence": round(float(np.median(r["persistence"])), 4) if r["persistence"].size else np.nan,
                     "mean_ari_vs_other_params": round(float(np.mean(ari[(mcs, ms)])), 4) if ari[(mcs, ms)] else np.nan,
                     "seconds": r["seconds"]})
    return pd.DataFrame(rows)


def choose_params(summary: pd.DataFrame) -> tuple[int, int]:
    """Aturan yang ditetapkan sebelum melihat hasil: kombinasi dengan ARI rata-rata tertinggi terhadap kombinasi lain
    (paling stabil terhadap perubahan parameter) di antara yang menghasilkan >=1 cluster; seri -> min_cluster_size
    lebih besar, lalu min_samples lebih besar. Noise minimum TIDAK dipakai sebagai kriteria."""
    s = summary[summary["n_clusters"] > 0]
    if s.empty:
        s = summary
    s = s.sort_values(["mean_ari_vs_other_params", "min_cluster_size", "min_samples"], ascending=False)
    r = s.iloc[0]
    return int(r["min_cluster_size"]), int(r["min_samples"])


def build_cfa(wpp: int, cells: pd.DataFrame, labels: np.ndarray, persistence: np.ndarray, size_m: int,
              crs: str) -> tuple[pd.DataFrame, "gpd.GeoDataFrame", pd.DataFrame]:
    """Katalog CFA, geometri (union sel), dan pemetaan sel->CFA. ID stabil diurutkan barat->timur menurut centroid."""
    import geopandas as gpd

    c = cells.assign(label=labels)
    c = c[c["label"] >= 0].copy()
    geoms = grid_cells_geoms(c["ix"].to_numpy(), c["iy"].to_numpy(), size_m)
    c["geometry"] = geoms
    recs = []
    for lab, grp in c.groupby("label"):
        poly = shapely.coverage_union_all(grp["geometry"].to_numpy())
        recs.append({"label": int(lab), "geometry": poly, "n_cells": len(grp),
                     "area_km2": poly.area / 1e6, "n_det_train": int(grp["n_det"].sum()),
                     "median_cell_n_years": float(grp["n_years"].median()),
                     "median_cell_n_dates": float(grp["n_dates"].median()),
                     "rad_median_of_cell_medians": float(grp["rad_median"].median()),
                     "persistence": float(persistence[lab]) if lab < len(persistence) else np.nan,
                     "n_parts": shapely.get_num_geometries(poly)})
    if not recs:
        return pd.DataFrame(), gpd.GeoDataFrame(geometry=[], crs=crs), pd.DataFrame()
    g = gpd.GeoDataFrame(recs, geometry="geometry", crs=crs)
    cen = g.geometry.centroid.to_crs("EPSG:4326")
    g["centroid_lon"], g["centroid_lat"] = cen.x.to_numpy(), cen.y.to_numpy()
    g = g.sort_values(["centroid_lon", "centroid_lat"]).reset_index(drop=True)
    g["cfa_id"] = [f"WPP{wpp}_CFA{i + 1:03d}" for i in range(len(g))]
    g["wpp"] = wpp
    g["det_share_of_wpp_train"] = g["n_det_train"] / cells["n_det"].sum()
    lab2id = dict(zip(g["label"], g["cfa_id"]))
    cellmap = c[["wpp", "ix", "iy", "cell_id", "label"]].assign(cfa_id=lambda d: d["label"].map(lab2id))
    cat = pd.DataFrame(g.drop(columns="geometry"))
    return cat, g, cellmap.drop(columns="label")
