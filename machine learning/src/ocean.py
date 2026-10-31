"""SST & klorofil-a bulanan per WPP dari ringkasan website (series.json), dengan metadata asal.

Asal (dibaca dari meta.json & scripts/oseanografi/build_ocean_summary.py repo website):
- SST : cmems_mod_glo_phy-mnstd_my_0.25deg_P1M-m (GLOBAL_MULTIYEAR_PHY_ENS_001_031) — REANALISIS (rata-rata ensemble),
        variabel thetao_mean lapisan pertama (min_depth 0.4 m – max_depth 1 m), 0.25°.
- CHL : cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M (OCEANCOLOUR_GLO_BGC_L4_MY_009_104) — OBSERVASI satelit
        multi-sensor L4 (gap-filled) 4 km; agregasi WPP = rata-rata GEOMETRIK (mean log10, lalu 10^x).
- Agregasi WPP: rata-rata berbobot cos(lintang) atas sel grid sumber yang pusatnya di dalam poligon WPP yang sama
  (shapefile yang sama, tanpa irisan EEZ). Nilai dibulatkan ke 4 angka penting (fungsi sig()).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import file_fingerprint


def load_ocean(ocean_dir: Path, vars_=("sst", "chl")) -> tuple[pd.DataFrame, dict]:
    series = json.loads((ocean_dir / "series.json").read_text(encoding="utf-8"))
    meta = json.loads((ocean_dir / "meta.json").read_text(encoding="utf-8"))
    frames = []
    for v in vars_:
        d = series[v]
        months = pd.to_datetime(pd.Series(d["months"]) + "-01")
        for w in [k for k in d if k not in ("months", "total")]:
            frames.append(pd.DataFrame({"wpp": int(w), "month": months, "var": v,
                                        "value": pd.to_numeric(pd.Series(d[w], dtype="object"), errors="coerce")}))
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["wpp", "month"], columns="var", values="value").reset_index()
    wide.columns.name = None
    if "chl" in wide:
        if (wide["chl"] <= 0).any():
            raise ValueError("klorofil <= 0 ditemukan; log tidak boleh dihitung")
        wide["log10_chl"] = np.log10(wide["chl"])
    var_meta = {x["key"]: x for x in meta["variables"] if x["key"] in vars_}
    info = {
        "files": [file_fingerprint(ocean_dir / "series.json"), file_fingerprint(ocean_dir / "meta.json")],
        "generatedAt_website": meta.get("generatedAt"), "source_statement": meta.get("source"),
        "variables": var_meta,
        "product_type": {"sst": "reanalysis ensemble mean (model), bukan pengukuran langsung",
                         "chl": "satellite ocean-colour L4 multi-sensor gap-filled (observasi terinterpolasi)"},
        "depth": {"sst": "lapisan permukaan model 0.4–1 m", "chl": "permukaan (ocean colour)"},
        "aggregation": "rata-rata berbobot cos(lat) sel di dalam poligon WPP; klorofil rata-rata geometrik (log10)",
        "rounding": "4 angka penting (sig(x, 4)) di series.json",
        "native_resolution": {"sst": "0.25°", "chl": "4 km"},
        "coverage": {v: [str(wide.loc[wide[v].notna(), "month"].min().date()), str(wide.loc[wide[v].notna(), "month"].max().date())]
                     for v in vars_},
        "n_missing": {v: int(wide[v].isna().sum()) for v in vars_},
    }
    return wide, info
