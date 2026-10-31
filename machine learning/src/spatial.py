"""Geometri WPP, penentuan WPP point-in-polygon, dan grid equal-area.

Shapefile WPP hanya berisi .shp/.shx (tanpa .dbf dan .prj): kode WPP diambil dari nama file
dan CRS diasumsikan EPSG:4326 (koordinat berupa derajat). Asumsi ini dicatat di audit.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from .common import cache_dir, p


WPP_PARQUET = "viirs_wpp.parquet"


def load_wpp(cfg) -> "gpd.GeoDataFrame":
    """GeoDataFrame 11 WPP (make_valid), di-cache sebagai GeoParquet."""
    import geopandas as gpd

    cache = cache_dir(cfg) / "wpp_valid.parquet"
    if cache.exists():
        return gpd.read_parquet(cache)
    rows = []
    for code in cfg["wpp_codes"]:
        f = p(cfg, "wpp_dir") / cfg["paths"]["wpp_file_pattern"].format(wpp=code)
        g = gpd.read_file(f)
        geom = shapely.union_all(shapely.make_valid(g.geometry.values))
        # make_valid bisa menghasilkan GeometryCollection berisi garis: ambil bagian poligonal saja
        if geom.geom_type == "GeometryCollection":
            geom = shapely.union_all([x for x in geom.geoms if x.geom_type in ("Polygon", "MultiPolygon")])
        rows.append({"wpp": int(code), "valid_original": bool(g.geometry.is_valid.all()),
                     "n_vertices": int(shapely.get_num_coordinates(geom)), "geometry": geom})
    gdf = gpd.GeoDataFrame(rows, crs=cfg["wpp_assumed_crs"])
    gdf.to_parquet(cache)
    return gdf


def wpp_simplified(cfg, tol_deg: float = 0.01):
    """Versi sederhana untuk peta (bukan untuk analisis)."""
    import geopandas as gpd

    cache = cache_dir(cfg) / f"wpp_simplified_{tol_deg}.parquet"
    if cache.exists():
        return gpd.read_parquet(cache)
    g = load_wpp(cfg).copy()
    g["geometry"] = g.geometry.simplify(tol_deg, preserve_topology=True)
    g.to_parquet(cache)
    return g


def assign_wpp(lon: np.ndarray, lat: np.ndarray, gdf) -> tuple[np.ndarray, np.ndarray]:
    """Kembalikan (kode WPP terpilih atau 0, jumlah poligon yang memuat titik).

    Aturan batas deterministik: jika titik jatuh di >1 poligon, pilih kode WPP terkecil.
    """
    n = len(lon)
    chosen = np.zeros(n, dtype=np.int16)
    hits = np.zeros(n, dtype=np.int8)
    ok = np.isfinite(lon) & np.isfinite(lat)
    for code, geom in sorted(zip(gdf["wpp"], gdf.geometry)):  # urut naik -> kode terkecil menang
        shapely.prepare(geom)
        minx, miny, maxx, maxy = geom.bounds
        sel = np.nonzero(ok & (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy))[0]
        if sel.size == 0:
            continue
        inside = sel[shapely.contains_xy(geom, lon[sel], lat[sel])]
        hits[inside] += 1
        first = inside[chosen[inside] == 0]
        chosen[first] = code
    return chosen, hits


def build_wpp_parquet(cfg, typed_parquet: Path, force: bool = False, batch_rows: int = 1_000_000) -> Path:
    """Tambahkan kolom wpp & wpp_hits ke setiap baris (urutan baris dipertahankan, tanpa membuang baris)."""
    out = cache_dir(cfg) / WPP_PARQUET
    if out.exists() and not force:
        return out
    gdf = load_wpp(cfg)
    pf = pq.ParquetFile(typed_parquet)
    tmp = out.with_suffix(".tmp.parquet")
    writer = None
    for batch in pf.iter_batches(batch_size=batch_rows):
        lon = batch.column("lon").to_numpy(zero_copy_only=False).astype(float)
        lat = batch.column("lat").to_numpy(zero_copy_only=False).astype(float)
        chosen, hits = assign_wpp(lon, lat, gdf)
        wpp_arr = pa.array(np.where(chosen > 0, chosen, 0).astype(np.int16), mask=chosen == 0)
        tbl = pa.Table.from_batches([batch]).append_column("wpp", wpp_arr).append_column("wpp_hits", pa.array(hits))
        if writer is None:
            writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
        writer.write_table(tbl)
    writer.close()
    tmp.replace(out)
    return out


# ---------------------------------------------------------------- grid equal-area
def grid_ids(lon: np.ndarray, lat: np.ndarray, crs: str, size_m: float):
    """Indeks sel grid persegi `size_m` pada CRS equal-area. Kembalikan (ix, iy)."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(lon, lat)
    return np.floor(np.asarray(x) / size_m).astype(np.int64), np.floor(np.asarray(y) / size_m).astype(np.int64)


def grid_centers_lonlat(ix: np.ndarray, iy: np.ndarray, crs: str, size_m: float):
    from pyproj import Transformer

    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform((ix + 0.5) * size_m, (iy + 0.5) * size_m)
    return np.asarray(lon), np.asarray(lat)


def grid_cells_geoms(ix: np.ndarray, iy: np.ndarray, size_m: float):
    """Poligon sel grid dalam CRS equal-area (meter)."""
    return shapely.box(ix * size_m, iy * size_m, (ix + 1) * size_m, (iy + 1) * size_m)


def cell_id(wpp, ix, iy, size_m: int) -> pd.Series:
    return pd.Series([f"G{int(size_m)}_{a}_{b}" for a, b in zip(ix, iy)])


# ---------------------------------------------------------------- dataset A: grid spasial
def build_points_grid(cfg, analytical: Path, out: Path, sizes=(1000, 2000), batch_rows: int = 1_000_000) -> Path:
    """Indeks sel grid per titik (EPSG:6933) untuk beberapa ukuran; tanpa menyalin seluruh atribut."""
    if out.exists():
        return out
    crs = cfg["spatial"]["grid_crs"]
    pf = pq.ParquetFile(analytical)
    writer = None
    tmp = out.with_suffix(".tmp.parquet")
    for b in pf.iter_batches(batch_size=batch_rows, columns=["date_utc", "year", "month", "wpp", "lat", "lon", "rad_dnb"]):
        t = pa.Table.from_batches([b])
        lon = t["lon"].to_numpy().astype(float)
        lat = t["lat"].to_numpy().astype(float)
        for s in sizes:
            ix, iy = grid_ids(lon, lat, crs, s)
            t = t.append_column(f"ix_{s}", pa.array(ix)).append_column(f"iy_{s}", pa.array(iy))
        t = t.drop_columns(["lat", "lon"])
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t)
    writer.close()
    tmp.replace(out)
    return out


def aggregate_grid(cfg, con, points: Path, size: int, start: str, end: str) -> pd.DataFrame:
    """Statistik per sel (WPP, ix, iy) pada rentang [start, end]. Satu tanggal UTC = satu malam."""
    df = con.sql(f"""
      SELECT wpp, ix_{size} AS ix, iy_{size} AS iy,
             count(*) AS n_det,
             count(DISTINCT date_utc) AS n_dates,
             count(DISTINCT year) AS n_years,
             count(DISTINCT year::INT * 100 + month::INT) AS n_months,
             median(rad_dnb) AS rad_median, avg(rad_dnb) AS rad_mean, max(rad_dnb) AS rad_max,
             min(date_utc) AS first_date, max(date_utc) AS last_date
      FROM read_parquet('{points.as_posix()}')
      WHERE date_utc BETWEEN DATE '{start}' AND DATE '{end}'
      GROUP BY ALL""").df()
    lon, lat = grid_centers_lonlat(df["ix"].to_numpy(), df["iy"].to_numpy(), cfg["spatial"]["grid_crs"], size)
    df["center_lon"], df["center_lat"] = lon, lat
    df["cell_id"] = [f"W{w}_G{size}_{a}_{b}" for w, a, b in zip(df["wpp"], df["ix"], df["iy"])]
    df["grid_size_m"] = size
    df["period_start"], df["period_end"] = start, end
    return df
