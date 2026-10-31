"""Konversi CSV VIIRS bersih (read-only) -> Parquet bertipe di cache, tanpa membuang baris.

Semua kolom dibaca sebagai teks lalu di-TRY_CAST, sehingga kegagalan parsing dapat dihitung
(parse_issue bitmask: 1=lat, 2=lon, 4=rad, 8=timestamp) alih-alih diam-diam hilang.
"""
from __future__ import annotations

from pathlib import Path

from .common import cache_dir, duck, file_fingerprint, p

TYPED_PARQUET = "viirs_typed.parquet"


def typed_parquet_path(cfg) -> Path:
    return cache_dir(cfg) / TYPED_PARQUET


def build_typed_parquet(cfg, force: bool = False) -> Path:
    src = p(cfg, "viirs_clean")
    out = typed_parquet_path(cfg)
    if out.exists() and not force:
        return out
    con = duck(cfg)
    tmp = out.with_suffix(".tmp.parquet")
    con.execute(f"""
    COPY (
      WITH r AS (
        SELECT * FROM read_csv('{src.as_posix()}', header=true, all_varchar=true)
      )
      SELECT
        TRY_CAST(Lat_DNB AS DOUBLE)      AS lat,
        TRY_CAST(Lon_DNB AS DOUBLE)      AS lon,
        TRY_CAST(Rad_DNB AS DOUBLE)      AS rad_dnb,
        TRY_CAST(Date_Mscan AS TIMESTAMP) AS ts_utc,
        TRY_CAST(TRY_CAST(QF_Detect AS DOUBLE) AS INTEGER)  AS qf_detect,
        TRY_CAST(TRY_CAST(QF_Bitflag AS DOUBLE) AS BIGINT)  AS qf_bitflag,
        TRY_CAST(TRY_CAST(Land_Mask AS DOUBLE) AS INTEGER)  AS land_mask,
        NULLIF(EEZ, '')                  AS eez,
        NULLIF(MPA, '')                  AS mpa,
        TRY_CAST(Rad_I04 AS DOUBLE)      AS rad_i04,
        TRY_CAST(Glint AS DOUBLE)        AS glint,
        TRY_CAST("Date" AS DATE)         AS date_col,
        TRY_CAST(TRY_CAST("Year" AS DOUBLE) AS INTEGER)  AS year_col,
        TRY_CAST(TRY_CAST("Month" AS DOUBLE) AS INTEGER) AS month_col,
        (CASE WHEN Lat_DNB IS NOT NULL AND TRY_CAST(Lat_DNB AS DOUBLE) IS NULL THEN 1 ELSE 0 END
         + CASE WHEN Lon_DNB IS NOT NULL AND TRY_CAST(Lon_DNB AS DOUBLE) IS NULL THEN 2 ELSE 0 END
         + CASE WHEN Rad_DNB IS NOT NULL AND TRY_CAST(Rad_DNB AS DOUBLE) IS NULL THEN 4 ELSE 0 END
         + CASE WHEN Date_Mscan IS NOT NULL AND TRY_CAST(Date_Mscan AS TIMESTAMP) IS NULL THEN 8 ELSE 0 END
        )::TINYINT AS parse_issue
      FROM r
    ) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
    """)
    con.close()
    tmp.replace(out)
    return out


def source_manifest(cfg) -> dict:
    return file_fingerprint(p(cfg, "viirs_clean"))
