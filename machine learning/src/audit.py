"""Audit ringan data bersih + pembentukan view analitis (tanpa menimpa input).

Semua angka di sini dihitung dari input saat ini. Angka paper/website hanya dicantumkan
sebagai kolom pembanding berlabel `quoted_*`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import duck, file_fingerprint, p, run_dir

# Dikutip dari paper (Tabel 2 & Tabel 5) dan dari preprocessing.json repo website — BUKAN hasil program ini.
QUOTED_PAPER = {
    "raw_records": 18115494, "post_cleaning_records": 14211269, "analytical_records": 7380357,
    "wpp_counts": {571: 432304, 572: 196361, 573: 409597, 711: 532758, 712: 3887263, 713: 826975,
                   714: 20828, 715: 70741, 716: 153085, 717: 37160, 718: 684256},
    "time_coverage_text": "2012-July 2026",
}
QUOTED_WEBSITE_STEPS = {"valid": 14211268, "qf1": 11164249, "land_mask_ne_1": 11098366,
                        "eez_indonesia": 7379853, "wpp": 7375153}

SENTINEL = 999999
EEZ_ID = "Indonesian Exclusive Economic Zone"

DICTIONARY = [
    # kolom_input, kolom_internal, tipe, unit, deskripsi V23 (readme EOG), peran
    ("Lat_DNB", "lat", "float64", "derajat", "VBD pixel latitude from VIIRS DNB geolocation file (fill 999999)", "lokasi titik; grid; HDBSCAN"),
    ("Lon_DNB", "lon", "float64", "derajat", "VBD pixel longitude from VIIRS DNB geolocation file (fill 999999)", "lokasi titik; grid; HDBSCAN"),
    ("Rad_DNB", "rad_dnb", "float64", "nW/cm2/sr", "Radiance of VBD pixel in VIIRS DNB band (fill 999999)", "ringkasan radiansi; kelas radiansi (deskriptif)"),
    ("Date_Mscan", "ts_utc", "timestamp", "UTC", "VBD pixel date-time at mid-point of DNB scan reported in Universal Time", "waktu kejadian; tanggal/bulan UTC"),
    ("QF_Detect", "qf_detect", "int", "-", "1 strong, 2 weak, 3 blurry, 4 gas flare, ... (V23 QF_Detect tab)", "filter kualitas QF=1"),
    ("QF_Bitflag", "qf_bitflag", "int64", "-", "Sum of VBD algorithm bit flags", "diagnostik"),
    ("Land_Mask", "land_mask", "int", "-", "Land-sea mask flag. Land=0. Water=3. Near-shore=1,2", "filter daratan (kode 0)"),
    ("EEZ", "eez", "string", "-", "Exclusive Economic Zone for VBD pixel", "konteks; tidak dipakai sebagai filter utama"),
    ("MPA", "mpa", "string", "-", "Marine Protected Area for VBD pixel", "konteks; bukan indikator pelanggaran"),
    ("Rad_I04", "rad_i04", "float64", "W/m2/sr/um", "Radiance of VBD pixel in VIIRS I4 band (3.7 um)", "diagnostik"),
    ("Glint", "glint", "float64", "0-1", "Probability of lunar glint impacting DNB data", "diagnostik"),
    ("Date", "date_col", "date", "UTC", "Turunan cleaning lama dari Date_Mscan", "diperiksa konsistensinya; tidak dipakai"),
    ("Year", "year_col", "int", "-", "Turunan cleaning lama", "diperiksa konsistensinya; tidak dipakai"),
    ("Month", "month_col", "int", "-", "Turunan cleaning lama", "diperiksa konsistensinya; tidak dipakai"),
    ("Day", "(tidak dibawa)", "-", "-", "Turunan cleaning lama", "tidak dipakai"),
    ("(turunan)", "wpp", "int16", "-", "Kode WPP hasil point-in-polygon pada 11 shapefile WPP (EPSG:4326 diasumsikan)", "unit stratifikasi"),
    ("(turunan)", "wpp_hits", "int8", "-", "Jumlah poligon WPP yang memuat titik", "pemeriksaan tumpang tindih batas"),
]

VALID_SQL = f"""ts_utc IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL AND rad_dnb IS NOT NULL
  AND lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
  AND abs(lat) <> {SENTINEL} AND abs(lon) <> {SENTINEL} AND abs(rad_dnb) <> {SENTINEL}"""


def analytical_path(cfg) -> Path:
    return run_dir(cfg, "data") / "viirs_analytical.parquet"


def run_audit(cfg, wpp_parquet: Path) -> dict:
    out = run_dir(cfg, "audit")
    con = duck(cfg)
    con.execute(f"CREATE VIEW v AS SELECT * FROM read_parquet('{wpp_parquet.as_posix()}')")
    q = lambda s: con.sql(s).df()
    one = lambda s: con.sql(s).fetchone()[0]

    src = file_fingerprint(p(cfg, "viirs_clean"))
    with open(p(cfg, "viirs_clean"), encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    # ---------------- inventaris
    inv = q("""SELECT count(*) n_rows, min(ts_utc) ts_min, max(ts_utc) ts_max,
               sum((parse_issue>0)::INT) n_parse_issue, sum((ts_utc IS NULL)::INT) n_ts_null,
               sum((lat IS NULL OR lon IS NULL)::INT) n_coord_null, sum((rad_dnb IS NULL)::INT) n_rad_null
               FROM v""").iloc[0].to_dict()
    sentinel = q(f"""SELECT {', '.join(f"sum((abs({c})={SENTINEL})::INT) AS {c}" for c in
                    ['lat', 'lon', 'rad_dnb', 'qf_detect', 'qf_bitflag', 'land_mask', 'rad_i04', 'glint'])} FROM v""").iloc[0].to_dict()
    diag_invalid = q("""SELECT sum((glint > 1)::INT) glint_gt_1, sum((glint >= 999999)::INT) glint_ge_999999,
                        max(glint) glint_max, sum((rad_i04 <= -999)::INT) rad_i04_le_minus999,
                        sum((rad_i04 <= -999 AND qf_detect = 1)::INT) rad_i04_le_minus999_qf1 FROM v""").iloc[0].to_dict()
    derived_consistency = q("""SELECT sum((date_col <> CAST(ts_utc AS DATE))::INT) date_mismatch,
                               sum((year_col <> year(ts_utc))::INT) year_mismatch,
                               sum((month_col <> month(ts_utc))::INT) month_mismatch FROM v""").iloc[0].to_dict()
    qf = q("SELECT qf_detect, count(*) n FROM v GROUP BY 1 ORDER BY 1")
    lm = q("SELECT land_mask, count(*) n FROM v GROUP BY 1 ORDER BY 1")
    hours = q("SELECT hour(ts_utc) hour_utc, count(*) n FROM v GROUP BY 1 ORDER BY 1")
    rad_nonpos = q("SELECT qf_detect, count(*) n FROM v WHERE rad_dnb <= 0 GROUP BY 1 ORDER BY 1")
    overlap = q("SELECT wpp_hits, count(*) n FROM v GROUP BY 1 ORDER BY 1")
    dup_all = one("SELECT count(*) - count(DISTINCT (ts_utc, lat, lon, rad_dnb, qf_detect)) FROM v")
    bbox = q("SELECT min(lat) lat_min, max(lat) lat_max, min(lon) lon_min, max(lon) lon_max FROM v").iloc[0].to_dict()

    # ---------------- tahap view analitis utama (berurutan)
    stages = []
    def stage(name, where, desc):
        n = one(f"SELECT count(*) FROM v WHERE {where}")
        stages.append({"variant": "primary", "stage": name, "rule": desc, "n_records": int(n)})
    w0 = "TRUE"; stage("00_input", w0, "Semua baris viirs_clean_all.csv")
    w1 = VALID_SQL; stage("01_valid", w1, "Waktu, koordinat, radiansi terbaca; bukan fill 999999; rentang koordinat valid")
    w2 = w1 + " AND qf_detect = 1"; stage("02_qf1", w2, "QF_Detect = 1 (strong/boat detection, V23)")
    w3 = w2 + " AND coalesce(land_mask, -1) <> 0"; stage("03_not_land", w3, "Land_Mask <> 0 (kode daratan V23); near-shore 1/2 dipertahankan")
    w4 = w3 + " AND wpp IS NOT NULL"; stage("04_in_wpp", w4, "Di dalam salah satu dari 11 poligon WPP (point-in-polygon)")
    n_dedup = one(f"SELECT count(DISTINCT (ts_utc, lat, lon, rad_dnb, qf_detect)) FROM v WHERE {w4}")
    stages.append({"variant": "primary", "stage": "05_dedup_exact", "n_records": int(n_dedup),
                   "rule": "Buang duplikat eksak (Date_Mscan, Lat, Lon, Rad_DNB, QF sama)"})
    # varian pembanding seperti paper / website lama
    lw = w2 + " AND land_mask <> 1"
    stages.append({"variant": "legacy_like", "stage": "L03_land_mask_ne_1", "rule": "Land_Mask != 1 (perlakuan lama; 1 = near-shore di V23)",
                   "n_records": int(one(f"SELECT count(*) FROM v WHERE {lw}"))})
    lw2 = lw + f" AND eez = '{EEZ_ID}'"
    stages.append({"variant": "legacy_like", "stage": "L04_eez_indonesia", "rule": "EEZ = Indonesian EEZ",
                   "n_records": int(one(f"SELECT count(*) FROM v WHERE {lw2}"))})
    lw3 = lw2 + " AND wpp IS NOT NULL"
    stages.append({"variant": "legacy_like", "stage": "L05_in_wpp", "rule": "Di dalam 11 poligon WPP (tanpa dedup)",
                   "n_records": int(one(f"SELECT count(*) FROM v WHERE {lw3}"))})
    st = pd.DataFrame(stages)
    st["quoted_website_preprocessing_json"] = st["stage"].map({
        "01_valid": QUOTED_WEBSITE_STEPS["valid"], "02_qf1": QUOTED_WEBSITE_STEPS["qf1"],
        "L03_land_mask_ne_1": QUOTED_WEBSITE_STEPS["land_mask_ne_1"],
        "L04_eez_indonesia": QUOTED_WEBSITE_STEPS["eez_indonesia"], "L05_in_wpp": QUOTED_WEBSITE_STEPS["wpp"]})
    st["quoted_paper_table2"] = st["stage"].map({"00_input": QUOTED_PAPER["post_cleaning_records"]})

    # ---------------- per WPP: utama vs varian lama vs paper
    per_wpp = q(f"""
      SELECT wpp,
        count(DISTINCT (ts_utc, lat, lon, rad_dnb, qf_detect)) FILTER (WHERE {w4}) AS primary_dedup,
        count(*) FILTER (WHERE {w4}) AS primary_with_dups,
        count(*) FILTER (WHERE {w4} AND land_mask = 1) AS primary_near_shore_lm1,
        count(*) FILTER (WHERE {w4} AND coalesce(eez,'') <> '{EEZ_ID}') AS primary_outside_id_eez,
        count(*) FILTER (WHERE {lw3}) AS legacy_like
      FROM v WHERE wpp IS NOT NULL GROUP BY 1 ORDER BY 1""")
    per_wpp["quoted_paper_table5"] = per_wpp["wpp"].map(QUOTED_PAPER["wpp_counts"])
    eez_in_wpp = q(f"SELECT coalesce(eez,'(kosong)') eez, count(*) n FROM v WHERE {w4} GROUP BY 1 ORDER BY n DESC")

    # ---------------- ketersediaan bulanan (keberadaan rekaman, bukan cakupan bebas awan)
    monthly_avail = q(f"""
      WITH m AS (SELECT date_trunc('month', ts_utc)::DATE AS "month", count(*) n_all_qf,
                        count(*) FILTER (WHERE {w4}) n_primary_in_wpp,
                        count(DISTINCT CAST(ts_utc AS DATE)) n_dates_any_record
                 FROM v WHERE ts_utc IS NOT NULL GROUP BY 1)
      SELECT * FROM m ORDER BY month""")
    cal = pd.DataFrame({"month": pd.date_range("2012-04-01", monthly_avail["month"].max(), freq="MS")})
    monthly_avail["month"] = pd.to_datetime(monthly_avail["month"])
    monthly_avail = cal.merge(monthly_avail, on="month", how="left")
    monthly_avail["days_in_month"] = monthly_avail["month"].dt.days_in_month
    missing_months = monthly_avail.loc[monthly_avail["n_all_qf"].isna(), "month"].dt.strftime("%Y-%m").tolist()

    # ---------------- tulis view analitis utama (dedup deterministik)
    ap = analytical_path(cfg)
    con.execute(f"""
      COPY (
        SELECT ts_utc, CAST(ts_utc AS DATE) AS date_utc, year(ts_utc)::SMALLINT AS year, month(ts_utc)::TINYINT AS month,
               lat, lon, rad_dnb, qf_detect, qf_bitflag, land_mask, eez, mpa, rad_i04, glint, wpp
        FROM (SELECT *, row_number() OVER (PARTITION BY ts_utc, lat, lon, rad_dnb, qf_detect
                                           ORDER BY qf_bitflag, land_mask, eez, rad_i04, glint) rn
              FROM v WHERE {w4})
        WHERE rn = 1
        ORDER BY wpp, ts_utc
      ) TO '{ap.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    n_written = one(f"SELECT count(*) FROM read_parquet('{ap.as_posix()}')")
    assert n_written == n_dedup, (n_written, n_dedup)
    span = q(f"SELECT min(ts_utc) mn, max(ts_utc) mx FROM read_parquet('{ap.as_posix()}')").iloc[0]

    # ---------------- simpan
    dd = pd.DataFrame(DICTIONARY, columns=["input_column", "internal_column", "dtype", "unit", "definition_v23_or_note", "role"])
    dd["present_in_input"] = dd["input_column"].isin(header) | dd["input_column"].eq("(turunan)")
    dd["source"] = "EOG VBD readme V23 (references/eog_vbd_readme_v23_r20180824.xlsx) / turunan program"
    dd.to_csv(out / "data_dictionary.csv", index=False)
    st.to_csv(out / "audit_counts.csv", index=False)
    per_wpp.to_csv(out / "audit_counts_by_wpp.csv", index=False)
    eez_in_wpp.to_csv(out / "eez_within_wpp_primary.csv", index=False)
    monthly_avail.to_csv(out / "monthly_record_availability.csv", index=False)

    audit = {
        "source_file": src, "input_columns": header,
        "inventory": {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in inv.items()},
        "bbox_input": bbox, "sentinel_999999_by_column": sentinel,
        "diagnostic_columns_out_of_range": diag_invalid,
        "derived_column_consistency": derived_consistency,
        "qf_detect_counts": qf.to_dict("records"), "land_mask_counts": lm.to_dict("records"),
        "hour_utc_counts": hours.to_dict("records"), "nonpositive_radiance_by_qf": rad_nonpos.to_dict("records"),
        "wpp_overlap_hits": overlap.to_dict("records"), "exact_duplicates_all_rows": int(dup_all),
        "stages": st.to_dict("records"), "analytical_view": {"path": str(ap), "n_records": int(n_written),
                                                              "ts_min": str(span["mn"]), "ts_max": str(span["mx"])},
        "months_without_any_record": missing_months,
        "quoted_paper_not_computed": QUOTED_PAPER,
        "decisions": {
            "qf": "QF_Detect=1 dipertahankan (sesuai paper). QF bukan label fishing/transit.",
            "land_mask": "Input tidak berisi Land_Mask=0 (daratan V23). Land_Mask=1 = near-shore menurut readme V23 dan dipertahankan; "
                         "varian lama 'Land_Mask != 1' dilaporkan sebagai pembanding.",
            "eez": "Tidak ada filter EEZ pada view utama: domain = poligon WPP, sama dengan domain ringkasan oseanografi.",
            "duplicates": "Duplikat eksak (waktu-scan, lat, lon, radiansi, QF identik) dibuang; koordinat sama pada malam berbeda dipertahankan.",
            "outliers": "Tidak ada pembuangan outlier radiansi (IQR/z-score).",
            "wpp_crs": "Shapefile tanpa .prj/.dbf; CRS EPSG:4326 diasumsikan dari rentang koordinat; kode WPP dari nama file.",
            "boundary_rule": "Titik di >1 poligon -> kode WPP terkecil.",
            "date": "Tanggal/bulan memakai UTC. Seluruh lintasan malam Indonesia (jam 14-20 UTC) jatuh pada satu tanggal UTC, "
                    "sehingga tanggal UTC dapat dipakai sebagai identitas malam.",
        },
    }
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    con.close()
    return audit
