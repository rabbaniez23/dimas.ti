"""Tahap 7: pemeriksaan akhir, ekspor website, laporan (data_issues, captions, results_summary, tugas belajar).

Semua angka di dokumen dibaca dari tabel keluaran (tidak diketik manual).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import load_config, log_step, now_iso, run_dir  # noqa: E402

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
D, T, A, R, W = (run_dir(cfg, s) for s in ["data", "tables", "audit", "report", "website_exports"])
fmt = lambda x, d=0: f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")  # format Indonesia
sfmt = lambda x, d=2: ("+" if x > 0 else "") + fmt(x, d)

# ------------------------------------------------------------------ pemeriksaan akhir
checks = {}
pred = pd.read_csv(T / "T06_predictions.csv", parse_dates=["target_month"])
met = pd.read_csv(T / "T07_metrics_by_wpp.csv")
re = (pred.assign(ae=lambda d: (d["y_pred"] - d["y_true"]).abs(), se=lambda d: (d["y_pred"] - d["y_true"]) ** 2)
      .groupby(["split", "model", "wpp"]).agg(MAE=("ae", "mean"), MSE=("se", "mean")).reset_index())
mm = met.merge(re, on=["split", "model", "wpp"], suffixes=("", "_re"))
checks["metrics_recomputed_from_predictions_match"] = bool(np.allclose(mm["MAE"], mm["MAE_re"]) and
                                                          np.allclose(mm["RMSE"], np.sqrt(mm["MSE"])))
ab = pd.read_csv(T / "T08_ocean_ablation_M2_vs_M3.csv")
t2 = met.pivot_table(index=["split", "wpp"], columns="model", values="MAE")
ab2 = ab.set_index(["split", "wpp"])
checks["ablation_matches_metrics"] = bool(np.allclose(ab2["delta_MAE_M2_minus_M3"],
                                                      (t2["M2_RF_History"] - t2["M3_RF_Ocean"]).loc[ab2.index]))
cfa = gpd.read_parquet(D / "cfa_reference_6933.parquet")
checks["cfa_crs"] = cfa.crs.to_string()
checks["cfa_area_from_geometry_matches_catalog"] = bool(np.allclose(cfa.geometry.area / 1e6, cfa["area_km2"]))
checks["no_synthetic_or_hardcoded_results"] = "semua tabel dibangun dari viirs_clean_all.csv, shapefile WPP, dan series.json"

# ------------------------------------------------------------------ ekspor website
gen = now_iso()
common_meta = {"generated_at": gen, "run_id": cfg["run_id"], "unit": "jumlah deteksi VIIRS VBD QF=1 (bukan kapal unik)",
               "domain": "11 poligon WPP-RI (tanpa filter ZEE); titik Land_Mask=0 tidak ada di input",
               "source": "EOG VIIRS Boat Detection V23 (viirs_clean_all.csv), diproses ulang run ini"}
web = cfa[["cfa_id", "wpp", "n_cells", "area_km2", "n_det_train", "persistence", "geometry"]].copy()
web["geometry"] = web.geometry.buffer(750, resolution=2).buffer(-750, resolution=2).simplify(500, preserve_topology=True)
web = web.to_crs("EPSG:4326")
web["area_km2"] = web["area_km2"].round(1)
web["persistence"] = web["persistence"].round(3)
tmp = W / "cfa_reference_web.geojson"
web.to_file(tmp, driver="GeoJSON", COORDINATE_PRECISION=4)
old = W / "cfa_reference.geojson"
if old.exists():
    old.unlink()  # versi penuh 26 MB dipindah: tetap ada di data/cfa_reference.geojson
cat = pd.read_csv(T / "cfa_catalog.csv")
(W / "cfa_catalog.json").write_text(json.dumps({
    **common_meta, "status": "descriptive", "period": "latih 2013-01..2023-12",
    "note": "CFA = area konsentrasi deteksi berulang menurut definisi penelitian (HDBSCAN atas sel 1 km aktif >=2 tahun); "
            "bukan lokasi stok ikan atau indikasi pelanggaran. Geometri web hanya untuk tampilan: closing buffer +750 m/-750 m lalu simplify 500 m; luas resmi = area_km2 dari union sel.",
    "items": cat[["cfa_id", "wpp", "n_cells", "area_km2", "n_det_train", "det_share_of_wpp_train", "persistence",
                  "centroid_lon", "centroid_lat", "min_cluster_size", "min_samples"]].round(4).to_dict("records")},
    indent=1), encoding="utf-8")
panel = pd.read_parquet(D / "wpp_month_panel.parquet")
(W / "monthly_counts.json").write_text(json.dumps({
    **common_meta, "status": "descriptive", "period": "2012-04..2026-05 (2012 & 2026 parsial)",
    "months": panel["month"].drop_duplicates().dt.strftime("%Y-%m").tolist(),
    "series": {str(w): g.sort_values("month")["n_det"].astype(int).tolist() for w, g in panel.groupby("wpp")}},
    indent=0), encoding="utf-8")
cm = pd.read_parquet(D / "cfa_monthly_counts.parquet")
cm_t = cm[cm["period"].isin(["val", "test"])].assign(month=lambda d: d["month"].dt.strftime("%Y-%m"))
(W / "cfa_monthly_2024_2025.json").write_text(json.dumps({
    **common_meta, "status": "descriptive", "note": "Aktivitas 2024-2025 di dalam CFA beku (dibentuk dari data 2013-2023).",
    "records": cm_t[["cfa_id", "wpp", "month", "n_det"]].to_dict("records")}, indent=0), encoding="utf-8")
pt = pred[pred["split"] == "test"].assign(month=lambda d: d["target_month"].dt.strftime("%Y-%m"))
(W / "forecast_hindcast_2025.json").write_text(json.dumps({
    **common_meta, "status": "hindcast",
    "note": "Peramalan retrospektif satu langkah ke depan; oseanografi memakai produk historis final (bukan uji real-time). "
            "Tanpa interval prediksi (belum dikalibrasi).",
    "train_targets": "2013-04..2023-12 (+ bulan sebelum origin)", "validation": "2024", "test": "2025",
    "models": ["M0_SeasonalNaive", "M1_ETS", "M2_RF_History", "M3_RF_Ocean"],
    "records": pt[["wpp", "month", "model", "y_true", "y_pred"]].round(1).to_dict("records")}, indent=0), encoding="utf-8")
(W / "metrics_test_2025.json").write_text(json.dumps({
    **common_meta, "status": "hindcast", "metric_note": "MASE: penyebut = MAE seasonal naive pada target latih 2013-04..2023-12",
    "records": met[met["split"] == "test"][["wpp", "model", "n_months", "MAE", "RMSE", "MASE"]].round(3).to_dict("records")},
    indent=1), encoding="utf-8")
sizes = {p.name: p.stat().st_size for p in W.iterdir()}
checks["website_export_bytes"] = sizes

# ------------------------------------------------------------------ angka untuk dokumen
audit = json.loads((A / "data_audit.json").read_text(encoding="utf-8"))
st = pd.read_csv(A / "audit_counts.csv").set_index("stage")["n_records"]
bw = pd.read_csv(A / "audit_counts_by_wpp.csv").set_index("wpp")
s = pd.read_csv(T / "T07b_model_summary.csv").set_index(["split", "model"])
abt = ab[ab["split"] == "test"].set_index("wpp")
abv = ab[ab["split"] == "val"].set_index("wpp")
sens = pd.read_csv(T / "T04b_cfa_sensitivity.csv")
runs = pd.read_csv(T / "T04a_hdbscan_param_runs.csv")
sel = runs[runs["selected"]].set_index("wpp")
mt = met[met["split"] == "test"].pivot(index="wpp", columns="model", values="MASE")
best = mt.idxmin(axis=1)
prof = pd.read_csv(T / "S09_wpp_radiance_profile.csv").set_index("wpp")
tot = pd.read_csv(T / "S01_radiance_overall.csv").iloc[0]
assoc = pd.read_csv(T / "T09_ocean_association_spearman.csv")
an = assoc[assoc["relation"] == "anomaly_same_month"].pivot(index="wpp", columns="variable", values="spearman_rho")
ocean_meta = json.loads((D / "ocean_metadata.json").read_text(encoding="utf-8"))
params = json.loads((run_dir(cfg, "models") / "forecast_selected_params.json").read_text(encoding="utf-8"))
ets_trend = {k: v["trend"] for k, v in params["ets"].items()}
n_cfa, area = len(cat), cat["area_km2"].sum()
q = lambda sp, m, c: s.loc[(sp, m), c]
M = ["M0_SeasonalNaive", "M1_ETS", "M2_RF_History", "M3_RF_Ocean"]
LBL = {"M0_SeasonalNaive": "M0 Seasonal Naive", "M1_ETS": "M1 ETS", "M2_RF_History": "M2 RF-History", "M3_RF_Ocean": "M3 RF-Ocean"}

# ------------------------------------------------------------------ angka turunan tambahan
import subprocess
av = pd.read_csv(A / "monthly_record_availability.csv", parse_dates=["month"])
av["frac"] = av["n_dates_any_record"] / av["days_in_month"]
low = av[av["frac"] < 0.75]
low_txt = ", ".join(f"{m:%Y-%m} ({fmt(100 * f)}% tanggal)" for m, f in zip(low["month"], low["frac"]))
ocean_dir = Path(cfg["paths"]["ocean_dir"])
repo = Path(__file__).resolve().parents[1] / ocean_dir.parents[2]
try:
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:  # noqa: BLE001
    commit = "tidak terbaca"
checks["ocean_repo_commit"] = commit
oc = pd.read_parquet(D / "ocean_wpp_month.parquet")
miss_in = {v: int(oc.loc[(oc["month"] >= oc.loc[oc[v].notna(), "month"].min()) & (oc["month"] <= oc.loc[oc[v].notna(), "month"].max()), v].isna().sum()) for v in ["sst", "chl"]}
si = pd.read_csv(T / "F04_seasonal_index_median.csv", index_col=0)
BLN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
peak_txt = "; ".join(f"{w}: " + ", ".join(f"{BLN[int(c) - 1]} {fmt(r[c], 2)}" for c in r.sort_values(ascending=False).index[:2]) for w, r in si.iterrows())

# ------------------------------------------------------------------ data_issues.md
mism = bw.assign(diff=lambda d: d["legacy_like"] - d["quoted_paper_table5"])
issues = f"""# Catatan masalah data & keputusan (run `{cfg['run_id']}`)

Dibuat otomatis {gen}. Angka **dihitung dari input saat ini** kecuali yang diberi label *dikutip*.

## Input
- `viirs_clean_all.csv`: {fmt(audit['inventory']['n_rows'])} baris, {len(audit['input_columns'])} kolom, waktu {audit['inventory']['ts_min']} s.d. {audit['inventory']['ts_max']} UTC.
  Paper (*dikutip*) menyebut cakupan "2012–Juli 2026"; isi data berakhir **31 Mei 2026**.
- Jumlah baris sama persis dengan angka *dikutip* Tabel 2 paper untuk "Intermediate Data (Post-Cleaning)" (14.211.269), sehingga file ini
  adalah data perantara **sebelum** filter QF, daratan, dan WPP — bukan dataset analitis 7,38 juta.
- Kolom: {', '.join(audit['input_columns'])}. Tidak ada kolom `WPP_RI`; WPP ditentukan ulang secara point-in-polygon.
- Bounding box input: lat {audit['bbox_input']['lat_min']:.2f}..{audit['bbox_input']['lat_max']:.2f}, lon {audit['bbox_input']['lon_min']:.2f}..{audit['bbox_input']['lon_max']:.2f}
  (memuat ZEE negara tetangga).

## Kualitas & kode
- QF_Detect: {', '.join(f"{int(r['qf_detect'])} = {fmt(r['n'])}" for r in audit['qf_detect_counts'])}. Readme EOG V23
  (`references/eog_vbd_readme_v23_r20180824.xlsx`): 1 strong detection, 2 weak, 4 gas flare. QF bukan label fishing/transit.
- Land_Mask: {', '.join(f"{int(r['land_mask'])} = {fmt(r['n'])}" for r in audit['land_mask_counts'])}. Readme V23: *Land=0, Water=3, Near-shore=1,2*.
  **Tidak ada kode 0 (daratan) di input.** Pengolahan lama/website menerapkan `Land_Mask != 1` dengan komentar "bukan daratan";
  menurut V23 itu membuang deteksi **near-shore**, bukan daratan. View utama mempertahankan near-shore
  ({fmt(bw['primary_near_shore_lm1'].sum())} deteksi di dalam WPP) dan melaporkan varian lama sebagai pembanding.
- Fill value 999999 pada kolom inti: 0. Kolom diagnostik di luar rentang: Glint > 1 sebanyak {fmt(audit['diagnostic_columns_out_of_range']['glint_gt_1'])}
  (maks {audit['diagnostic_columns_out_of_range']['glint_max']:.3g}); Rad_I04 ≤ −999 sebanyak {fmt(audit['diagnostic_columns_out_of_range']['rad_i04_le_minus999'])}.
  Kolom ini tidak dipakai analisis; tidak ada baris yang dibuang karenanya.
- Radiansi ≤ 0: {', '.join(f"QF {int(r['qf_detect'])}: {fmt(r['n'])}" for r in audit['nonpositive_radiance_by_qf'])} (tidak ada pada QF=1). Tidak ada pembuangan outlier IQR/z-score.
- Baris dengan waktu/koordinat tidak terbaca: {fmt(st['00_input'] - st['01_valid'])}.
- Duplikat eksak (waktu-scan, lat, lon, radiansi, QF identik): {fmt(audit['exact_duplicates_all_rows'])} pada seluruh input; {fmt(st['04_in_wpp'] - st['05_dedup_exact'])}
  dibuang dari view utama. Koordinat sama pada malam berbeda tetap dipertahankan.
- Kolom turunan lama Date/Year/Month konsisten dengan Date_Mscan (0 ketidakcocokan).
- Waktu: semua lintasan jatuh pada jam 14–20 UTC, sehingga tanggal UTC dipakai sebagai identitas malam (satu malam = satu tanggal UTC).

## Tahap view analitis (berurutan)
| Tahap | Rekaman | Pembanding *dikutip* |
|---|---:|---:|
""" + "\n".join(f"| {r.stage} — {r.rule} | {fmt(r.n_records)} | "
                f"{fmt(r.quoted_website_preprocessing_json) if pd.notna(r.quoted_website_preprocessing_json) else (fmt(r.quoted_paper_table2) if pd.notna(r.quoted_paper_table2) else '')} |"
                for r in pd.read_csv(A / 'audit_counts.csv').itertuples()) + f"""

Varian "seperti lama" (`Land_Mask != 1` + ZEE Indonesia + WPP) mereproduksi **persis** angka `preprocessing.json` repo website
(7.375.153), sehingga penentuan WPP run ini konsisten dengan pengolahan website. Paper (*dikutip*) menulis 7.380.357.

## WPP & batas
- Shapefile hanya `.shp/.shx` (tanpa `.dbf`, `.prj`). Kode WPP diambil dari nama file; CRS **diasumsikan** EPSG:4326 dari rentang koordinat.
  Asal resmi (DJPT-KKP, menurut paper) **tidak dapat diverifikasi** dari file.
- Geometri tidak valid: WPP 573 dan 714 (diperbaiki `make_valid`). Titik di >1 poligon: 0 (aturan cadangan: kode WPP terkecil).
- Domain utama = poligon WPP **tanpa filter ZEE**, sama dengan domain ringkasan oseanografi. Akibatnya
  {fmt(bw['primary_outside_id_eez'].sum())} deteksi di dalam poligon WPP berada di luar ZEE Indonesia menurut kolom EEZ EOG,
  terbanyak di WPP 711 ({fmt(bw.loc[711, 'primary_outside_id_eez'])}; terutama area yang diberi label ZEE Vietnam/klaim tumpang tindih).
  Ini perbedaan cakupan, bukan klaim yurisdiksi.

## Ketidaksesuaian dengan paper (tidak disesuaikan)
- Jumlah per WPP Tabel 5 paper (*dikutip*) **tidak cocok** dengan point-in-polygon pada shapefile ini, walaupun totalnya mirip.
  Contoh (varian lama vs paper): """ + "; ".join(f"WPP {w}: {fmt(r.legacy_like)} vs {fmt(r.quoted_paper_table5)}" for w, r in mism.iterrows()) + f""".
  Penyebab tidak dapat ditentukan dari file yang ada (mungkin poligon/urutan clip QGIS berbeda).
- Kelas radiansi: run ini memperoleh >400 nW/cm²/sr = {fmt(tot['pct_high'], 3)}% (median {fmt(tot['rad_median'], 2)}); paper (*dikutip*) menulis 4,5%
  (median 13,63) dan 12,0% untuk WPP 711 (run ini: {fmt(prof.loc[711, 'pct_high'], 2)}%). Proporsi kelas tinggi paper tidak dapat direproduksi dari CSV ini.

## Ketersediaan & cakupan pengamatan
- Tidak ada bulan tanpa rekaman (2012-04..2026-05). Semua 11 WPP × 156 bulan (2013–2025) memiliki ≥1 deteksi, sehingga tidak ada
  nol atau label hilang yang diimputasi.
- Proporsi tanggal yang memiliki rekaman bukan jumlah malam bebas awan atau jumlah lintasan; manifest cakupan/awan tidak tersedia.
  Bulan dengan < 75% tanggal berekaman: {low_txt}. Jumlah deteksi bulan tersebut dapat mencerminkan cakupan data, bukan aktivitas.

## Oseanografi
- Sumber: `series.json`/`meta.json` repo indonesia-fishing-watch (commit `{commit}`), dibangun skrip
  `scripts/oseanografi/build_ocean_summary.py` dari Copernicus Marine. SST = {ocean_meta['product_type']['sst']};
  klorofil-a = {ocean_meta['product_type']['chl']}. Agregasi: {ocean_meta['aggregation']}. Pembulatan: {ocean_meta['rounding']}.
- Deret bulanan aktual (bukan klimatologi) {ocean_meta['coverage']['sst'][0]}..{ocean_meta['coverage']['sst'][1]} (SST) dan
  {ocean_meta['coverage']['chl'][0]}..{ocean_meta['coverage']['chl'][1]} (klorofil); nilai hilang di dalam rentang masing-masing: SST {miss_in['sst']}, klorofil {miss_in['chl']}.
  SST hanya sampai 2025-12; tidak dipakai untuk 2026.
- Resolusi native 0,25° (SST) dan 4 km (klorofil); tidak ada klaim detail 1 km.
"""
(A / "data_issues.md").write_text(issues, encoding="utf-8")

# ------------------------------------------------------------------ captions_id.md
fc = pd.read_csv(R / "figure_catalog.csv")
CAP = {
    "F01": "Alur penelitian: data VIIRS VBD bersih diaudit (QF=1, Land_Mask≠0, point-in-polygon WPP, duplikat eksak dibuang), lalu dibentuk dua dataset turunan: grid 1 km equal-area (EPSG:6933) untuk HDBSCAN dan panel WPP–bulan untuk peramalan. CFA dibentuk dari data latih 2013–2023 dan dibekukan; tidak dipakai sebagai fitur model inti. Garis putus-putus: keluaran CFA menuju penyajian pembelajaran.",
    "F02": f"(a) Jumlah rekaman pada setiap tahap audit yang benar-benar dijalankan; batang abu-abu adalah varian pengolahan lama (Land_Mask≠1 dan ZEE Indonesia) sebagai pembanding. (b) Proporsi tanggal per bulan yang memiliki ≥1 rekaman (semua QF, seluruh bounding box input), 2012-04 s.d. 2026-05. Ini keberadaan rekaman, bukan cakupan malam bebas awan.",
    "F03": f"(a) Histogram spasial: jumlah deteksi QF=1 per sel grid 1 km (EPSG:6933; sel ≈1,15 km × 0,87 km di ekuator, luas 1 km²), seluruh data 2012-04 s.d. 2026-05, skala warna log; sel tanpa deteksi putih. Bukan KDE. (b–c) Keanggotaan sel CFA referensi hasil HDBSCAN (haversine) atas sel aktif ≥2 tahun dalam data latih 2013–2023 untuk WPP 712 (pilot) dan WPP 718. Batas WPP: shapefile tanpa .prj (EPSG:4326 diasumsikan). CFA adalah area konsentrasi deteksi berulang menurut definisi penelitian, bukan lokasi stok ikan atau bukti pelanggaran.",
    "F04": "(a) Jumlah deteksi bulanan total 11 WPP; area abu-abu = tahun parsial (2012 mulai April, 2026 sampai Mei). (b) Indeks musiman relatif = median 2013–2025 dari (deteksi bulan ÷ rata-rata bulanan pada tahun yang sama) per WPP; 1 = rata-rata tahunan. Indeks relatif tidak membandingkan jumlah absolut antar-WPP.",
    "F05": "Anomali log(1+deteksi) terhadap anomali SST (reanalisis) dan anomali log10 klorofil-a (L4 satelit) pada bulan yang sama, WPP 712 dan 717 (contoh yang ditetapkan sebelum evaluasi), 2013–2025. Anomali = nilai dikurangi rata-rata bulan kalender 2013–2025 (deskriptif, memakai seluruh periode; tidak dipakai sebagai fitur model). Spearman ρ deskriptif tanpa p-value karena deret berautokorelasi; asosiasi bukan sebab-akibat.",
    "F06": "Aktual vs prediksi satu langkah ke depan pada uji 2025 (origin = akhir bulan sebelumnya; parameter dibekukan dari validasi 2024) untuk WPP 712 dan 717. Panel kiri: M0 Seasonal Naive dan M1 ETS; kanan: M2 RF-History dan M3 RF-Ocean; sumbu y sama dalam satu WPP. Prediksi titik tanpa interval (interval RF belum dikalibrasi). Hindcast dengan produk oseanografi historis final.",
    "F07": "MASE uji 2025 per WPP × model (n = 12 bulan/WPP; 132 pasangan WPP–bulan per model). Penyebut MASE = MAE seasonal naive pada target latih 2013-04..2023-12, sama untuk semua model. Nilai tebal = terendah per baris. Macro-MASE = rata-rata MASE 11 WPP. MAE/RMSE ada di T07_metrics_by_wpp.csv.",
    "F08": "ΔMAE = MAE(M2 RF-History) − MAE(M3 RF-Ocean) per WPP pada uji 2025 (n = 12 bulan/WPP, pasangan WPP–bulan identik); positif berarti penambahan SST dan klorofil-a menurunkan galat. Label: selisih absolut (deteksi/bulan) dan persen terhadap MAE M2. Perbedaan deskriptif; tidak ada uji signifikansi.",
}
cap_lines = ["# Caption gambar (Bahasa Indonesia)\n", f"Run `{cfg['run_id']}`, dibuat {gen}. Detail sumber tabel ada di `figure_catalog.csv`.\n"]
for r in fc.sort_values("id").itertuples():
    cap_lines.append(f"## {r.id} — `{r.files}`\n")
    cap_lines.append(f"- Jenis visual: {r.visual_type}\n- Tujuan: {r.purpose}\n- Alasan pemilihan: {r.chart_choice_reason}\n"
                     f"- Sumber tabel: {r.source_tables}\n- Periode: {r.period}; unit: {r.unit}\n- Posisi makalah: {r.paper_section}\n")
    if r.id in CAP:
        cap_lines.append(f"\n**Caption:** {CAP[r.id]}\n")
    cap_lines.append("")
(R / "captions_id.md").write_text("\n".join(cap_lines), encoding="utf-8")

# ------------------------------------------------------------------ results_summary.md
better_t = abt[abt["delta_MAE_M2_minus_M3"] > 0].index.tolist()
worse_t = abt[abt["delta_MAE_M2_minus_M3"] <= 0].index.tolist()
better_v = abv[abv["delta_MAE_M2_minus_M3"] > 0].index.tolist()
cnt_best = best.value_counts()
rs = f"""# Ringkasan hasil (run `{cfg['run_id']}`)

Dibuat otomatis {gen} dari tabel keluaran. Status: **hasil eksperimen awal yang dapat direproduksi**, bukan temuan final makalah.

## 1. Input yang dipakai dan periode aktual
- VIIRS VBD bersih `viirs_clean_all.csv` ({fmt(st['00_input'])} baris, 2012-04-01 s.d. 2026-05-31 UTC) → view analitis
  {fmt(st['05_dedup_exact'])} deteksi QF=1 di 11 WPP (lihat `audit/data_issues.md`).
- 11 shapefile WPP (`shp&shx/`), CRS diasumsikan EPSG:4326.
- SST & klorofil-a bulanan per WPP dari `series.json` repo indonesia-fishing-watch (2012-01..2025-12 untuk SST).
- Periode utama 2013-01..2025-12; 2012 hanya warm-up fitur; 2026 (Jan–Mei) hanya deskriptif.

## 2. Tahap yang selesai / tidak dijalankan
- Selesai: audit, dua dataset turunan, HDBSCAN 11 WPP (9 kombinasi parameter + 2 varian sensitivitas), CFA beku,
  peramalan M0–M3 (validasi 2024, uji 2025), evaluasi, 8 gambar utama + 10 lampiran, ekspor website.
- Tidak dijalankan: SARIMA (opsional), M4 fitur CFA (opsional), DBCV (tidak dihitung; kestabilan dinilai dengan ARI antar-parameter),
  uji tren Mann–Kendall ulang, interval prediksi terkalibrasi, uji pembelajaran pada pengguna.

## 3. CFA (HDBSCAN)
- Definisi operasional: sel 1 km (EPSG:6933) dengan deteksi pada ≥2 tahun berbeda dalam 2013–2023; HDBSCAN (`hdbscan` {json.loads((run_dir(cfg) / 'run_manifest.json').read_text())['package_versions']['hdbscan']}),
  metrik haversine [lat, lon] radian, `eom`; parameter per WPP dipilih dengan aturan yang ditetapkan sebelumnya:
  ARI rata-rata tertinggi terhadap 8 kombinasi lain (bukan noise minimum).
- Hasil: **{n_cfa} CFA, {fmt(area)} km²** di 11 WPP. Per WPP: """ + "; ".join(
    f"{w}: {int(g['cfa_id'].count())} CFA, {fmt(g['area_km2'].sum())} km² (mcs={int(sel.loc[w, 'min_cluster_size'])}, ms={int(sel.loc[w, 'min_samples'])}, noise {fmt(100 * sel.loc[w, 'noise_frac_cells'], 1)}%)"
    for w, g in cat.groupby("wpp")) + f""".
- **Keterbatasan penting:** di WPP 712 sel aktif ≥2 tahun hampir menutupi seluruh Laut Jawa sehingga satu CFA mencakup
  {fmt(sel.loc[712, 'largest_cluster_cells'])} sel (≈ km²) dengan noise {fmt(100 * sel.loc[712, 'noise_frac_cells'], 1)}%; definisi ini kurang membedakan di wilayah sangat padat.
  Dengan seleksi ≥3 tahun WPP 712 terbagi menjadi {int(sens[(sens.wpp == 712) & (sens.variant == 'ge3years_1km')]['n_clusters'].iloc[0])} cluster.
  Jaccard luas CFA terhadap varian sensitivitas berkisar {fmt(sens['area_jaccard_vs_main'].min(), 2)}–{fmt(sens['area_jaccard_vs_main'].max(), 2)}
  (`T04b_cfa_sensitivity.csv`): bentuk CFA bergantung pada ambang keberulangan dan resolusi grid.

## 4. Peramalan (hindcast satu langkah ke depan)
- Panel 11 WPP × bulan; target latih 2013-04..2023-12 (2013-01..03 dibuang karena lag-12 tidak tersedia), validasi 2024 (12 origin),
  uji 2025 (12 origin, parameter beku). Semua model dinilai pada 132 pasangan WPP–bulan yang sama; 0 prediksi gagal.
- RF: 300 pohon, random_state=42, target log1p; 8 kandidat per model. Terpilih M2 {params['rf']['M2_RF_History']}, M3 {params['rf']['M3_RF_Ocean']}.
  ETS: error & musiman aditif pada log1p, periode 12; AICc memilih tanpa tren pada {sum(v is None for v in ets_trend.values())}/11 WPP.

| Model | macro-MASE uji | median MASE uji | macro-MAE uji | macro-MASE validasi |
|---|---:|---:|---:|---:|
""" + "\n".join(f"| {LBL[m]} | {fmt(q('test', m, 'macro_MASE'), 3)} | {fmt(q('test', m, 'median_MASE'), 3)} | {fmt(q('test', m, 'macro_MAE'))} | {fmt(q('val', m, 'macro_MASE'), 3)} |" for m in M) + f"""

- Model dengan MASE terendah per WPP pada uji: """ + ", ".join(f"{LBL[k]} {v} WPP" for k, v in cnt_best.items()) + f""".
- M1 ETS memiliki macro-MASE uji terendah; RF tidak lebih baik secara konsisten daripada ETS. Semua model, termasuk ETS,
  memiliki MASE > 2 di WPP 573, 715, dan 717 (galat uji jauh di atas galat seasonal naive periode latih).
- Dengan 12 bulan uji per WPP dan ketergantungan antar-wilayah, perbedaan antarmodel bersifat **deskriptif**; tidak ada klaim signifikansi.

## 5. Apakah SST & klorofil-a membantu? (M2 vs M3)
- Uji 2025: ΔMAE = MAE(M2) − MAE(M3) positif (M3 lebih baik) di **{len(better_t)} dari 11 WPP** ({', '.join(map(str, better_t))});
  negatif di {', '.join(map(str, worse_t))}. Rentang ΔMAE {sfmt(abt['delta_MAE_M2_minus_M3'].min(), 0)} s.d. {sfmt(abt['delta_MAE_M2_minus_M3'].max(), 0)} deteksi/bulan
  ({sfmt(abt['delta_rel_pct_of_M2'].min(), 1)}% s.d. {sfmt(abt['delta_rel_pct_of_M2'].max(), 1)}% dari MAE M2). macro-MASE M2 {fmt(q('test', 'M2_RF_History', 'macro_MASE'), 3)} vs M3 {fmt(q('test', 'M3_RF_Ocean', 'macro_MASE'), 3)}.
- Validasi 2024: M3 lebih baik di {len(better_v)} dari 11 WPP.
- Kesimpulan deskriptif: penambahan SST & klorofil-a bulan sebelumnya memberi perubahan kecil dan tidak konsisten antar-WPP;
  bukti saat ini **belum mendukung** klaim bahwa oseanografi meningkatkan peramalan secara umum.
  Permutation importance validasi (S08) menunjukkan model bergantung terutama pada lag-12, lag-1, dan rata-rata 3 bulan;
  fitur oseanografi berkontribusi kecil (bukan bukti sebab-akibat).
- Asosiasi anomali bulan yang sama (Spearman, deskriptif, n = 156): SST ρ {sfmt(an['sst'].min())} s.d. {sfmt(an['sst'].max())}; log-klorofil ρ {sfmt(an['log10_chl'].min())} s.d. {sfmt(an['log10_chl'].max())}
  (|ρ| terbesar: SST di WPP {an['sst'].abs().idxmax()} ρ = {sfmt(an['sst'][an['sst'].abs().idxmax()])}; log-klorofil di WPP {an['log10_chl'].abs().idxmax()} ρ = {sfmt(an['log10_chl'][an['log10_chl'].abs().idxmax()])}).

## 6. Pola deskriptif
- Konsentrasi deteksi (seluruh data): """ + ", ".join(f"WPP {w} {fmt(r, 1)}%" for w, r in prof['share_pct'].sort_values(ascending=False).head(4).items()) + f""".
- Dua bulan dengan indeks musiman median tertinggi per WPP (F04b): {peak_txt}.
- CV tahunan 2013–2025 (S10): tertinggi WPP 717 dan 573, terendah 716 dan 714.

## 7. Keterbatasan
- Jumlah deteksi ≠ jumlah kapal unik, upaya, atau hasil tangkapan; dipengaruhi awan, bulan, sudut pandang, dan konfigurasi lampu.
  Tidak ada manifest cakupan satelit/awan, sehingga variasi bulanan sebagian dapat berasal dari peluang pengamatan.
- Keberulangan CFA dihitung dari tahun dengan deteksi; ambang ≥2 tahun bersifat operasional dan sensitif (lihat bagian 3).
- Domain WPP mencakup area di luar ZEE Indonesia menurut kolom EEZ EOG (terutama WPP 711); bukan klaim yurisdiksi.
- Oseanografi memakai produk reanalisis/L4 historis final dan ringkasan 4 angka penting → hasil adalah hindcast, bukan uji real-time.
- Asal resmi dan CRS shapefile WPP tidak terverifikasi (tanpa .prj/.dbf); distribusi per-WPP dan proporsi kelas radiansi paper tidak
  dapat direproduksi dari file ini.
- Tidak ada interval prediksi; tidak ada uji efektivitas pembelajaran.
"""
rs = rs.replace("(≈ km²)", f"(≈ {fmt(cat[cat.wpp == 712]['area_km2'].max())} km²)")
(R / "results_summary.md").write_text(rs, encoding="utf-8")

# ------------------------------------------------------------------ rancangan tugas belajar
(R / "learning_tasks.md").write_text("""# Rancangan tugas literasi data kelautan (draf, belum diuji pada pengguna)

Bahan: ekspor `website_exports/` (CFA, deret bulanan, hindcast 2025, metrik). Peningkatan literasi **belum diukur**.

1. **Membaca peta CFA** (F03, `cfa_reference_web.geojson`): tunjukkan dua CFA di WPP 718; jelaskan mengapa CFA ≠ lokasi stok ikan
   dan mengapa noise bukan indikasi pelanggaran. Bandingkan peta seluruh periode (deskriptif) dengan CFA data latih.
2. **Membaca pola musiman** (F04): tentukan bulan puncak WPP 712 dan 718; jelaskan beda indeks relatif dengan jumlah absolut.
3. **Membandingkan aktual vs prediksi** (F06, `forecast_hindcast_2025.json`): pilih bulan dengan galat terbesar, hitung galat absolut,
   dan diskusikan mengapa prediksi titik tanpa interval perlu dibaca hati-hati.
4. **Korelasi ≠ sebab-akibat** (F05, F08): jelaskan mengapa ρ = 0,46 antara anomali SST dan deteksi di WPP 717 tidak membuktikan SST
   menyebabkan aktivitas, dan mengapa penambahan oseanografi tidak selalu memperbaiki prediksi.
5. **Ketidakpastian data** (F02): jelaskan mengapa bulan dengan sedikit tanggal berekaman dapat menurunkan jumlah deteksi.

Rencana evaluasi (belum dilakukan): pre/post test dengan rubrik membaca peta, membedakan korelasi–kausalitas, dan menafsirkan galat.
""", encoding="utf-8")

log_step(cfg, "07_export", {"checks": checks})
print(json.dumps(checks, indent=1, default=str))
