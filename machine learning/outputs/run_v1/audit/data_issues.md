# Catatan masalah data & keputusan (run `run_v1`)

Dibuat otomatis 2026-10-18T05:43:03.601359+00:00. Angka **dihitung dari input saat ini** kecuali yang diberi label *dikutip*.

## Input
- `viirs_clean_all.csv`: 14.211.269 baris, 15 kolom, waktu 2012-04-01 15:46:46 s.d. 2026-05-31 19:30:40 UTC.
  Paper (*dikutip*) menyebut cakupan "2012–Juli 2026"; isi data berakhir **31 Mei 2026**.
- Jumlah baris sama persis dengan angka *dikutip* Tabel 2 paper untuk "Intermediate Data (Post-Cleaning)" (14.211.269), sehingga file ini
  adalah data perantara **sebelum** filter QF, daratan, dan WPP — bukan dataset analitis 7,38 juta.
- Kolom: Lat_DNB, Lon_DNB, Rad_DNB, Date_Mscan, QF_Detect, QF_Bitflag, Land_Mask, EEZ, MPA, Rad_I04, Glint, Date, Year, Month, Day. Tidak ada kolom `WPP_RI`; WPP ditentukan ulang secara point-in-polygon.
- Bounding box input: lat -14.00..8.00, lon 92.00..141.50
  (memuat ZEE negara tetangga).

## Kualitas & kode
- QF_Detect: 1 = 11.164.250, 2 = 2.511.375, 4 = 535.644. Readme EOG V23
  (`references/eog_vbd_readme_v23_r20180824.xlsx`): 1 strong detection, 2 weak, 4 gas flare. QF bukan label fishing/transit.
- Land_Mask: 1 = 210.928, 3 = 14.000.341. Readme V23: *Land=0, Water=3, Near-shore=1,2*.
  **Tidak ada kode 0 (daratan) di input.** Pengolahan lama/website menerapkan `Land_Mask != 1` dengan komentar "bukan daratan";
  menurut V23 itu membuang deteksi **near-shore**, bukan daratan. View utama mempertahankan near-shore
  (28.271 deteksi di dalam WPP) dan melaporkan varian lama sebagai pembanding.
- Fill value 999999 pada kolom inti: 0. Kolom diagnostik di luar rentang: Glint > 1 sebanyak 1.571
  (maks 3.9e+14); Rad_I04 ≤ −999 sebanyak 17.369.
  Kolom ini tidak dipakai analisis; tidak ada baris yang dibuang karenanya.
- Radiansi ≤ 0: QF 2: 90 (tidak ada pada QF=1). Tidak ada pembuangan outlier IQR/z-score.
- Baris dengan waktu/koordinat tidak terbaca: 1.
- Duplikat eksak (waktu-scan, lat, lon, radiansi, QF identik): 22.781 pada seluruh input; 9.692
  dibuang dari view utama. Koordinat sama pada malam berbeda tetap dipertahankan.
- Kolom turunan lama Date/Year/Month konsisten dengan Date_Mscan (0 ketidakcocokan).
- Waktu: semua lintasan jatuh pada jam 14–20 UTC, sehingga tanggal UTC dipakai sebagai identitas malam (satu malam = satu tanggal UTC).

## Tahap view analitis (berurutan)
| Tahap | Rekaman | Pembanding *dikutip* |
|---|---:|---:|
| 00_input — Semua baris viirs_clean_all.csv | 14.211.269 | 14.211.269 |
| 01_valid — Waktu, koordinat, radiansi terbaca; bukan fill 999999; rentang koordinat valid | 14.211.268 | 14.211.268 |
| 02_qf1 — QF_Detect = 1 (strong/boat detection, V23) | 11.164.249 | 11.164.249 |
| 03_not_land — Land_Mask <> 0 (kode daratan V23); near-shore 1/2 dipertahankan | 11.164.249 |  |
| 04_in_wpp — Di dalam salah satu dari 11 poligon WPP (point-in-polygon) | 7.654.423 |  |
| 05_dedup_exact — Buang duplikat eksak (Date_Mscan, Lat, Lon, Rad_DNB, QF sama) | 7.644.731 |  |
| L03_land_mask_ne_1 — Land_Mask != 1 (perlakuan lama; 1 = near-shore di V23) | 11.098.366 | 11.098.366 |
| L04_eez_indonesia — EEZ = Indonesian EEZ | 7.379.853 | 7.379.853 |
| L05_in_wpp — Di dalam 11 poligon WPP (tanpa dedup) | 7.375.153 | 7.375.153 |

Varian "seperti lama" (`Land_Mask != 1` + ZEE Indonesia + WPP) mereproduksi **persis** angka `preprocessing.json` repo website
(7.375.153), sehingga penentuan WPP run ini konsisten dengan pengolahan website. Paper (*dikutip*) menulis 7.380.357.

## WPP & batas
- Shapefile hanya `.shp/.shx` (tanpa `.dbf`, `.prj`). Kode WPP diambil dari nama file; CRS **diasumsikan** EPSG:4326 dari rentang koordinat.
  Asal resmi (DJPT-KKP, menurut paper) **tidak dapat diverifikasi** dari file.
- Geometri tidak valid: WPP 573 dan 714 (diperbaiki `make_valid`). Titik di >1 poligon: 0 (aturan cadangan: kode WPP terkecil).
- Domain utama = poligon WPP **tanpa filter ZEE**, sama dengan domain ringkasan oseanografi. Akibatnya
  253.925 deteksi di dalam poligon WPP berada di luar ZEE Indonesia menurut kolom EEZ EOG,
  terbanyak di WPP 711 (200.264; terutama area yang diberi label ZEE Vietnam/klaim tumpang tindih).
  Ini perbedaan cakupan, bukan klaim yurisdiksi.

## Ketidaksesuaian dengan paper (tidak disesuaikan)
- Jumlah per WPP Tabel 5 paper (*dikutip*) **tidak cocok** dengan point-in-polygon pada shapefile ini, walaupun totalnya mirip.
  Contoh (varian lama vs paper): WPP 571: 322.383 vs 432.304; WPP 572: 410.402 vs 196.361; WPP 573: 293.861 vs 409.597; WPP 711: 1.070.884 vs 532.758; WPP 712: 3.446.812 vs 3.887.263; WPP 713: 772.675 vs 826.975; WPP 714: 143.287 vs 20.828; WPP 715: 56.460 vs 70.741; WPP 716: 152.218 vs 153.085; WPP 717: 23.542 vs 37.160; WPP 718: 682.629 vs 684.256.
  Penyebab tidak dapat ditentukan dari file yang ada (mungkin poligon/urutan clip QGIS berbeda).
- Kelas radiansi: run ini memperoleh >400 nW/cm²/sr = 0,043% (median 13,47); paper (*dikutip*) menulis 4,5%
  (median 13,63) dan 12,0% untuk WPP 711 (run ini: 0,08%). Proporsi kelas tinggi paper tidak dapat direproduksi dari CSV ini.

## Ketersediaan & cakupan pengamatan
- Tidak ada bulan tanpa rekaman (2012-04..2026-05). Semua 11 WPP × 156 bulan (2013–2025) memiliki ≥1 deteksi, sehingga tidak ada
  nol atau label hilang yang diimputasi.
- Proporsi tanggal yang memiliki rekaman bukan jumlah malam bebas awan atau jumlah lintasan; manifest cakupan/awan tidak tersedia.
  Bulan dengan < 75% tanggal berekaman: 2014-01 (23% tanggal), 2022-07 (45% tanggal), 2022-08 (68% tanggal), 2024-07 (52% tanggal), 2025-11 (57% tanggal). Jumlah deteksi bulan tersebut dapat mencerminkan cakupan data, bukan aktivitas.

## Oseanografi
- Sumber: `series.json`/`meta.json` repo indonesia-fishing-watch (commit `0bacaf492f22798fb111628eb88c79ffafb6359c`), dibangun skrip
  `scripts/oseanografi/build_ocean_summary.py` dari Copernicus Marine. SST = reanalysis ensemble mean (model), bukan pengukuran langsung;
  klorofil-a = satellite ocean-colour L4 multi-sensor gap-filled (observasi terinterpolasi). Agregasi: rata-rata berbobot cos(lat) sel di dalam poligon WPP; klorofil rata-rata geometrik (log10). Pembulatan: 4 angka penting (sig(x, 4)) di series.json.
- Deret bulanan aktual (bukan klimatologi) 2012-01-01..2025-12-01 (SST) dan
  2012-01-01..2026-09-01 (klorofil); nilai hilang di dalam rentang masing-masing: SST 0, klorofil 0.
  SST hanya sampai 2025-12; tidak dipakai untuk 2026.
- Resolusi native 0,25° (SST) dan 4 km (klorofil); tidak ada klaim detail 1 km.
