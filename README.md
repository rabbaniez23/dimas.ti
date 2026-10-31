# DIMAS-TI 2026 — Indonesia Fishing Watch

Penambangan data deteksi kapal malam hari **VIIRS Boat Detection (VBD)** di 11 Wilayah Pengelolaan Perikanan (WPP-RI):

- **Area Penangkapan Ikan Kolektif (CFA)** dengan klasterisasi **HDBSCAN** pada grid 1 km / 2 km (EPSG:6933).
- **Peramalan jumlah deteksi bulanan per WPP** dengan **Random Forest** (M0–M3, termasuk variabel oseanografi), validasi 2024 dan uji 2025.

## Struktur repositori

```
machine learning/
├── config/research.yaml        # path, periode, aturan kualitas, parameter, seed
├── src/                        # modul: ingest, audit, spatial, ocean, features, clustering, forecast, evaluation, plotting
├── scripts/                    # pipeline 01–07
├── references/                 # kamus kolom resmi EOG VBD V23
├── shp&shx/                    # batas 11 WPP-RI (.shp/.shx)
├── outputs/run_v1/
│   ├── audit/                  # audit data, kamus data, catatan isu
│   ├── tables/                 # tabel hasil (T03–T08, S01–S10)
│   ├── figures/main|supplement # gambar utama F01–F08 dan lampiran S01–S10
│   ├── models/                 # parameter terpilih & log
│   ├── website_exports/        # JSON/GeoJSON untuk dasbor web
│   ├── report/                 # ringkasan hasil, keterangan gambar, katalog gambar
│   └── run_manifest.json       # versi paket, konfigurasi, checksum input, hasil pemeriksaan
└── requirements.txt
visualisasi/
└── model komputasi.drawio      # diagram model komputasi
```

## Menjalankan ulang

Input utama `viirs_clean_all.csv` (±2 GB, hasil pembersihan VIIRS VBD) tidak disertakan karena ukurannya.
Letakkan di folder `machine learning/`, lalu:

```bash
cd "machine learning"
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# data oseanografi (ringkasan JSON dari dasbor web)
git clone --depth 1 https://github.com/rabbaniez23/indonesia-fishing-watch external/indonesia-fishing-watch

.venv/Scripts/python.exe scripts/01_audit.py      # ingest → Parquet, penentuan WPP, audit
.venv/Scripts/python.exe scripts/02_datasets.py   # grid 1 km/2 km, panel WPP–bulan, tabel oseanografi
.venv/Scripts/python.exe scripts/03_cfa.py        # HDBSCAN 11 WPP × 9 parameter + sensitivitas
.venv/Scripts/python.exe scripts/04_forecast.py   # model M0–M3, validasi 2024, uji 2025
.venv/Scripts/python.exe scripts/05_evaluate.py   # metrik, ablasi, asosiasi oseanografi
.venv/Scripts/python.exe scripts/06_figures.py    # gambar F01–F08, S01–S10
.venv/Scripts/python.exe scripts/07_export.py     # pemeriksaan akhir, ekspor web, laporan
```

File antara berukuran besar (`outputs/_cache/`, `outputs/run_v1/data/*.parquet`, `outputs/run_v1/models/hdbscan_runs/`)
dibuat ulang otomatis oleh pipeline.

## Repositori terkait

- Dasbor web: [rabbaniez23/indonesia-fishing-watch](https://github.com/rabbaniez23/indonesia-fishing-watch)
- Analisis eksploratif awal (fase 0–8): [rabbaniez23/riset_anjay](https://github.com/rabbaniez23/riset_anjay)
