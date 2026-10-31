# Caption gambar (Bahasa Indonesia)

Run `run_v1`, dibuat 2026-10-18T05:43:03.601359+00:00. Detail sumber tabel ada di `figure_catalog.csv`.

## F01 — `fig01_workflow.png;fig01_workflow.pdf;fig01_workflow.svg`

- Jenis visual: diagram alur (vektor)
- Tujuan: Alur data bersih → dua dataset → HDBSCAN & regresi → evaluasi → pembelajaran
- Alasan pemilihan: Diagram menunjukkan percabangan proses yang sulit dijelaskan dengan chart numerik
- Sumber tabel: konfigurasi & skrip
- Periode: -; unit: -
- Posisi makalah: Metodologi / arsitektur solusi


**Caption:** Alur penelitian: data VIIRS VBD bersih diaudit (QF=1, Land_Mask≠0, point-in-polygon WPP, duplikat eksak dibuang), lalu dibentuk dua dataset turunan: grid 1 km equal-area (EPSG:6933) untuk HDBSCAN dan panel WPP–bulan untuk peramalan. CFA dibentuk dari data latih 2013–2023 dan dibekukan; tidak dipakai sebagai fitur model inti. Garis putus-putus: keluaran CFA menuju penyajian pembelajaran.


## F02 — `fig02_data_audit.png;fig02_data_audit.pdf`

- Jenis visual: batang horizontal + heatmap kalender
- Tujuan: Jumlah rekaman per tahap audit; keberadaan rekaman per bulan
- Alasan pemilihan: Batang membandingkan antar-tahap; heatmap memperlihatkan pola bolong menurut waktu
- Sumber tabel: audit/audit_counts.csv; audit/monthly_record_availability.csv
- Periode: 2012-04..2026-05; unit: rekaman; proporsi tanggal
- Posisi makalah: Dataset dan hasil preprocessing


**Caption:** (a) Jumlah rekaman pada setiap tahap audit yang benar-benar dijalankan; batang abu-abu adalah varian pengolahan lama (Land_Mask≠1 dan ZEE Indonesia) sebagai pembanding. (b) Proporsi tanggal per bulan yang memiliki ≥1 rekaman (semua QF, seluruh bounding box input), 2012-04 s.d. 2026-05. Ini keberadaan rekaman, bukan cakupan malam bebas awan.


## F03 — `fig03_spatial_cfa.png;fig03_spatial_cfa.pdf`

- Jenis visual: peta grid (histogram spasial) + inset CFA
- Tujuan: Lokasi konsentrasi deteksi dan batas CFA referensi
- Alasan pemilihan: Lokasi dan bentuk area memerlukan peta; grid menghindari jutaan titik saling menutup (bukan KDE)
- Sumber tabel: data/grid_all_1000m.parquet; data/grid_train_1000m.parquet; data/cfa_reference_6933.parquet
- Periode: a: 2012-04..2026-05; b-c: 2013-2023; unit: deteksi per sel 1 km²
- Posisi makalah: Pola spasial dan CFA


**Caption:** (a) Histogram spasial: jumlah deteksi QF=1 per sel grid 1 km (EPSG:6933; sel ≈1,15 km × 0,87 km di ekuator, luas 1 km²), seluruh data 2012-04 s.d. 2026-05, skala warna log; sel tanpa deteksi putih. Bukan KDE. (b–c) Keanggotaan sel CFA referensi hasil HDBSCAN (haversine) atas sel aktif ≥2 tahun dalam data latih 2013–2023 untuk WPP 712 (pilot) dan WPP 718. Batas WPP: shapefile tanpa .prj (EPSG:4326 diasumsikan). CFA adalah area konsentrasi deteksi berulang menurut definisi penelitian, bukan lokasi stok ikan atau bukti pelanggaran.


## F04 — `fig04_temporal_seasonality.png;fig04_temporal_seasonality.pdf`

- Jenis visual: line chart + heatmap WPP × bulan
- Tujuan: Perubahan bulanan dan waktu puncak musiman antar-WPP
- Alasan pemilihan: Line chart untuk urutan waktu; heatmap membandingkan pola bulan antar-WPP (indeks relatif, bukan jumlah absolut)
- Sumber tabel: data/wpp_month_panel.parquet; tables/F04_seasonal_index_median.csv
- Periode: a: 2012-04..2026-05; b: 2013-2025; unit: deteksi/bulan; indeks tanpa satuan
- Posisi makalah: Dinamika temporal dan musiman


**Caption:** (a) Jumlah deteksi bulanan total 11 WPP; area abu-abu = tahun parsial (2012 mulai April, 2026 sampai Mei). (b) Indeks musiman relatif = median 2013–2025 dari (deteksi bulan ÷ rata-rata bulanan pada tahun yang sama) per WPP; 1 = rata-rata tahunan. Indeks relatif tidak membandingkan jumlah absolut antar-WPP.


## F05 — `fig05_ocean_association.png;fig05_ocean_association.pdf`

- Jenis visual: scatter anomali
- Tujuan: Asosiasi deskriptif anomali deteksi dengan SST dan klorofil-a (bulan yang sama)
- Alasan pemilihan: Scatter menampilkan hubungan dan sebaran; anomali (dikurangi rata-rata bulan kalender 2013-2025) memisahkan pola musim bersama
- Sumber tabel: data/ocean_association_frame.parquet; tables/T09_ocean_association_spearman.csv
- Periode: 2013-01..2025-12; unit: log(1+deteksi); °C; log10 mg/m³
- Posisi makalah: Hubungan dengan oseanografi


**Caption:** Anomali log(1+deteksi) terhadap anomali SST (reanalisis) dan anomali log10 klorofil-a (L4 satelit) pada bulan yang sama, WPP 712 dan 717 (contoh yang ditetapkan sebelum evaluasi), 2013–2025. Anomali = nilai dikurangi rata-rata bulan kalender 2013–2025 (deskriptif, memakai seluruh periode; tidak dipakai sebagai fitur model). Spearman ρ deskriptif tanpa p-value karena deret berautokorelasi; asosiasi bukan sebab-akibat.


## F06 — `fig06_forecast_test.png;fig06_forecast_test.pdf`

- Jenis visual: small multiples line chart
- Tujuan: Aktual vs prediksi satu langkah ke depan 2025 untuk WPP contoh
- Alasan pemilihan: Line chart mempertahankan urutan waktu dan memperlihatkan kapan prediksi meleset; dua panel per WPP agar 4 model terbaca
- Sumber tabel: tables/T06_predictions.csv
- Periode: 2025-01..2025-12 (uji); unit: deteksi per bulan
- Posisi makalah: Hasil peramalan


**Caption:** Aktual vs prediksi satu langkah ke depan pada uji 2025 (origin = akhir bulan sebelumnya; parameter dibekukan dari validasi 2024) untuk WPP 712 dan 717. Panel kiri: M0 Seasonal Naive dan M1 ETS; kanan: M2 RF-History dan M3 RF-Ocean; sumbu y sama dalam satu WPP. Prediksi titik tanpa interval (interval RF belum dikalibrasi). Hindcast dengan produk oseanografi historis final.


## F07 — `fig07_model_comparison.png;fig07_model_comparison.pdf`

- Jenis visual: heatmap beranotasi
- Tujuan: Perbandingan MASE 4 model di 11 WPP pada uji 2025
- Alasan pemilihan: Heatmap membandingkan banyak model × wilayah tanpa batang berdesakan; tabel MAE/RMSE terpisah (T07)
- Sumber tabel: tables/T07_metrics_by_wpp.csv; tables/T07b_model_summary.csv
- Periode: 2025-01..2025-12; unit: MASE (tanpa satuan)
- Posisi makalah: Evaluasi model


**Caption:** MASE uji 2025 per WPP × model (n = 12 bulan/WPP; 132 pasangan WPP–bulan per model). Penyebut MASE = MAE seasonal naive pada target latih 2013-04..2023-12, sama untuk semua model. Nilai tebal = terendah per baris. Macro-MASE = rata-rata MASE 11 WPP. MAE/RMSE ada di T07_metrics_by_wpp.csv.


## F08 — `fig08_ocean_ablation.png;fig08_ocean_ablation.pdf`

- Jenis visual: lollipop divergen
- Tujuan: Apakah penambahan oseanografi menurunkan MAE per WPP
- Alasan pemilihan: Posisi terhadap nol langsung menjawab membantu/merugikan; label persen relatif terhadap MAE M2
- Sumber tabel: tables/T08_ocean_ablation_M2_vs_M3.csv
- Periode: 2025-01..2025-12; unit: deteksi per bulan
- Posisi makalah: Pengujian kontribusi oseanografi


**Caption:** ΔMAE = MAE(M2 RF-History) − MAE(M3 RF-Ocean) per WPP pada uji 2025 (n = 12 bulan/WPP, pasangan WPP–bulan identik); positif berarti penambahan SST dan klorofil-a menurunkan galat. Label: selisih absolut (deteksi/bulan) dan persen terhadap MAE M2. Perbedaan deskriptif; tidak ada uji signifikansi.


## S01 — `S01_radiance_hist.png;S01_radiance_hist.pdf`

- Jenis visual: histogram sumbu x log
- Tujuan: Distribusi radiansi view analitis
- Alasan pemilihan: Distribusi menceng kanan
- Sumber tabel: tables/S01_radiance_hist.csv
- Periode: 2012-04..2026-05; unit: nW/cm²/sr
- Posisi makalah: Lampiran


## S02 — `S02_radiance_box_by_wpp.png;S02_radiance_box_by_wpp.pdf`

- Jenis visual: boxplot (kuantil) skala log
- Tujuan: Membandingkan median dan rentang radiansi antar-WPP
- Alasan pemilihan: Lebih ringkas daripada violin; whisker = P5-P95 (bukan aturan 1.5 IQR)
- Sumber tabel: tables/S02_radiance_quantiles_by_wpp.csv
- Periode: 2012-04..2026-05; unit: nW/cm²/sr
- Posisi makalah: Lampiran


## S03 — `S03_hdbscan_param_sensitivity.png;S03_hdbscan_param_sensitivity.pdf`

- Jenis visual: heatmap parameter × WPP
- Tujuan: Bentuk/jumlah CFA bergantung pada parameter; noise minimum bukan kriteria
- Alasan pemilihan: Heatmap membandingkan 9 kombinasi × 11 WPP
- Sumber tabel: tables/T04a_hdbscan_param_runs.csv
- Periode: latih 2013-2023; unit: -
- Posisi makalah: Lampiran


## S03b — `S03b_cfa_reference_map.png;S03b_cfa_reference_map.pdf`

- Jenis visual: peta poligon
- Tujuan: Sebaran CFA referensi semua WPP (data latih)
- Alasan pemilihan: Lokasi memerlukan peta
- Sumber tabel: data/cfa_reference_6933.parquet; tables/cfa_catalog.csv
- Periode: 2013-2023; unit: km²
- Posisi makalah: Lampiran


## S04 — `S04_forecast_all_wpp.png;S04_forecast_all_wpp.pdf`

- Jenis visual: small multiples
- Tujuan: Aktual-prediksi seluruh 11 WPP (tidak hanya yang sukses)
- Alasan pemilihan: Sumbu y per panel; tinggi panel tidak sebanding sebagai jumlah absolut
- Sumber tabel: tables/T06_predictions.csv
- Periode: 2025; unit: deteksi per bulan
- Posisi makalah: Lampiran


## S05 — `S05_residuals.png;S05_residuals.pdf`

- Jenis visual: line + scatter residual
- Tujuan: Bias dan pola galat menurut waktu dan besar prediksi
- Alasan pemilihan: Residual relatif agar WPP berskala besar tidak mendominasi
- Sumber tabel: tables/T06_predictions.csv
- Periode: 2025; unit: %
- Posisi makalah: Lampiran


## S06 — `S06_error_distribution.png;S06_error_distribution.pdf`

- Jenis visual: paired dot plot
- Tujuan: Variasi galat antar-WPP per model
- Alasan pemilihan: Menampilkan semua titik; tidak memakai violin
- Sumber tabel: tables/T07_metrics_by_wpp.csv
- Periode: 2025; unit: MASE
- Posisi makalah: Lampiran


## S07 — `S07_annual_counts.png;S07_annual_counts.pdf`

- Jenis visual: small multiples line
- Tujuan: Deret tahunan 2013-2025 (tahun penuh)
- Alasan pemilihan: Deskriptif; uji tren inferensial tidak dijalankan ulang
- Sumber tabel: tables/S07_annual_counts_2013_2025.csv
- Periode: 2013-2025; unit: deteksi/tahun
- Posisi makalah: Lampiran


## S08 — `S08_permutation_importance.png;S08_permutation_importance.pdf`

- Jenis visual: batang horizontal + sd
- Tujuan: Ketergantungan model pada fitur (bukan sebab-akibat)
- Alasan pemilihan: Pengacakan lintas baris panel; fitur berkorelasi berbagi pentingnya; dummy WPP tidak ditampilkan
- Sumber tabel: tables/S08_permutation_importance_val.csv
- Periode: validasi 2024; unit: MAE skala log1p
- Posisi makalah: Lampiran


## S09 — `S09_wpp_profile.png;S09_wpp_profile.pdf`

- Jenis visual: batang + stacked bar
- Tujuan: Profil jumlah deteksi dan kelas radiansi per WPP
- Alasan pemilihan: Kelas radiansi, bukan kelas ukuran kapal
- Sumber tabel: tables/S09_wpp_radiance_profile.csv
- Periode: 2012-04..2026-05; unit: deteksi; %
- Posisi makalah: Lampiran


## S10 — `S10_interannual_cv.png;S10_interannual_cv.pdf`

- Jenis visual: dot plot
- Tujuan: Variasi antartahun per WPP
- Alasan pemilihan: Cutoff CV bukan standar universal; mean/SD di tabel
- Sumber tabel: tables/S10_interannual_cv.csv
- Periode: 2013-2025; unit: %
- Posisi makalah: Lampiran

