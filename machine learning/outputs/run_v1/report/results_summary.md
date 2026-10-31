# Ringkasan hasil (run `run_v1`)

Dibuat otomatis 2026-10-18T05:43:03.601359+00:00 dari tabel keluaran. Status: **hasil eksperimen awal yang dapat direproduksi**, bukan temuan final makalah.

## 1. Input yang dipakai dan periode aktual
- VIIRS VBD bersih `viirs_clean_all.csv` (14.211.269 baris, 2012-04-01 s.d. 2026-05-31 UTC) → view analitis
  7.644.731 deteksi QF=1 di 11 WPP (lihat `audit/data_issues.md`).
- 11 shapefile WPP (`shp&shx/`), CRS diasumsikan EPSG:4326.
- SST & klorofil-a bulanan per WPP dari `series.json` repo indonesia-fishing-watch (2012-01..2025-12 untuk SST).
- Periode utama 2013-01..2025-12; 2012 hanya warm-up fitur; 2026 (Jan–Mei) hanya deskriptif.

## 2. Tahap yang selesai / tidak dijalankan
- Selesai: audit, dua dataset turunan, HDBSCAN 11 WPP (9 kombinasi parameter + 2 varian sensitivitas), CFA beku,
  peramalan M0–M3 (validasi 2024, uji 2025), evaluasi, 8 gambar utama + 10 lampiran, ekspor website.
- Tidak dijalankan: SARIMA (opsional), M4 fitur CFA (opsional), DBCV (tidak dihitung; kestabilan dinilai dengan ARI antar-parameter),
  uji tren Mann–Kendall ulang, interval prediksi terkalibrasi, uji pembelajaran pada pengguna.

## 3. CFA (HDBSCAN)
- Definisi operasional: sel 1 km (EPSG:6933) dengan deteksi pada ≥2 tahun berbeda dalam 2013–2023; HDBSCAN (`hdbscan` 0.8.44),
  metrik haversine [lat, lon] radian, `eom`; parameter per WPP dipilih dengan aturan yang ditetapkan sebelumnya:
  ARI rata-rata tertinggi terhadap 8 kombinasi lain (bukan noise minimum).
- Hasil: **322 CFA, 542.184 km²** di 11 WPP. Per WPP: 571: 13 CFA, 24.239 km² (mcs=50, ms=10, noise 5,0%); 572: 46 CFA, 28.152 km² (mcs=50, ms=10, noise 14,2%); 573: 46 CFA, 19.726 km² (mcs=50, ms=10, noise 24,0%); 711: 95 CFA, 112.610 km² (mcs=50, ms=10, noise 16,5%); 712: 9 CFA, 244.428 km² (mcs=100, ms=20, noise 0,3%); 713: 43 CFA, 44.496 km² (mcs=50, ms=10, noise 2,3%); 714: 24 CFA, 6.233 km² (mcs=50, ms=5, noise 5,5%); 715: 21 CFA, 2.432 km² (mcs=50, ms=10, noise 17,1%); 716: 12 CFA, 4.401 km² (mcs=50, ms=20, noise 6,1%); 717: 6 CFA, 1.003 km² (mcs=50, ms=10, noise 9,1%); 718: 7 CFA, 54.464 km² (mcs=100, ms=5, noise 1,1%).
- **Keterbatasan penting:** di WPP 712 sel aktif ≥2 tahun hampir menutupi seluruh Laut Jawa sehingga satu CFA mencakup
  233.181 sel (≈ 233.181 km²) dengan noise 0,3%; definisi ini kurang membedakan di wilayah sangat padat.
  Dengan seleksi ≥3 tahun WPP 712 terbagi menjadi 46 cluster.
  Jaccard luas CFA terhadap varian sensitivitas berkisar 0,25–0,75
  (`T04b_cfa_sensitivity.csv`): bentuk CFA bergantung pada ambang keberulangan dan resolusi grid.

## 4. Peramalan (hindcast satu langkah ke depan)
- Panel 11 WPP × bulan; target latih 2013-04..2023-12 (2013-01..03 dibuang karena lag-12 tidak tersedia), validasi 2024 (12 origin),
  uji 2025 (12 origin, parameter beku). Semua model dinilai pada 132 pasangan WPP–bulan yang sama; 0 prediksi gagal.
- RF: 300 pohon, random_state=42, target log1p; 8 kandidat per model. Terpilih M2 {'max_depth': 12, 'min_samples_leaf': 1, 'max_features': 0.5}, M3 {'max_depth': 12, 'min_samples_leaf': 5, 'max_features': 0.5}.
  ETS: error & musiman aditif pada log1p, periode 12; AICc memilih tanpa tren pada 11/11 WPP.

| Model | macro-MASE uji | median MASE uji | macro-MAE uji | macro-MASE validasi |
|---|---:|---:|---:|---:|
| M0 Seasonal Naive | 1,526 | 0,984 | 1.284 | 1,546 |
| M1 ETS | 1,192 | 0,948 | 1.066 | 1,069 |
| M2 RF-History | 1,252 | 0,862 | 1.073 | 1,097 |
| M3 RF-Ocean | 1,244 | 0,807 | 1.065 | 1,090 |

- Model dengan MASE terendah per WPP pada uji: M1 ETS 6 WPP, M3 RF-Ocean 3 WPP, M2 RF-History 2 WPP.
- M1 ETS memiliki macro-MASE uji terendah; RF tidak lebih baik secara konsisten daripada ETS. Semua model, termasuk ETS,
  memiliki MASE > 2 di WPP 573, 715, dan 717 (galat uji jauh di atas galat seasonal naive periode latih).
- Dengan 12 bulan uji per WPP dan ketergantungan antar-wilayah, perbedaan antarmodel bersifat **deskriptif**; tidak ada klaim signifikansi.

## 5. Apakah SST & klorofil-a membantu? (M2 vs M3)
- Uji 2025: ΔMAE = MAE(M2) − MAE(M3) positif (M3 lebih baik) di **6 dari 11 WPP** (571, 712, 714, 715, 716, 718);
  negatif di 572, 573, 711, 713, 717. Rentang ΔMAE -42 s.d. +92 deteksi/bulan
  (-5,1% s.d. +14,2% dari MAE M2). macro-MASE M2 1,252 vs M3 1,244.
- Validasi 2024: M3 lebih baik di 8 dari 11 WPP.
- Kesimpulan deskriptif: penambahan SST & klorofil-a bulan sebelumnya memberi perubahan kecil dan tidak konsisten antar-WPP;
  bukti saat ini **belum mendukung** klaim bahwa oseanografi meningkatkan peramalan secara umum.
  Permutation importance validasi (S08) menunjukkan model bergantung terutama pada lag-12, lag-1, dan rata-rata 3 bulan;
  fitur oseanografi berkontribusi kecil (bukan bukti sebab-akibat).
- Asosiasi anomali bulan yang sama (Spearman, deskriptif, n = 156): SST ρ -0,17 s.d. +0,46; log-klorofil ρ -0,36 s.d. +0,34
  (|ρ| terbesar: SST di WPP 717 ρ = +0,46; log-klorofil di WPP 713 ρ = -0,36).

## 6. Pola deskriptif
- Konsentrasi deteksi (seluruh data): WPP 712 45,1%, WPP 711 16,6%, WPP 713 10,2%, WPP 718 8,9%.
- Dua bulan dengan indeks musiman median tertinggi per WPP (F04b): 571: Mar 1,63, Feb 1,35; 572: Mei 1,36, Jun 1,36; 573: Sep 1,47, Agu 1,41; 711: Mei 1,64, Apr 1,44; 712: Sep 1,54, Agu 1,51; 713: Sep 1,43, Agu 1,37; 714: Okt 1,75, Nov 1,52; 715: Okt 1,33, Nov 1,30; 716: Nov 1,20, Okt 1,20; 717: Nov 1,41, Okt 1,34; 718: Okt 1,85, Nov 1,74.
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
