"""Tahap 6: gambar utama (F01-F08) dan lampiran (S01-S10) dari tabel keluaran + katalog & caption."""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from pyproj import Transformer  # noqa: E402

from src import plotting as P  # noqa: E402
from src.common import load_config, log_step, run_dir  # noqa: E402
from src.spatial import wpp_simplified  # noqa: E402

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
D, T = run_dir(cfg, "data"), run_dir(cfg, "tables")
A = run_dir(cfg, "audit")
MAIN, SUPP = run_dir(cfg, "figures/main"), run_dir(cfg, "figures/supplement")
CRS = cfg["spatial"]["grid_crs"]
SIZE = cfg["spatial"]["grid_size_m"]
EX_WPP = [712, 717]  # contoh ditetapkan di awal (pilot HDBSCAN & WPP kontras paper), bukan dipilih dari skor
P.setup()
CATALOG = []


def reg(fid, files, kind, purpose, why, source, period, unit, section):
    CATALOG.append(dict(id=fid, files=";".join(Path(f).name for f in files), visual_type=kind, purpose=purpose,
                        chart_choice_reason=why, source_tables=source, period=period, unit=unit, paper_section=section))


def wpp_6933():
    return wpp_simplified(cfg, 0.01).to_crs(CRS)


def lonlat_ticks(ax, lons, lats):
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    xs = [tr.transform(lo, 0)[0] for lo in lons]
    ys = [tr.transform(100, la)[1] for la in lats]
    ax.set_xticks(xs, [f"{abs(lo)}°{'BT' if lo >= 0 else 'BB'}" for lo in lons])
    ax.set_yticks(ys, [f"{abs(la)}°{'LU' if la > 0 else ('LS' if la < 0 else '')}" for la in lats])
    ax.grid(False)


def raster_from_cells(cells, size):
    g = cells.groupby(["ix", "iy"], as_index=False)["n_det"].sum()
    x0, y0 = g["ix"].min(), g["iy"].min()
    img = np.full((g["iy"].max() - y0 + 1, g["ix"].max() - x0 + 1), np.nan, dtype=np.float32)
    img[g["iy"] - y0, g["ix"] - x0] = g["n_det"]
    ext = [x0 * size, (g["ix"].max() + 1) * size, y0 * size, (g["iy"].max() + 1) * size]
    return img, ext


# =========================================================================== F01
def f01():
    fig, ax = plt.subplots(figsize=(P.FULL_W, 112 * P.MM))
    ax.set_xlim(0, 106), ax.set_ylim(0, 66), ax.axis("off")
    FS = 6.4

    def box(x, y, w, h, text, fc="#f4f6f8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=P.MUTED, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, linespacing=1.25)

    def arr(x1, y1, x2, y2, **kw):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=P.MUTED, **kw))

    C = [1, 27, 53, 79]; W = 23
    ax.text(C[0] + W / 2, 64.5, "Input", ha="center", fontsize=7, color=P.MUTED)
    ax.text(C[1] + W / 2, 64.5, "Persiapan", ha="center", fontsize=7, color=P.MUTED)
    ax.text(C[2] + W / 2, 64.5, "Dataset turunan", ha="center", fontsize=7, color=P.MUTED)
    ax.text(C[3] + W / 2, 64.5, "Model & keluaran", ha="center", fontsize=7, color=P.MUTED)
    box(C[0], 52, W, 10, "VIIRS VBD bersih\nviirs_clean_all.csv\n14,2 jt baris\n2012-04 s.d. 2026-05")
    box(C[0], 38, W, 10, "11 poligon WPP\n(.shp tanpa .prj;\nEPSG:4326 diasumsikan)")
    box(C[0], 22, W, 10, "SST (reanalisis) &\nklorofil-a (L4 satelit)\nbulanan per WPP")
    box(C[1], 40, W, 22, "Audit ringan\n\nQF_Detect = 1\nLand_Mask ≠ 0\npoint-in-polygon WPP\nbuang duplikat eksak\n\n"
                         "→ view analitis\n(7,64 jt deteksi)", fc="#e8f0f8")
    box(C[2], 52, W, 10, "A. Grid 1 km\nEPSG:6933 (equal-area)\ndeteksi · malam · tahun", fc="#eef6f1")
    box(C[2], 38, W, 10, "B. Panel WPP–bulan\ntarget_month\nforecast_origin", fc="#fbf1e8")
    box(C[2], 20, W, 14, "Fitur kausal\nlag 1, 2, 3, 12\nrolling m−3..m−1\nbulan sin/cos · WPP 1-hot\n"
                         "SST & Chl m−1 (M3)", fc="#fbf1e8")
    box(C[3], 52, W, 10, "HDBSCAN haversine\nsel aktif ≥2 tahun\nlatih 2013–2023", fc="#eef6f1")
    box(C[3], 38, W, 10, "CFA referensi beku\nunion sel (km²)\naktivitas 2024–2025", fc="#eef6f1")
    box(C[3], 22, W, 12, "M0 Seasonal Naive\nM1 ETS\nM2 RF-History\nM3 RF-Ocean", fc="#fbf1e8")
    box(C[3], 3, W, 15, "Evaluasi rolling\n1 langkah ke depan\nvalidasi 2024: tuning\nuji 2025: param beku\n"
                        "MAE · RMSE · MASE", fc="#f4f6f8")
    box(C[1], 4, W + 26, 13, "Penyajian pembelajaran (Indonesia Fishing Watch)\n"
                             "membaca peta CFA · pola musiman · aktual vs prediksi\n"
                             "korelasi ≠ sebab-akibat · ketidakpastian", fc="#f4f6f8")
    arr(C[0] + W, 57, C[1], 54); arr(C[0] + W, 43, C[1], 46)
    arr(C[1] + W, 55, C[2], 57); arr(C[1] + W, 45, C[2], 43)
    arr(C[2] + W, 57, C[3], 57); arr(C[3] + W / 2, 52, C[3] + W / 2, 48)
    arr(C[2] + W / 2, 38, C[2] + W / 2, 34); arr(C[2] + W, 28, C[3], 28)
    arr(C[0] + W, 27, C[2], 27); arr(C[3] + W / 2, 22, C[3] + W / 2, 17)
    arr(C[3], 10.5, C[1] + W + 26, 10.5)
    # CFA -> penyajian (jalur di tepi kanan, tidak menjadi fitur model)
    ax.plot([C[3] + W + 0.4, 104.5, 104.5], [43, 43, 1.5], color=P.MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.plot([104.5, C[1] + 10], [1.5, 1.5], color=P.MUTED, lw=0.8, ls=(0, (3, 2)))
    arr(C[1] + 10, 1.5, C[1] + 10, 4, ls=(0, (3, 2)))
    ax.text(C[3] + W / 2, 35.6, "cabang regresi", ha="center", fontsize=6, color=P.MUTED)
    ax.text(C[3] + W / 2, 63.2 - 0.2, "", fontsize=6)
    f = P.save(fig, MAIN, "fig01_workflow", ("png", "pdf", "svg"))
    reg("F01", f, "diagram alur (vektor)", "Alur data bersih → dua dataset → HDBSCAN & regresi → evaluasi → pembelajaran",
        "Diagram menunjukkan percabangan proses yang sulit dijelaskan dengan chart numerik", "konfigurasi & skrip",
        "-", "-", "Metodologi / arsitektur solusi")


# =========================================================================== F02
def f02():
    st = pd.read_csv(A / "audit_counts.csv")
    av = pd.read_csv(A / "monthly_record_availability.csv", parse_dates=["month"])
    fig = plt.figure(figsize=(P.FULL_W, 80 * P.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    lab = {"00_input": "Input bersih", "01_valid": "Waktu/koord./rad. valid", "02_qf1": "QF_Detect = 1",
           "03_not_land": "Land_Mask ≠ 0 (daratan)", "04_in_wpp": "Di dalam 11 WPP", "05_dedup_exact": "Tanpa duplikat eksak",
           "L03_land_mask_ne_1": "Lama: Land_Mask ≠ 1", "L04_eez_indonesia": "Lama: + ZEE Indonesia",
           "L05_in_wpp": "Lama: + di dalam WPP"}
    st = st.iloc[::-1]
    y = np.arange(len(st))
    colors = ["#0072B2" if v == "primary" else "#9aa0a6" for v in st["variant"]]
    ax.barh(y, st["n_records"] / 1e6, color=colors, height=0.7)
    for yi, n in zip(y, st["n_records"]):
        ax.text(n / 1e6 + 0.2, yi, f"{n:,}".replace(",", "."), va="center", fontsize=6.5)
    ax.set_yticks(y, [lab[s] for s in st["stage"]])
    ax.set_xlabel("Jumlah rekaman (juta)")
    ax.set_xlim(0, 24)
    ax.grid(axis="y", visible=False)
    ax.set_title("Tahap audit yang dijalankan")
    P.panel_label(ax, "a")
    ax.text(0.99, 0.03, "biru = view utama\nabu = varian lama\n(pembanding)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.2, color=P.MUTED)
    ax2 = fig.add_subplot(gs[1])
    av["year"], av["moy"] = av["month"].dt.year, av["month"].dt.month
    av["frac"] = av["n_dates_any_record"] / av["days_in_month"]
    pv = av.pivot(index="year", columns="moy", values="frac")
    im = ax2.imshow(pv.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_xticks(range(12), list("JFMAMJJASOND"))
    ax2.set_yticks(range(len(pv)), pv.index)
    ax2.grid(False)
    ax2.set_title("Proporsi tanggal dengan ≥1 rekaman")
    cb = fig.colorbar(im, ax=ax2, fraction=0.05, pad=0.03)
    cb.set_label("semua QF, seluruh bbox input;\nbukan cakupan bebas awan", fontsize=6.5)
    P.panel_label(ax2, "b")
    f = P.save(fig, MAIN, "fig02_data_audit")
    reg("F02", f, "batang horizontal + heatmap kalender", "Jumlah rekaman per tahap audit; keberadaan rekaman per bulan",
        "Batang membandingkan antar-tahap; heatmap memperlihatkan pola bolong menurut waktu",
        "audit/audit_counts.csv; audit/monthly_record_availability.csv", "2012-04..2026-05", "rekaman; proporsi tanggal",
        "Dataset dan hasil preprocessing")


# =========================================================================== F03
def f03():
    allc = pd.read_parquet(D / f"grid_all_{SIZE}m.parquet", columns=["ix", "iy", "n_det"])
    img, ext = raster_from_cells(allc, SIZE)
    wpp = wpp_6933()
    fig = plt.figure(figsize=(P.FULL_W, 150 * P.MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.22, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    norm = LogNorm(vmin=1, vmax=np.nanmax(img))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    im = ax.imshow(img, extent=ext, origin="lower", cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)
    wpp.boundary.plot(ax=ax, color="#3c4043", lw=0.4)
    for _, r in wpp.iterrows():
        pt = r.geometry.representative_point()
        ax.text(pt.x, pt.y, str(r["wpp"]), fontsize=6.5, ha="center", va="center", color="#b3261e", fontweight="bold")
    b = wpp.total_bounds
    ax.set_xlim(b[0], b[2]), ax.set_ylim(b[1], b[3])
    lonlat_ticks(ax, [95, 105, 115, 125, 135], [-10, -5, 0, 5])
    ax.set_xlabel(""), ax.set_ylabel("")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("deteksi per sel 1 km² (log)", fontsize=7)
    ax.set_title("Jumlah deteksi QF=1 per sel grid 1 km, seluruh data 2012-04 s.d. 2026-05 (deskriptif)")
    P.panel_label(ax, "a")
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cat = pd.read_csv(T / "cfa_catalog.csv")
    cells = pd.read_parquet(D / "cfa_cells.parquet")
    tr = pd.read_parquet(D / f"grid_train_{SIZE}m.parquet", columns=["wpp", "ix", "iy", "n_det"])
    memb_cmap = ListedColormap(["#cfd4da", "#D55E00"])
    for k, w in enumerate([712, 718]):
        axi = fig.add_subplot(gs[1, k])
        tw = tr[tr["wpp"] == w].merge(cells[cells["wpp"] == w][["ix", "iy", "cfa_id"]], on=["ix", "iy"], how="left")
        tw["n_det"] = tw["cfa_id"].notna().astype(float)  # 0 = sel aktif non-CFA, 1 = anggota CFA
        g = tw.groupby(["ix", "iy"], as_index=False)["n_det"].max()
        x0, y0 = g["ix"].min(), g["iy"].min()
        img2 = np.full((g["iy"].max() - y0 + 1, g["ix"].max() - x0 + 1), np.nan)
        img2[g["iy"] - y0, g["ix"] - x0] = g["n_det"]
        ext2 = [x0 * SIZE, (g["ix"].max() + 1) * SIZE, y0 * SIZE, (g["iy"].max() + 1) * SIZE]
        axi.imshow(img2, extent=ext2, origin="lower", cmap=memb_cmap, vmin=0, vmax=1, interpolation="nearest", rasterized=True)
        wpp[wpp["wpp"] == w].boundary.plot(ax=axi, color="#3c4043", lw=0.5)
        bb = wpp[wpp["wpp"] == w].total_bounds
        axi.set_xlim(bb[0], bb[2]), axi.set_ylim(bb[1], bb[3])
        lo0, lo1 = (106, 116) if w == 712 else (128, 140)
        la = [-8, -6, -4] if w == 712 else [-10, -8, -6]
        lonlat_ticks(axi, list(range(lo0, lo1 + 1, 2 if w == 712 else 4)), la)
        axi.tick_params(labelsize=6.5)
        cw = cat[cat["wpp"] == w]
        area_txt = f"{cw['area_km2'].sum():,.0f}".replace(",", ".")
        axi.set_title(f"WPP {w} · latih 2013–2023 · {len(cw)} CFA, {area_txt} km²", fontsize=7.5)
        axi.set_xlabel(""), axi.set_ylabel("")
        P.panel_label(axi, "bc"[k])
    fig.legend(handles=[Patch(color="#D55E00", label="sel anggota CFA (HDBSCAN)"),
                        Patch(color="#cfd4da", label="sel lain dengan deteksi 2013–2023 (noise/tidak berulang)")],
               loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    f = P.save(fig, MAIN, "fig03_spatial_cfa")
    reg("F03", f, "peta grid (histogram spasial) + inset CFA", "Lokasi konsentrasi deteksi dan batas CFA referensi",
        "Lokasi dan bentuk area memerlukan peta; grid menghindari jutaan titik saling menutup (bukan KDE)",
        "data/grid_all_1000m.parquet; data/grid_train_1000m.parquet; data/cfa_reference_6933.parquet",
        "a: 2012-04..2026-05; b-c: 2013-2023", "deteksi per sel 1 km²", "Pola spasial dan CFA")


# =========================================================================== F04
def f04():
    pn = pd.read_parquet(D / "wpp_month_panel.parquet")
    tot = pn.groupby("month")["n_det"].sum()
    fig = plt.figure(figsize=(P.FULL_W, 128 * P.MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.38)
    ax = fig.add_subplot(gs[0])
    ax.plot(tot.index, tot.to_numpy() / 1e3, color=P.INK, lw=1.0)
    ax.axvspan(pd.Timestamp("2012-04-01"), pd.Timestamp("2012-12-31"), color="#ececec", lw=0)
    ax.axvspan(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-31"), color="#ececec", lw=0)
    ax.text(pd.Timestamp("2012-05-01"), ax.get_ylim()[1] * 0.92, "2012 parsial", fontsize=6.5, color=P.MUTED)
    ax.text(pd.Timestamp("2026-01-15"), ax.get_ylim()[1] * 0.92, "2026\nparsial", fontsize=6.5, color=P.MUTED, va="top")
    ax.set_ylabel("Deteksi per bulan (ribu)")
    ax.set_ylim(bottom=0)
    ax.set_title("Jumlah deteksi bulanan, total 11 WPP")
    P.panel_label(ax, "a")
    ax2 = fig.add_subplot(gs[1])
    p = pn[(pn["year"] >= 2013) & (pn["year"] <= 2025)].copy()
    p["idx"] = p["n_det"] / p.groupby(["wpp", "year"])["n_det"].transform("mean")
    si = p.groupby(["wpp", "moy"])["idx"].median().unstack()
    si.to_csv(T / "F04_seasonal_index_median.csv")
    im = ax2.imshow(si.to_numpy(), aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(1.0, 0.3, 1.7), interpolation="nearest")
    for i in range(si.shape[0]):
        for j in range(12):
            v = si.iat[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8, color="white" if abs(v - 1) > 0.45 else P.INK)
    ax2.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"])
    ax2.set_yticks(range(len(si)), [f"WPP {w}" for w in si.index])
    ax2.grid(False)
    cb = fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.01)
    cb.set_label("indeks (1 = rata-rata\nbulanan tahun itu)", fontsize=6.5)
    ax2.set_title("Indeks musiman relatif per WPP (median 2013–2025)")
    P.panel_label(ax2, "b")
    f = P.save(fig, MAIN, "fig04_temporal_seasonality")
    reg("F04", f, "line chart + heatmap WPP × bulan", "Perubahan bulanan dan waktu puncak musiman antar-WPP",
        "Line chart untuk urutan waktu; heatmap membandingkan pola bulan antar-WPP (indeks relatif, bukan jumlah absolut)",
        "data/wpp_month_panel.parquet; tables/F04_seasonal_index_median.csv", "a: 2012-04..2026-05; b: 2013-2025",
        "deteksi/bulan; indeks tanpa satuan", "Dinamika temporal dan musiman")


# =========================================================================== F05
def f05():
    a = pd.read_parquet(D / "ocean_association_frame.parquet")
    rho = pd.read_csv(T / "T09_ocean_association_spearman.csv")
    fig, axs = plt.subplots(2, 2, figsize=(P.FULL_W, 120 * P.MM))
    for i, w in enumerate(EX_WPP):
        g = a[a["wpp"] == w]
        for j, (var, lab) in enumerate([("sst", "Anomali SST (°C)"), ("log10_chl", "Anomali log10 klorofil-a (log10 mg/m³)")]):
            ax = axs[i, j]
            ax.scatter(g[f"{var}_anom"], g["log_det_anom"], s=9, color="#0072B2", alpha=0.7, lw=0)
            ax.axhline(0, color=P.MUTED, lw=0.5), ax.axvline(0, color=P.MUTED, lw=0.5)
            r = rho[(rho["wpp"] == w) & (rho["variable"] == var) & (rho["relation"] == "anomaly_same_month")].iloc[0]
            ax.text(0.02, 0.97, f"Spearman ρ = {r['spearman_rho']:.2f}\nn = {r['n_months']} bulan", transform=ax.transAxes,
                    va="top", fontsize=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.set_xlabel(lab)
            if j == 0:
                ax.set_ylabel(f"WPP {w}\nAnomali log(1+deteksi)")
            ax.set_title(f"WPP {w}", fontsize=8)
            P.panel_label(ax, "abcd"[i * 2 + j])
    fig.tight_layout()
    f = P.save(fig, MAIN, "fig05_ocean_association")
    reg("F05", f, "scatter anomali", "Asosiasi deskriptif anomali deteksi dengan SST dan klorofil-a (bulan yang sama)",
        "Scatter menampilkan hubungan dan sebaran; anomali (dikurangi rata-rata bulan kalender 2013-2025) memisahkan pola musim bersama",
        "data/ocean_association_frame.parquet; tables/T09_ocean_association_spearman.csv", "2013-01..2025-12",
        "log(1+deteksi); °C; log10 mg/m³", "Hubungan dengan oseanografi")


# =========================================================================== F06 / S04
def forecast_panels(wpps, fname, outdir, fid, ncol_models=2):
    pr = pd.read_csv(T / "T06_predictions.csv", parse_dates=["target_month"])
    pr = pr[pr["split"] == "test"]
    groups = [["M0_SeasonalNaive", "M1_ETS"], ["M2_RF_History", "M3_RF_Ocean"]]
    n = len(wpps)
    if ncol_models == 2:
        fig, axs = plt.subplots(n, 2, figsize=(P.FULL_W, 52 * P.MM * n), sharex=True, squeeze=False)
    else:
        nc = 3
        nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(P.FULL_W, 50 * P.MM * nr), sharex=True, squeeze=False)
    for i, w in enumerate(wpps):
        g = pr[pr["wpp"] == w]
        act = g[g["model"] == "M0_SeasonalNaive"].sort_values("target_month")
        if ncol_models == 2:
            lo = min(g["y_true"].min(), g["y_pred"].min()) * 0.9
            hi = max(g["y_true"].max(), g["y_pred"].max()) * 1.08
            for j, mods in enumerate(groups):
                ax = axs[i, j]
                ax.plot(act["target_month"], act["y_true"], **P.ACTUAL_STYLE)
                for m in mods:
                    gm = g[g["model"] == m].sort_values("target_month")
                    st = P.MODEL_STYLE[m]
                    ax.plot(gm["target_month"], gm["y_pred"], color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.3,
                            ms=3.5, label=st["label"])
                ax.set_ylim(max(lo, 0), hi)
                ax.set_title(f"WPP {w}", fontsize=8)
                if j == 0:
                    ax.set_ylabel("Deteksi per bulan")
                P.panel_label(ax, "abcdefgh"[i * 2 + j])
        else:
            ax = axs.flat[i]
            ax.plot(act["target_month"], act["y_true"], **{**P.ACTUAL_STYLE, "lw": 1.6, "ms": 2.5})
            for m in P.MODELS:
                gm = g[g["model"] == m].sort_values("target_month")
                st = P.MODEL_STYLE[m]
                ax.plot(gm["target_month"], gm["y_pred"], color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.0, ms=2.2,
                        label=st["label"])
            ax.set_title(f"WPP {w}", fontsize=8)
            ax.tick_params(labelsize=6.5)
    for ax in axs.flat:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    if ncol_models == 2:
        hs, ls_ = [], []
        for ax in axs[0]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in ls_:
                    hs.append(h), ls_.append(l)
        fig.legend(hs, ls_, loc="upper center", ncol=5, fontsize=7, handlelength=2.8, bbox_to_anchor=(0.5, 1.035))
    if ncol_models != 2:
        for ax in list(axs.flat)[n:]:
            ax.axis("off")
        h, l = axs.flat[0].get_legend_handles_labels()
        list(axs.flat)[n].legend(h, l, loc="center", fontsize=7.5)
        fig.supylabel("Deteksi per bulan (sumbu y per panel)", fontsize=8)
    fig.tight_layout()
    f = P.save(fig, outdir, fname)
    return f


def f06():
    f = forecast_panels(EX_WPP, "fig06_forecast_test", MAIN, "F06")
    reg("F06", f, "small multiples line chart", "Aktual vs prediksi satu langkah ke depan 2025 untuk WPP contoh",
        "Line chart mempertahankan urutan waktu dan memperlihatkan kapan prediksi meleset; dua panel per WPP agar 4 model terbaca",
        "tables/T06_predictions.csv", "2025-01..2025-12 (uji)", "deteksi per bulan", "Hasil peramalan")


# =========================================================================== F07
def f07():
    m = pd.read_csv(T / "T07_metrics_by_wpp.csv")
    m = m[m["split"] == "test"]
    pv = m.pivot(index="wpp", columns="model", values="MASE")[P.MODELS]
    s = pd.read_csv(T / "T07b_model_summary.csv")
    s = s[s["split"] == "test"].set_index("model").loc[P.MODELS]
    full = pd.concat([pv, pd.DataFrame([s["macro_MASE"].to_numpy()], index=["macro"], columns=P.MODELS)])
    fig, ax = plt.subplots(figsize=(P.HALF_W * 1.45, 105 * P.MM))
    im = ax.imshow(full.to_numpy(), aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(1.0, 0.3, 3.3), interpolation="nearest")
    for i in range(full.shape[0]):
        best = np.nanargmin(full.iloc[i].to_numpy())
        for j in range(full.shape[1]):
            v = full.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                    fontweight="bold" if j == best else None, color="white" if (v - 1) > 1.0 or v < 0.45 else P.INK)
    ax.axhline(full.shape[0] - 1.5, color="white", lw=2)
    ax.set_xticks(range(4), [P.MODEL_STYLE[x]["label"].replace(" ", "\n", 1) for x in P.MODELS], fontsize=7)
    ax.set_yticks(range(full.shape[0]), [f"WPP {w}" for w in pv.index] + ["Macro-MASE"])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("MASE uji 2025 (1 = galat seasonal naive\npada periode latih)", fontsize=7)
    ax.set_title("MASE per WPP × model (n = 12 bulan/WPP)\ntebal = terendah per baris", fontsize=8)
    f = P.save(fig, MAIN, "fig07_model_comparison")
    reg("F07", f, "heatmap beranotasi", "Perbandingan MASE 4 model di 11 WPP pada uji 2025",
        "Heatmap membandingkan banyak model × wilayah tanpa batang berdesakan; tabel MAE/RMSE terpisah (T07)",
        "tables/T07_metrics_by_wpp.csv; tables/T07b_model_summary.csv", "2025-01..2025-12", "MASE (tanpa satuan)",
        "Evaluasi model")


# =========================================================================== F08
def f08():
    ab = pd.read_csv(T / "T08_ocean_ablation_M2_vs_M3.csv")
    ab = ab[ab["split"] == "test"].sort_values("delta_MAE_M2_minus_M3")
    fig, ax = plt.subplots(figsize=(P.HALF_W * 1.45, 95 * P.MM))
    y = np.arange(len(ab))
    col = [P.POS if d > 0 else P.NEG for d in ab["delta_MAE_M2_minus_M3"]]
    ax.hlines(y, 0, ab["delta_MAE_M2_minus_M3"], color=col, lw=1.6)
    ax.scatter(ab["delta_MAE_M2_minus_M3"], y, color=col, s=30, zorder=3,
               marker=None)
    ax.axvline(0, color=P.INK, lw=0.8)
    for yi, d, r in zip(y, ab["delta_MAE_M2_minus_M3"], ab["delta_rel_pct_of_M2"]):
        ax.text(d + (4 if d >= 0 else -4), yi, f"{d:+.0f} ({r:+.1f}%)", va="center", ha="left" if d >= 0 else "right", fontsize=6.5)
    ax.set_yticks(y, [f"WPP {w}" for w in ab["wpp"]])
    lim = np.abs(ab["delta_MAE_M2_minus_M3"]).max() * 1.6
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("ΔMAE = MAE(M2) − MAE(M3)  [deteksi/bulan]")
    ax.text(0.98, 1.01, "M3 lebih baik →", transform=ax.transAxes, ha="right", fontsize=7, color=P.POS)
    ax.text(0.02, 1.01, "← M3 lebih buruk", transform=ax.transAxes, ha="left", fontsize=7, color=P.NEG)
    ax.grid(axis="y", visible=False)
    ax.set_title("Kontribusi SST & klorofil-a (uji 2025, n = 12 bulan/WPP)\n", fontsize=8)
    f = P.save(fig, MAIN, "fig08_ocean_ablation")
    reg("F08", f, "lollipop divergen", "Apakah penambahan oseanografi menurunkan MAE per WPP",
        "Posisi terhadap nol langsung menjawab membantu/merugikan; label persen relatif terhadap MAE M2",
        "tables/T08_ocean_ablation_M2_vs_M3.csv", "2025-01..2025-12", "deteksi per bulan", "Pengujian kontribusi oseanografi")


# =========================================================================== lampiran
def s01_s02():
    h = pd.read_csv(T / "S01_radiance_hist.csv")
    tot = pd.read_csv(T / "S01_radiance_overall.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(P.HALF_W * 1.3, 65 * P.MM))
    ax.bar(h["bin_left"], h["count"] / 1e3, width=h["bin_right"] - h["bin_left"], align="edge", color="#0072B2", lw=0)
    ax.set_xscale("log")
    for v in (10, 400):
        ax.axvline(v, color=P.NEG, lw=0.9, ls="--")
        ax.text(v * 1.1, ax.get_ylim()[1] * 0.97, f"{v}", color=P.NEG, fontsize=7, va="top",
                bbox=dict(fc="white", ec="none", pad=0.8))
    ax.axvline(tot["rad_median"], color=P.INK, lw=0.9)
    ax.text(tot["rad_median"] * 1.1, ax.get_ylim()[1] * 0.75, f"median {tot['rad_median']:.2f}", fontsize=7)
    ax.set_xlabel("Rad_DNB (nW/cm²/sr, skala log)")
    ax.set_ylabel("Deteksi per bin (ribu)")
    n_txt = f"{int(tot['n_det']):,}".replace(",", ".")
    ax.set_title(f"Distribusi radiansi (n = {n_txt} deteksi)", fontsize=8)
    f = P.save(fig, SUPP, "S01_radiance_hist")
    reg("S01", f, "histogram sumbu x log", "Distribusi radiansi view analitis", "Distribusi menceng kanan",
        "tables/S01_radiance_hist.csv", "2012-04..2026-05", "nW/cm²/sr", "Lampiran")
    q = pd.read_csv(T / "S02_radiance_quantiles_by_wpp.csv")
    fig, ax = plt.subplots(figsize=(P.FULL_W, 72 * P.MM))
    for i, r in q.iterrows():
        ax.add_patch(Rectangle((i - 0.3, r["p25"]), 0.6, r["p75"] - r["p25"], fc="#cfe3f3", ec="#0072B2", lw=0.8))
        ax.plot([i - 0.3, i + 0.3], [r["p50"]] * 2, color=P.INK, lw=1.2)
        ax.plot([i, i], [r["p05"], r["p25"]], color="#0072B2", lw=0.8)
        ax.plot([i, i], [r["p75"], r["p95"]], color="#0072B2", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(q)), [f"{w}\n{n / 1e3:,.0f} rb".replace(",", ".") for w, n in zip(q["wpp"], q["n"])], fontsize=6.5)
    ax.set_xlabel("WPP (n deteksi, ribu)")
    ax.set_xlim(-0.6, len(q) - 0.4)
    ax.set_ylabel("Rad_DNB (nW/cm²/sr, log)")
    ax.set_title("Radiansi per WPP: kotak = P25–P75, garis = median, whisker = P5–P95", fontsize=8)
    f = P.save(fig, SUPP, "S02_radiance_box_by_wpp")
    reg("S02", f, "boxplot (kuantil) skala log", "Membandingkan median dan rentang radiansi antar-WPP",
        "Lebih ringkas daripada violin; whisker = P5-P95 (bukan aturan 1.5 IQR)", "tables/S02_radiance_quantiles_by_wpp.csv",
        "2012-04..2026-05", "nW/cm²/sr", "Lampiran")


def s03():
    r = pd.read_csv(T / "T04a_hdbscan_param_runs.csv")
    r["param"] = "mcs" + r["min_cluster_size"].astype(str) + "/ms" + r["min_samples"].astype(str)
    order = r.drop_duplicates("param").sort_values(["min_cluster_size", "min_samples"])["param"].tolist()
    fig, axs = plt.subplots(1, 3, figsize=(P.FULL_W, 85 * P.MM), sharey=True)
    for ax, col, lab, cm in [(axs[0], "n_clusters", "Jumlah cluster", "Blues"),
                             (axs[1], "noise_frac_cells", "Proporsi noise (sel)", "Blues"),
                             (axs[2], "mean_ari_vs_other_params", "ARI rata-rata vs param lain", "Blues")]:
        pv = r.pivot(index="wpp", columns="param", values=col)[order]
        vals = pv.to_numpy().astype(float)
        im = ax.imshow(np.log10(vals + 1) if col == "n_clusters" else vals, aspect="auto", cmap=cm, interpolation="nearest")
        sel = r[r["selected"]].set_index("wpp")["param"]
        for i, w in enumerate(pv.index):
            for j, p_ in enumerate(order):
                v = pv.iat[i, j]
                ax.text(j, i, f"{int(v)}" if col == "n_clusters" else f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color=P.INK)
                if sel.get(w) == p_:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=P.NEG, lw=1.4))
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
        ax.set_yticks(range(len(pv)), [f"WPP {w}" for w in pv.index])
        ax.grid(False)
        ax.set_title(lab, fontsize=8)
    fig.suptitle("Sensitivitas HDBSCAN terhadap parameter (kotak jingga = terpilih: ARI rata-rata tertinggi)", fontsize=8, x=0.01, ha="left")
    fig.tight_layout()
    f = P.save(fig, SUPP, "S03_hdbscan_param_sensitivity")
    reg("S03", f, "heatmap parameter × WPP", "Bentuk/jumlah CFA bergantung pada parameter; noise minimum bukan kriteria",
        "Heatmap membandingkan 9 kombinasi × 11 WPP", "tables/T04a_hdbscan_param_runs.csv", "latih 2013-2023", "-", "Lampiran")
    # peta CFA nasional (data latih)
    cfa = gpd.read_parquet(D / "cfa_reference_6933.parquet")
    wpp = wpp_6933()
    fig, ax = plt.subplots(figsize=(P.FULL_W, 80 * P.MM))
    wpp.plot(ax=ax, color="#f4f6f8", edgecolor="#3c4043", lw=0.4)
    cfa.plot(ax=ax, color="#D55E00", lw=0, rasterized=True)
    b = wpp.total_bounds
    ax.set_xlim(b[0], b[2]), ax.set_ylim(b[1], b[3])
    lonlat_ticks(ax, [95, 105, 115, 125, 135], [-10, -5, 0, 5])
    ax.set_xlabel(""), ax.set_ylabel("")
    area_txt = f"{cfa['area_km2'].sum():,.0f}".replace(",", ".")
    ax.set_title(f"CFA referensi (latih 2013–2023): {len(cfa)} CFA, {area_txt} km²", fontsize=8)
    f = P.save(fig, SUPP, "S03b_cfa_reference_map")
    reg("S03b", f, "peta poligon", "Sebaran CFA referensi semua WPP (data latih)", "Lokasi memerlukan peta",
        "data/cfa_reference_6933.parquet; tables/cfa_catalog.csv", "2013-2023", "km²", "Lampiran")


def s04():
    f = forecast_panels(cfg["wpp_codes"], "S04_forecast_all_wpp", SUPP, "S04", ncol_models=4)
    reg("S04", f, "small multiples", "Aktual-prediksi seluruh 11 WPP (tidak hanya yang sukses)",
        "Sumbu y per panel; tinggi panel tidak sebanding sebagai jumlah absolut", "tables/T06_predictions.csv", "2025",
        "deteksi per bulan", "Lampiran")


def s05_s06():
    pr = pd.read_csv(T / "T06_predictions.csv", parse_dates=["target_month"])
    pr = pr[pr["split"] == "test"].copy()
    pr["resid"] = pr["y_true"] - pr["y_pred"]
    pr["rel_resid"] = pr["resid"] / pr["y_true"]
    fig, axs = plt.subplots(1, 2, figsize=(P.FULL_W, 70 * P.MM))
    for m in P.MODELS:
        g = pr[pr["model"] == m].groupby("target_month")["rel_resid"].median()
        st = P.MODEL_STYLE[m]
        axs[0].plot(g.index, 100 * g, color=st["color"], marker=st["marker"], ls=st["ls"], ms=3, lw=1.1, label=st["label"])
        gm = pr[pr["model"] == m]
        axs[1].scatter(gm["y_pred"], 100 * gm["rel_resid"], s=8, color=st["color"], marker=st["marker"], alpha=0.7,
                       label=st["label"], lw=0)
    for ax in axs:
        ax.axhline(0, color=P.INK, lw=0.6)
    axs[0].set_ylabel("Median residual relatif antar-WPP (%)\n(aktual − prediksi)/aktual")
    axs[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    axs[0].legend(fontsize=6.5, ncol=2)
    axs[0].set_title("Residual relatif menurut waktu (uji 2025)", fontsize=8)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Prediksi (deteksi/bulan, log)")
    axs[1].set_ylabel("Residual relatif (%)")
    axs[1].set_title("Prediksi vs residual relatif (132 titik/model)", fontsize=8)
    fig.tight_layout()
    f = P.save(fig, SUPP, "S05_residuals")
    reg("S05", f, "line + scatter residual", "Bias dan pola galat menurut waktu dan besar prediksi",
        "Residual relatif agar WPP berskala besar tidak mendominasi", "tables/T06_predictions.csv", "2025", "%", "Lampiran")
    m = pd.read_csv(T / "T07_metrics_by_wpp.csv")
    m = m[m["split"] == "test"]
    fig, ax = plt.subplots(figsize=(P.HALF_W * 1.4, 75 * P.MM))
    rng = np.random.default_rng(0)
    pv = m.pivot(index="wpp", columns="model", values="MASE")[P.MODELS]
    for i, w in enumerate(pv.index):
        ax.plot(range(4), pv.loc[w].to_numpy(), color="#c8c8c8", lw=0.6, zorder=1)
    for j, mo in enumerate(P.MODELS):
        st = P.MODEL_STYLE[mo]
        ax.scatter(np.full(len(pv), j) + rng.uniform(-0.06, 0.06, len(pv)), pv[mo], color=st["color"], marker=st["marker"], s=18, zorder=2)
        ax.plot([j - 0.25, j + 0.25], [pv[mo].median()] * 2, color=P.INK, lw=1.4)
    ax.axhline(1, color=P.MUTED, lw=0.6, ls="--")
    ax.set_xticks(range(4), [P.MODEL_STYLE[x]["label"].replace(" ", "\n", 1) for x in P.MODELS], fontsize=7)
    ax.set_ylabel("MASE uji 2025 per WPP")
    ax.set_title("Sebaran MASE per WPP (11 titik/model, n = 12 bulan/WPP);\ngaris hitam = median, garis abu menghubungkan WPP yang sama", fontsize=7.5)
    f = P.save(fig, SUPP, "S06_error_distribution")
    reg("S06", f, "paired dot plot", "Variasi galat antar-WPP per model", "Menampilkan semua titik; tidak memakai violin",
        "tables/T07_metrics_by_wpp.csv", "2025", "MASE", "Lampiran")


def s07_s10():
    ann = pd.read_csv(T / "S07_annual_counts_2013_2025.csv")
    fig, axs = plt.subplots(3, 4, figsize=(P.FULL_W, 105 * P.MM), sharex=True)
    for ax, (w, g) in zip(axs.flat, ann.groupby("wpp")):
        ax.plot(g["year"], g["n_det"] / 1e3, color="#0072B2", marker="o", ms=2.5, lw=1)
        ax.set_title(f"WPP {w}", fontsize=7.5)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6)
    tot = ann.groupby("year")["n_det"].sum()
    axs.flat[11].plot(tot.index, tot / 1e3, color=P.INK, marker="o", ms=2.5, lw=1)
    axs.flat[11].set_title("Total 11 WPP", fontsize=7.5)
    axs.flat[11].set_ylim(bottom=0)
    axs.flat[11].tick_params(labelsize=6)
    fig.supylabel("Deteksi per tahun (ribu; sumbu y per panel)", fontsize=7.5)
    fig.tight_layout()
    f = P.save(fig, SUPP, "S07_annual_counts")
    reg("S07", f, "small multiples line", "Deret tahunan 2013-2025 (tahun penuh)",
        "Deskriptif; uji tren inferensial tidak dijalankan ulang", "tables/S07_annual_counts_2013_2025.csv", "2013-2025",
        "deteksi/tahun", "Lampiran")
    imp = pd.read_csv(T / "S08_permutation_importance_val.csv")
    imp = imp[~imp["feature"].str.startswith("wpp_")]
    fig, axs = plt.subplots(1, 2, figsize=(P.FULL_W, 70 * P.MM), sharex=True)
    for ax, mo in zip(axs, ["M2_RF_History", "M3_RF_Ocean"]):
        g = imp[imp["model"] == mo].sort_values("importance_mean")
        ax.barh(g["feature"], g["importance_mean"], xerr=g["importance_std"], color=P.MODEL_STYLE[mo]["color"],
                error_kw=dict(lw=0.6, ecolor=P.MUTED), height=0.65)
        ax.set_title(P.MODEL_STYLE[mo]["label"], fontsize=8)
        ax.axvline(0, color=P.INK, lw=0.6)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=6.5)
    axs[0].set_xlabel("Kenaikan MAE log1p bila fitur diacak\n(validasi 2024, 30 ulangan)")
    axs[1].set_xlabel("Kenaikan MAE log1p bila fitur diacak\n(validasi 2024, 30 ulangan)")
    fig.tight_layout()
    f = P.save(fig, SUPP, "S08_permutation_importance")
    reg("S08", f, "batang horizontal + sd", "Ketergantungan model pada fitur (bukan sebab-akibat)",
        "Pengacakan lintas baris panel; fitur berkorelasi berbagi pentingnya; dummy WPP tidak ditampilkan",
        "tables/S08_permutation_importance_val.csv", "validasi 2024", "MAE skala log1p", "Lampiran")
    pr = pd.read_csv(T / "S09_wpp_radiance_profile.csv").sort_values("n_det")
    fig, axs = plt.subplots(1, 2, figsize=(P.FULL_W, 70 * P.MM), sharey=True)
    y = np.arange(len(pr))
    axs[0].barh(y, pr["n_det"] / 1e6, color="#0072B2", height=0.7)
    for yi, n, s in zip(y, pr["n_det"], pr["share_pct"]):
        axs[0].text(n / 1e6 + 0.03, yi, f"{s:.1f}%", va="center", fontsize=6.5)
    axs[0].set_yticks(y, [f"WPP {w}" for w in pr["wpp"]])
    axs[0].set_xlabel("Deteksi (juta), 2012-04..2026-05")
    axs[0].set_xlim(0, pr["n_det"].max() / 1e6 * 1.2)
    left = np.zeros(len(pr))
    for col, lab, c in [("pct_low", "<10", "#cfe3f3"), ("pct_medium", "10–400", "#5aa0d6"), ("pct_high", ">400", "#08306b")]:
        axs[1].barh(y, pr[col], left=left, color=c, height=0.7, label=lab, edgecolor="white", lw=0.8)
        left += pr[col].to_numpy()
    for yi, ph in zip(y, pr["pct_high"]):
        axs[1].text(101, yi, f">400: {ph:.2f}%", va="center", fontsize=6)
    axs[1].set_xlim(0, 118)
    axs[1].set_xlabel("Komposisi kelas radiansi (%)")
    axs[1].legend(title="nW/cm²/sr", fontsize=6.5, title_fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for ax in axs:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    f = P.save(fig, SUPP, "S09_wpp_profile")
    reg("S09", f, "batang + stacked bar", "Profil jumlah deteksi dan kelas radiansi per WPP",
        "Kelas radiansi, bukan kelas ukuran kapal", "tables/S09_wpp_radiance_profile.csv", "2012-04..2026-05", "deteksi; %", "Lampiran")
    cv = pd.read_csv(T / "S10_interannual_cv.csv").sort_values("cv_pct")
    fig, ax = plt.subplots(figsize=(P.HALF_W * 1.3, 70 * P.MM))
    ax.scatter(cv["cv_pct"], range(len(cv)), color="#0072B2", s=22)
    ax.hlines(range(len(cv)), 0, cv["cv_pct"], color="#c8c8c8", lw=0.8)
    ax.set_yticks(range(len(cv)), [f"WPP {w}" for w in cv["wpp"]])
    ax.set_xlim(0, cv["cv_pct"].max() * 1.15)
    ax.set_xlabel("CV jumlah deteksi tahunan (%), 2013–2025, n = 13 tahun")
    ax.grid(axis="y", visible=False)
    f = P.save(fig, SUPP, "S10_interannual_cv")
    reg("S10", f, "dot plot", "Variasi antartahun per WPP", "Cutoff CV bukan standar universal; mean/SD di tabel",
        "tables/S10_interannual_cv.csv", "2013-2025", "%", "Lampiran")


if __name__ == "__main__":
    only = set(sys.argv[2:]) if len(sys.argv) > 2 else None
    status = {}
    for fn in [f01, f02, f03, f04, f05, f06, f07, f08, s01_s02, s03, s04, s05_s06, s07_s10]:
        if only and fn.__name__ not in only:
            continue
        try:
            fn()
            status[fn.__name__] = "ok"
        except Exception as e:  # noqa: BLE001
            status[fn.__name__] = f"failed: {e}"
            traceback.print_exc()
    cat_path = run_dir(cfg, "report") / "figure_catalog.csv"
    new = pd.DataFrame(CATALOG)
    if cat_path.exists() and only:
        old = pd.read_csv(cat_path)
        new = pd.concat([old[~old["id"].isin(new["id"])], new]).sort_values("id")
    new.to_csv(cat_path, index=False)
    log_step(cfg, "06_figures" + ("_partial" if only else ""), {"status": status})
    print(json.dumps(status, indent=1))
