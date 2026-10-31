"""Gaya & utilitas gambar ilmiah (Matplotlib). Semua gambar dibangun dari tabel keluaran."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
FULL_W = 178 * MM  # lebar halaman makalah
HALF_W = 86 * MM

# Okabe-Ito, urutan tetap per model (divalidasi dengan validator palet; oranye kontras rendah -> marker+legend)
MODEL_STYLE = {
    "M0_SeasonalNaive": dict(color="#E69F00", marker="s", ls=(0, (1, 1.5)), label="M0 Seasonal Naive"),
    "M1_ETS": dict(color="#0072B2", marker="^", ls=(0, (5, 2)), label="M1 ETS"),
    "M2_RF_History": dict(color="#009E73", marker="o", ls="-.", label="M2 RF-History"),
    "M3_RF_Ocean": dict(color="#D55E00", marker="D", ls="-", label="M3 RF-Ocean"),
}
ACTUAL_STYLE = dict(color="black", marker="o", ls="-", lw=2.0, ms=4, label="Aktual", zorder=5)
MODELS = list(MODEL_STYLE)
INK, MUTED, GRID = "#1f1f1f", "#5f6368", "#e3e3e3"
POS, NEG = "#0072B2", "#D55E00"  # pasangan divergen (lebih baik / lebih buruk)


def setup():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "figure.dpi": 100,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "savefig.facecolor": "white", "figure.facecolor": "white",
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig, out_dir: Path, name: str, formats=("png", "pdf")) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in formats:
        pth = out_dir / f"{name}.{f}"
        fig.savefig(pth, dpi=300, bbox_inches="tight", pad_inches=0.03)
        paths.append(str(pth))
    plt.close(fig)
    return paths


def panel_label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
