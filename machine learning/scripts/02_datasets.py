"""Tahap 2: dataset A (grid spasial) dan B (panel WPP-bulan) + tabel oseanografi."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.audit import analytical_path  # noqa: E402
from src.common import Timer, duck, load_config, log_step, p, run_dir  # noqa: E402
from src.features import build_wpp_month  # noqa: E402
from src.ocean import load_ocean  # noqa: E402
from src.spatial import aggregate_grid, build_points_grid  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    data, tables = run_dir(cfg, "data"), run_dir(cfg, "tables")
    checks = {}
    with Timer() as t:
        ana = analytical_path(cfg)
        sizes = [cfg["spatial"]["grid_size_m"], *cfg["spatial"]["sensitivity_grid_sizes_m"]]
        pts = build_points_grid(cfg, ana, data / "points_grid.parquet", sizes=sizes)
        con = duck(cfg)
        n_pts = lambda a, b: con.sql(f"SELECT count(*) FROM read_parquet('{pts.as_posix()}') "
                                     f"WHERE date_utc BETWEEN DATE '{a}' AND DATE '{b}'").fetchone()[0]
        periods = {"all": ("2012-04-01", "2026-05-31"),
                   "train": (cfg["hdbscan"]["cfa_train_start"], cfg["hdbscan"]["cfa_train_end"])}
        for name, (a, b) in periods.items():
            for s in sizes:
                g = aggregate_grid(cfg, con, pts, s, a, b)
                g.to_parquet(data / f"grid_{name}_{s}m.parquet", index=False)
                checks[f"grid_{name}_{s}m_reconciles"] = bool(g["n_det"].sum() == n_pts(a, b))
                checks[f"grid_{name}_{s}m_cells"] = int(len(g))
                checks[f"grid_{name}_{s}m_cells_ge2years"] = int((g["n_years"] >= 2).sum())
        avail = pd.read_csv(run_dir(cfg, "audit") / "monthly_record_availability.csv", parse_dates=["month"])
        panel = build_wpp_month(cfg, con, ana, avail)
        n_ana = con.sql(f"SELECT count(*) FROM read_parquet('{ana.as_posix()}')").fetchone()[0]
        checks["panel_reconciles_with_points"] = bool(panel["n_det"].sum() == n_ana)
        checks["panel_status_counts"] = panel["count_status"].value_counts().to_dict()
        panel.to_parquet(data / "wpp_month_panel.parquet", index=False)

        ocean, ocean_info = load_ocean(p(cfg, "ocean_dir"))
        ocean.to_parquet(data / "ocean_wpp_month.parquet", index=False)
        (data / "ocean_metadata.json").write_text(json.dumps(ocean_info, indent=2, default=str), encoding="utf-8")

        # ringkasan CSV kelengkapan & statistik WPP-bulan
        main = panel[(panel["month"] >= cfg["period"]["main_start"]) & (panel["month"] <= cfg["period"]["main_end"])]
        summ = main.groupby("wpp").agg(n_months=("month", "size"),
                                       n_positive=("count_status", lambda s: (s == "observed_positive").sum()),
                                       n_zero=("count_status", lambda s: (s == "zero_no_detection").sum()),
                                       n_not_available=("count_status", lambda s: (s == "not_available").sum()),
                                       total_det=("n_det", "sum"), mean_monthly=("n_det", "mean"),
                                       median_monthly=("n_det", "median"), min_monthly=("n_det", "min"),
                                       max_monthly=("n_det", "max")).reset_index()
        summ["period"] = f"{cfg['period']['main_start']}..{cfg['period']['main_end']}"
        summ.to_csv(tables / "T03_wpp_month_completeness.csv", index=False)
        panel.assign(month=panel["month"].dt.strftime("%Y-%m")).to_csv(data / "wpp_month_panel.csv", index=False)
    log_step(cfg, "02_datasets", {"seconds": t.seconds, "checks": checks})
    print(json.dumps(checks, indent=1, default=str))
    print(summ.to_string())
