"""Tahap 1: ingest CSV bersih -> Parquet bertipe, penentuan WPP, audit, view analitis."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import run_audit  # noqa: E402
from src.common import Timer, load_config, log_step  # noqa: E402
from src.ingest import build_typed_parquet, source_manifest  # noqa: E402
from src.spatial import build_wpp_parquet  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    with Timer() as t:
        typed = build_typed_parquet(cfg)
        wppq = build_wpp_parquet(cfg, typed)
        audit = run_audit(cfg, wppq)
    log_step(cfg, "01_audit", {"seconds": t.seconds, "input": source_manifest(cfg),
                               "analytical_records": audit["analytical_view"]["n_records"]})
    for s in audit["stages"]:
        print(f"{s['variant']:12s} {s['stage']:22s} {s['n_records']:>12,}")
