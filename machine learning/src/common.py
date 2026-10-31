"""Utilitas bersama: konfigurasi, path keluaran, koneksi DuckDB, log manifest."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "research.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def p(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def run_dir(cfg: dict, sub: str | None = None) -> Path:
    d = ROOT / cfg["paths"]["outputs"] / cfg["run_id"]
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir(cfg: dict) -> Path:
    d = ROOT / cfg["paths"]["cache"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def duck(cfg: dict):
    import duckdb

    con = duckdb.connect()
    tmp = cache_dir(cfg) / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET memory_limit='{cfg['resources']['duckdb_memory_limit']}'")
    con.execute(f"SET threads={int(cfg['resources']['threads'])}")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    return con


PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "geopandas", "shapely", "pyproj",
            "hdbscan", "duckdb", "pyarrow", "statsmodels", "matplotlib", "seaborn", "pyyaml"]


def package_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def file_fingerprint(path: Path, full_hash_max_bytes: int = 200 * 1024 * 1024) -> dict:
    """Checksum layak: SHA-256 penuh untuk file kecil, untuk file besar hash 3 blok 16 MB + ukuran + mtime."""
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if st.st_size <= full_hash_max_bytes:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
            method = "sha256_full"
        else:
            blk = 16 << 20
            for off in (0, st.st_size // 2, max(0, st.st_size - blk)):
                f.seek(off)
                h.update(f.read(blk))
            method = "sha256_sampled_3x16MB"
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "hash_method": method, "hash": h.hexdigest()}


def log_step(cfg: dict, step: str, info: dict) -> None:
    """Tambahkan catatan eksekusi ke manifest run (versi paket, waktu, konfigurasi)."""
    man_path = run_dir(cfg) / "run_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"steps": {}}
    man["run_id"] = cfg["run_id"]
    man["config"] = {k: v for k, v in cfg.items() if not k.startswith("_")}
    man["package_versions"] = package_versions()
    man["steps"][step] = {"finished_at_utc": datetime.now(timezone.utc).isoformat(), **info}
    man_path.write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = round(time.time() - self.t0, 1)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


os.environ.setdefault("OMP_NUM_THREADS", "4")
