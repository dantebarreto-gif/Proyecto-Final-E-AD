"""Prepara el stream de eventos en Parquet.

    python scripts/prepare_data.py --events 1000000                    # sintético con esquema REES46
    python scripts/prepare_data.py --source rees46 --events 10000000   # REES46 real (CSV en data/raw/)
"""
import argparse
import time

import _path  # noqa: F401
from eda.data.rees46 import prepare


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "rees46"], default="synthetic")
    ap.add_argument("--events", type=int, default=1_000_000)
    ap.add_argument("--out", default=None, help="carpeta de salida (por defecto data/<source>_<n>)")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = a.out or f"data/{a.source}_{a.events}"
    t = time.perf_counter()
    ev, cat = prepare(a.source, a.events, out, a.raw, a.seed)
    print(f"Listo en {time.perf_counter() - t:.1f} s: {ev} y {cat}")


if __name__ == "__main__":
    main()
