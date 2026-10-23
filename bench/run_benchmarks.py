"""Corre todos los experimentos y guarda los CSV en results/bench/.

    python bench/run_benchmarks.py --quick          # 1K a 100K (minutos)
    python bench/run_benchmarks.py                  # 1K a 10M (horas en Python puro)
    python bench/run_benchmarks.py --only bloom cms # algunos experimentos
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_base  # noqa: E402
import bench_graph  # noqa: E402
import bench_probabilistic  # noqa: E402
import bench_similarity  # noqa: E402
from common import QUICK_SIZES, SIZES, save_platform  # noqa: E402

EXPERIMENTS = {
    "base": bench_base.run,
    "bloom": bench_probabilistic.run_bloom,
    "cms": bench_probabilistic.run_cms,
    "minhash": bench_similarity.run_minhash,
    "ann": bench_similarity.run_ann,
    "pagerank": bench_graph.run_pagerank,
    "communities": bench_graph.run_communities,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", choices=list(EXPERIMENTS))
    ap.add_argument("--sizes", nargs="*", type=int)
    ap.add_argument("--louvain-max-edges", type=int, default=1_000_000,
                    help="Louvain en Python puro supera los 40 min con 10M aristas; por encima solo corre LPA")
    a = ap.parse_args()
    bench_graph.LOUVAIN_MAX_EDGES = a.louvain_max_edges
    sizes = a.sizes or (QUICK_SIZES if a.quick else SIZES)
    save_platform()
    for name in a.only or EXPERIMENTS:
        t = time.perf_counter()
        print(f"=== {name} {sizes}", flush=True)
        EXPERIMENTS[name](sizes)
        print(f"=== {name} terminado en {time.perf_counter() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
