"""Utilidades de medición: tiempo con repeticiones, memoria con tracemalloc y registro en CSV.

Protocolo (sección 3 del enunciado):
- Tamaños 1K, 10K, 100K, 1M y 10M.
- 100 repeticiones cuando es viable. Para tamaños grandes se reduce el
  número (ver REPS) porque una sola corrida en Python puro a 10M toma de
  decenas de segundos a minutos; se reporta promedio y desviación estándar.
- Memoria actual y pico con tracemalloc en una corrida aparte (tracemalloc
  agrega sobrecosto, así que no se mezcla con la medición de tiempo).
- Semilla fija y datos de la plataforma en results/bench/platform.json.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import platform
import sys
import time
import tracemalloc
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
REPS = {1_000: 100, 10_000: 100, 100_000: 10, 1_000_000: 3, 10_000_000: 1}
QUICK_SIZES = [1_000, 10_000, 100_000]
SEED = 2026
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bench")


def reps_for(n: int, scale: float = 1.0) -> int:
    base = REPS.get(n, 1)
    return max(1, int(base * scale))


def timeit(fn: Callable[[], object], reps: int, setup: Optional[Callable[[], object]] = None) -> Tuple[float, float]:
    """Ejecuta fn `reps` veces (setup fuera del cronómetro). Devuelve (media, desv. est.) en segundos."""
    times = []
    for _ in range(reps):
        arg = setup() if setup else None
        gc.collect()
        t = time.perf_counter()
        fn(arg) if setup else fn()
        times.append(time.perf_counter() - t)
    return float(np.mean(times)), float(np.std(times))


def memory(fn: Callable[[], object]) -> Tuple[int, int, object]:
    """(bytes actuales retenidos por el resultado, pico, resultado) medido con tracemalloc."""
    gc.collect()
    tracemalloc.start()
    result = fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return current, peak, result


class Recorder:
    def __init__(self, name: str) -> None:
        os.makedirs(OUT_DIR, exist_ok=True)
        self.path = os.path.join(OUT_DIR, f"{name}.csv")
        self.rows: List[Dict] = []

    def add(self, **row) -> None:
        row = {k: (round(v, 9) if isinstance(v, float) else v) for k, v in row.items()}
        self.rows.append(row)
        print("   ", ", ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        self.save()

    def save(self) -> None:
        keys: List[str] = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


def save_platform() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    info = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(), "cpus": os.cpu_count(),
            "seed": SEED, "reps": REPS}
    try:
        with open("/proc/meminfo") as f:
            info["memory_kb"] = int(f.readline().split()[1])
    except OSError:
        pass
    with open(os.path.join(OUT_DIR, "platform.json"), "w") as f:
        json.dump(info, f, indent=2)
