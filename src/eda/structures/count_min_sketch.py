"""Etapa 6 [FIJA] - Count-Min Sketch (Cormode y Muthukrishnan, 2005).

Matriz de d filas x w contadores; cada fila tiene su función hash. update suma
c en una celda por fila; estimate devuelve el mínimo de las d celdas. Nunca
subestima (solo hay colisiones que suman) y, con w = ceil(e/eps) y
d = ceil(ln(1/delta)):

    f_real <= f_estimado <= f_real + eps * N   con probabilidad >= 1 - delta

donde N es la suma de todas las actualizaciones. La memoria es O(w·d),
independiente del número de claves distintas.

WindowedCountMin mantiene una ventana deslizante con B sub-sketches (uno por
intervalo de tiempo, en anillo): al avanzar el tiempo se limpia el más
antiguo. Así se estima "intentos de pago por tarjeta en los últimos 10 min".
"""
from __future__ import annotations

import math
from typing import Iterable, Union

import numpy as np

from .hashing import splitmix64, splitmix64_vec, to_u64

_SEEDS = [0x51ED270B, 0x2545F491, 0x6C8E9CF5, 0x9E3779B9, 0x7F4A7C15, 0x85EBCA6B,
          0xC2B2AE35, 0x27D4EB2F, 0x165667B1, 0xD3A2646C, 0xFD7046C5, 0xB55A4F09]


class CountMinSketch:
    def __init__(self, eps: float = 0.001, delta: float = 0.01, width: int = 0, depth: int = 0,
                 conservative: bool = False) -> None:
        self.w = width or int(math.ceil(math.e / eps))
        self.d = depth or int(math.ceil(math.log(1 / delta)))
        if self.d > len(_SEEDS):
            raise ValueError("depth máximo %d" % len(_SEEDS))
        self.eps = math.e / self.w
        self.delta = math.exp(-self.d)
        self.conservative = conservative  # conservative update: reduce sobreestimación
        self.table = np.zeros((self.d, self.w), dtype=np.uint32)
        self.total = 0

    def _cols(self, key: Union[int, str]):
        x = to_u64(key)
        w = self.w
        return [splitmix64(x ^ s) % w for s in _SEEDS[: self.d]]

    def update(self, key: Union[int, str], count: int = 1) -> None:
        cols = self._cols(key)
        t = self.table
        if self.conservative:
            cur = min(int(t[r, c]) for r, c in enumerate(cols))
            for r, c in enumerate(cols):
                if t[r, c] < cur + count:
                    t[r, c] = cur + count
        else:
            for r, c in enumerate(cols):
                t[r, c] += count
        self.total += count

    def estimate(self, key: Union[int, str]) -> int:
        t = self.table
        return int(min(t[r, c] for r, c in enumerate(self._cols(key))))

    def update_and_estimate(self, key: Union[int, str], count: int = 1) -> int:
        self.update(key, count)
        return self.estimate(key)

    # ------------------------------------------------------------- vectorizado
    def _cols_vec(self, keys: np.ndarray) -> np.ndarray:
        x = keys.astype(np.uint64)
        return np.stack([(splitmix64_vec(x ^ np.uint64(s)) % np.uint64(self.w)).astype(np.int64)
                         for s in _SEEDS[: self.d]])

    def update_many(self, keys: Iterable[int], chunk: int = 1 << 20) -> None:
        keys = np.asarray(keys, dtype=np.int64)
        for s in range(0, keys.size, chunk):
            cols = self._cols_vec(keys[s:s + chunk])
            for r in range(self.d):
                self.table[r] += np.bincount(cols[r], minlength=self.w).astype(np.uint32)
        self.total += int(keys.size)

    def estimate_many(self, keys: Iterable[int]) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.int64)
        cols = self._cols_vec(keys)
        return np.min(self.table[np.arange(self.d)[:, None], cols], axis=0)

    def memory_bytes(self) -> int:
        return int(self.table.nbytes)

    def error_bound(self) -> float:
        return self.eps * self.total


class WindowedCountMin:
    """Ventana deslizante de `window` segundos dividida en `buckets` intervalos."""

    def __init__(self, window_seconds: int = 600, buckets: int = 10, eps: float = 0.001,
                 delta: float = 0.01) -> None:
        self.window = window_seconds
        self.B = buckets
        self.span = window_seconds / buckets
        self.sketches = [CountMinSketch(eps, delta) for _ in range(buckets)]
        self.bucket_id = [-1] * buckets  # intervalo absoluto que ocupa cada casilla

    def _slot(self, ts: float) -> int:
        b = int(ts // self.span)
        i = b % self.B
        if self.bucket_id[i] != b:  # la casilla tenía un intervalo vencido: se limpia
            self.sketches[i].table.fill(0)
            self.sketches[i].total = 0
            self.bucket_id[i] = b
        return i

    def update(self, key: Union[int, str], ts: float, count: int = 1) -> None:
        self.sketches[self._slot(ts)].update(key, count)

    def estimate(self, key: Union[int, str], ts: float) -> int:
        b_now = int(ts // self.span)
        total = 0
        for i in range(self.B):
            if b_now - self.B < self.bucket_id[i] <= b_now:
                total += self.sketches[i].estimate(key)
        return total

    def memory_bytes(self) -> int:
        return sum(s.memory_bytes() for s in self.sketches)
