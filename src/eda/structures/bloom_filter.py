"""Etapa 6 [FIJA] - Bloom Filter (Bloom, 1970).

Arreglo de m bits y k funciones hash. add pone k bits en 1; contains responde
"posiblemente presente" si los k bits están en 1, o "seguro ausente" si alguno
es 0. No hay falsos negativos; la tasa de falsos positivos con n elementos es

    p ~ (1 - e^{-kn/m})^k,  óptimo con  m = -n ln p / (ln 2)^2,  k = (m/n) ln 2.

Las k posiciones se generan con doble hashing (Kirsch y Mitzenmacher, 2006):
g_i(x) = h1(x) + i·h2(x) mod m, que conserva la tasa asintótica de FP con solo
dos hashes. Los bits se guardan empaquetados (8 por byte) en un arreglo numpy:
para 10M elementos y p = 1 % son ~11.4 MB frente a cientos de MB de un set.
"""
from __future__ import annotations

import math
from typing import Iterable, Union

import numpy as np

from .hashing import M64, splitmix64, splitmix64_vec, to_u64


class BloomFilter:
    def __init__(self, capacity: int, fp_rate: float = 0.01, seed: int = 0) -> None:
        if capacity <= 0 or not 0 < fp_rate < 1:
            raise ValueError("capacity > 0 y 0 < fp_rate < 1")
        self.capacity = capacity
        self.fp_rate = fp_rate
        self.m = max(8, int(math.ceil(-capacity * math.log(fp_rate) / (math.log(2) ** 2))))
        self.k = max(1, int(round(self.m / capacity * math.log(2))))
        self.seed = seed
        self.bits = np.zeros((self.m + 7) // 8, dtype=np.uint8)
        self.count = 0

    # ----------------------------------------------------------------- escalar
    def _positions(self, key: Union[int, str]):
        x = to_u64(key) ^ self.seed
        h1 = splitmix64(x)
        h2 = splitmix64(h1) | 1
        m = self.m
        for i in range(self.k):
            yield ((h1 + i * h2) & M64) % m

    def add(self, key: Union[int, str]) -> None:
        bits = self.bits
        for p in self._positions(key):
            bits[p >> 3] |= 1 << (p & 7)
        self.count += 1

    def __contains__(self, key: Union[int, str]) -> bool:
        bits = self.bits
        for p in self._positions(key):
            if not bits[p >> 3] & (1 << (p & 7)):
                return False
        return True

    # ------------------------------------------------------------- vectorizado
    def _positions_vec(self, keys: np.ndarray) -> np.ndarray:
        x = keys.astype(np.uint64) ^ np.uint64(self.seed)
        h1 = splitmix64_vec(x)
        h2 = splitmix64_vec(h1) | np.uint64(1)
        i = np.arange(self.k, dtype=np.uint64)
        with np.errstate(over="ignore"):
            pos = (h1[:, None] + i[None, :] * h2[:, None]) % np.uint64(self.m)
        return pos.astype(np.int64)  # forma (n, k)

    def add_many(self, keys: Iterable[int], chunk: int = 1 << 20) -> None:
        keys = np.asarray(keys, dtype=np.int64)
        for s in range(0, keys.size, chunk):
            pos = self._positions_vec(keys[s:s + chunk]).ravel()
            np.bitwise_or.at(self.bits, pos >> 3, (1 << (pos & 7)).astype(np.uint8))
        self.count += int(keys.size)

    def contains_many(self, keys: Iterable[int], chunk: int = 1 << 20) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.int64)
        out = np.empty(keys.size, dtype=bool)
        for s in range(0, keys.size, chunk):
            pos = self._positions_vec(keys[s:s + chunk])
            hit = (self.bits[pos >> 3] >> (pos & 7).astype(np.uint8)) & 1
            out[s:s + chunk] = hit.all(axis=1)
        return out

    # ------------------------------------------------------------------ métricas
    def expected_fp_rate(self) -> float:
        return (1 - math.exp(-self.k * self.count / self.m)) ** self.k

    def fill_ratio(self) -> float:
        return float(np.unpackbits(self.bits)[: self.m].mean())

    def memory_bytes(self) -> int:
        return int(self.bits.nbytes)
