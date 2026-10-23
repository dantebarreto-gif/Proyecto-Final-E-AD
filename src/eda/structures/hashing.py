"""Funciones hash deterministas compartidas por las estructuras probabilísticas.

- splitmix64: mezclador de 64 bits (Steele et al., 2014), escalar y vectorizado.
- fnv1a64: hash de cadenas (FNV-1a), para no depender del hash aleatorizado
  de Python (PYTHONHASHSEED) y que los experimentos sean reproducibles.
"""
from __future__ import annotations

from typing import Union

import numpy as np

M64 = (1 << 64) - 1
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def splitmix64_vec(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64) + _C1
        x = (x ^ (x >> np.uint64(30))) * _C2
        x = (x ^ (x >> np.uint64(27))) * _C3
        return x ^ (x >> np.uint64(31))


def fnv1a64(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & M64
    return h


def to_u64(key: Union[int, str]) -> int:
    """Clave arbitraria (entero o cadena) -> entero de 64 bits."""
    if isinstance(key, str):
        return fnv1a64(key)
    return int(key) & M64
