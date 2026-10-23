"""Etapa 1 - Ordenación eficiente y búsqueda binaria para limpiar y deduplicar.

- merge_sort: O(n log n), estable, para lotes de objetos Python con clave.
- merge_sorted_arrays: mezcla de dos arreglos ordenados en O(n + m log n)
  usando la búsqueda binaria vectorizada (mantiene la ventana de claves vistas).
- binary_search / searchsorted_vec: búsqueda binaria propia, escalar y
  vectorizada (log2(n) iteraciones sobre todo el arreglo de consultas).
- dedup_sorted: elimina duplicados adyacentes en un arreglo ordenado, O(n).
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import numpy as np


def merge_sort(items: List[Any], key: Optional[Callable[[Any], Any]] = None) -> List[Any]:
    """Merge sort bottom-up (iterativo) y estable. O(n log n) tiempo, O(n) memoria."""
    n = len(items)
    if n < 2:
        return list(items)
    keys = [key(x) for x in items] if key else list(items)
    idx = list(range(n))
    buf = [0] * n
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid = min(lo + width, n)
            hi = min(lo + 2 * width, n)
            i, j, k = lo, mid, lo
            while i < mid and j < hi:
                # <= mantiene la estabilidad
                if keys[idx[i]] <= keys[idx[j]]:
                    buf[k] = idx[i]
                    i += 1
                else:
                    buf[k] = idx[j]
                    j += 1
                k += 1
            while i < mid:
                buf[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                buf[k] = idx[j]
                j += 1
                k += 1
        idx, buf = buf, idx
        width *= 2
    return [items[i] for i in idx]


def binary_search(arr: Sequence[Any], target: Any) -> int:
    """Índice de target en arr ordenado, o -1. O(log n)."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        v = arr[mid]
        if v == target:
            return mid
        if v < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def lower_bound(arr: Sequence[Any], target: Any) -> int:
    """Primer índice i con arr[i] >= target. O(log n)."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def searchsorted_vec(sorted_arr: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Búsqueda binaria vectorizada (lower bound) para muchas consultas.

    Ejecuta ceil(log2(n+1)) iteraciones; en cada una todas las consultas
    avanzan un paso. Tiempo O(m log n) con m consultas.
    """
    n = sorted_arr.shape[0]
    lo = np.zeros(queries.shape[0], dtype=np.int64)
    hi = np.full(queries.shape[0], n, dtype=np.int64)
    while True:
        active = lo < hi
        if not active.any():
            break
        mid = (lo + hi) >> 1
        midc = np.minimum(mid, n - 1) if n else mid
        go_right = active & (sorted_arr[midc] < queries) if n else active & False
        go_left = active & ~go_right
        lo = np.where(go_right, mid + 1, lo)
        hi = np.where(go_left, mid, hi)
    return lo


def contains_sorted(sorted_arr: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Máscara booleana: qué consultas están en sorted_arr (búsqueda binaria)."""
    if sorted_arr.shape[0] == 0:
        return np.zeros(queries.shape[0], dtype=bool)
    pos = searchsorted_vec(sorted_arr, queries)
    posc = np.minimum(pos, sorted_arr.shape[0] - 1)
    return (pos < sorted_arr.shape[0]) & (sorted_arr[posc] == queries)


def dedup_sorted(sorted_arr: np.ndarray) -> np.ndarray:
    """Máscara de primeras apariciones en un arreglo ordenado. O(n)."""
    keep = np.ones(sorted_arr.shape[0], dtype=bool)
    if sorted_arr.shape[0] > 1:
        keep[1:] = sorted_arr[1:] != sorted_arr[:-1]
    return keep


def merge_sorted_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mezcla dos arreglos ordenados en uno ordenado (estable, a antes que b).

    Destino de b[j] = lower_bound(a, b[j]) + j; el resto de casillas es de a.
    """
    if a.shape[0] == 0:
        return b.copy()
    if b.shape[0] == 0:
        return a.copy()
    out = np.empty(a.shape[0] + b.shape[0], dtype=np.result_type(a, b))
    pos_b = searchsorted_vec(a, b) + np.arange(b.shape[0])
    mask = np.ones(out.shape[0], dtype=bool)
    mask[pos_b] = False
    out[pos_b] = b
    out[mask] = a
    return out
