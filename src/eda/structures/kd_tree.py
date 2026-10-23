"""Etapa 5 - KD-Tree para KNN exacto y consultas por rango multidimensional.

Se usa sobre pocas dimensiones (precio, popularidad, tasa de conversión), donde
el KD-Tree es eficiente: construcción O(n log n) con partición por mediana
(argpartition, O(n) por nivel) y KNN en O(log n) promedio para d pequeño.
En alta dimensión degrada a O(n) (maldición de la dimensionalidad), por eso
para embeddings de 32 dimensiones se usa ANN (etapa 7).

Representación compacta en arreglos (sin objetos por nodo):
  split_dim[i], split_val[i], left[i], right[i], start[i], end[i]
Los puntos se reordenan en `perm` para que cada hoja sea un rango contiguo.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .heap import BinaryHeap


class KDTree:
    def __init__(self, points: np.ndarray, leaf_size: int = 16) -> None:
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.n, self.d = self.points.shape
        self.leaf_size = leaf_size
        self.perm = np.arange(self.n)
        cap = 2 * max(1, self.n // max(1, leaf_size // 2)) + 1
        self.split_dim = np.full(cap, -1, dtype=np.int32)
        self.split_val = np.zeros(cap)
        self.left = np.full(cap, -1, dtype=np.int64)
        self.right = np.full(cap, -1, dtype=np.int64)
        self.start = np.zeros(cap, dtype=np.int64)
        self.end = np.zeros(cap, dtype=np.int64)
        self.lo = np.zeros((cap, self.d))  # caja envolvente del nodo
        self.hi = np.zeros((cap, self.d))
        self.n_nodes = 0
        if self.n:
            self._build()

    def _build(self) -> None:
        stack = [(self._alloc(0, self.n), 0, self.n)]
        while stack:
            node, s, e = stack.pop()
            pts = self.points[self.perm[s:e]]
            self.lo[node] = pts.min(axis=0)
            self.hi[node] = pts.max(axis=0)
            if e - s <= self.leaf_size:
                continue
            dim = int(np.argmax(self.hi[node] - self.lo[node]))  # eje de mayor extensión
            mid = (s + e) // 2
            idx = self.perm[s:e]
            part = np.argpartition(self.points[idx, dim], mid - s)
            self.perm[s:e] = idx[part]
            self.split_dim[node] = dim
            self.split_val[node] = self.points[self.perm[mid], dim]
            l = self._alloc(s, mid)
            r = self._alloc(mid, e)
            self.left[node], self.right[node] = l, r
            stack.append((l, s, mid))
            stack.append((r, mid, e))

    def _alloc(self, s: int, e: int) -> int:
        i = self.n_nodes
        if i >= self.split_dim.shape[0]:
            self._grow()
        self.start[i], self.end[i] = s, e
        self.n_nodes += 1
        return i

    def _grow(self) -> None:
        for name in ("split_dim", "split_val", "left", "right", "start", "end", "lo", "hi"):
            arr = getattr(self, name)
            fill = -1 if name in ("split_dim", "left", "right") else 0
            extra = np.full((arr.shape[0],) + arr.shape[1:], fill, dtype=arr.dtype)
            setattr(self, name, np.concatenate([arr, extra]))

    def _min_dist2(self, node: int, q: np.ndarray) -> float:
        diff = np.maximum(self.lo[node] - q, 0) + np.maximum(q - self.hi[node], 0)
        return float(diff @ diff)

    def query(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """K vecinos más cercanos exactos (distancia euclídea).

        Búsqueda best-first: un min-heap de nodos por distancia mínima a su
        caja y un max-heap acotado (prioridad = -dist) con los K mejores.
        """
        q = np.asarray(q, dtype=np.float64)
        best = BinaryHeap()  # (-dist2) -> índice; la raíz es el peor de los K
        frontier = BinaryHeap()
        frontier.push(self._min_dist2(0, q), 0)
        while len(frontier):
            dist, node = frontier.pop()
            if len(best) == k and dist >= -best.peek()[0]:
                break
            if self.split_dim[node] < 0:  # hoja
                idx = self.perm[self.start[node]:self.end[node]]
                diff = self.points[idx] - q
                d2 = np.einsum("ij,ij->i", diff, diff)
                for j in range(idx.shape[0]):
                    if len(best) < k:
                        best.push(-d2[j], int(idx[j]))
                    elif d2[j] < -best.peek()[0]:
                        best.pushpop(-d2[j], int(idx[j]))
            else:
                for child in (self.left[node], self.right[node]):
                    cd = self._min_dist2(child, q)
                    if len(best) < k or cd < -best.peek()[0]:
                        frontier.push(cd, int(child))
        pairs = sorted(((-p, i) for p, i in best.items()))
        return (np.array([i for _, i in pairs], dtype=np.int64),
                np.sqrt(np.array([d for d, _ in pairs])))

    def range_query(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Índices de los puntos dentro de la caja [lo, hi] (ortogonal)."""
        out: List[np.ndarray] = []
        stack = [0]
        while stack:
            node = stack.pop()
            if np.any(self.hi[node] < lo) or np.any(self.lo[node] > hi):
                continue
            if np.all(self.lo[node] >= lo) and np.all(self.hi[node] <= hi):
                out.append(self.perm[self.start[node]:self.end[node]])
                continue
            if self.split_dim[node] < 0:
                idx = self.perm[self.start[node]:self.end[node]]
                p = self.points[idx]
                out.append(idx[np.all((p >= lo) & (p <= hi), axis=1)])
            else:
                stack.append(int(self.left[node]))
                stack.append(int(self.right[node]))
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def memory_bytes(self) -> int:
        return sum(getattr(self, a).nbytes for a in
                   ("perm", "split_dim", "split_val", "left", "right", "start", "end", "lo", "hi"))


def knn_brute_force(points: np.ndarray, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Línea base exacta: distancia a todos los puntos, O(n·d)."""
    diff = points - q
    d2 = np.einsum("ij,ij->i", diff, diff)
    k = min(k, d2.shape[0])
    idx = np.argpartition(d2, k - 1)[:k]
    idx = idx[np.argsort(d2[idx], kind="stable")]
    return idx, np.sqrt(d2[idx])
