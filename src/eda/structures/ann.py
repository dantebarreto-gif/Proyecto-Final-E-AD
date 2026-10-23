"""Etapa 7 [FIJA] - Búsqueda aproximada de vecinos (ANN) sobre embeddings de producto.

Se implementan dos familias y se comparan con KNN exacto por fuerza bruta:

IVF (inverted file, Jégou et al., 2011): k-means divide el espacio en nlist
celdas de Voronoi; cada vector se guarda en la lista de su centroide. Una
consulta compara contra los nlist centroides y recorre solo las nprobe
listas más cercanas: O(nlist·d + nprobe·(n/nlist)·d). Es la opción del
pipeline porque escala a 10M vectores con operaciones vectorizadas.

HNSW (Malkov y Yashunin, 2020): grafo de proximidad navegable en capas.
Cada nodo recibe un nivel aleatorio geométrico; la búsqueda baja greedy por
las capas superiores (pocas aristas largas) y hace best-first con ef
candidatos en la capa 0. Inserción y consulta O(log n) esperado. Se
implementa en Python puro como referencia de calidad (recall alto con pocas
comparaciones) en escalas menores.

Los vectores se normalizan: distancia euclídea sobre vectores unitarios es
equivalente a similitud coseno (||a-b||^2 = 2 - 2 cos).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .heap import BinaryHeap


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return (x / n).astype(np.float32)


def exact_knn(data: np.ndarray, q: np.ndarray, k: int = 10) -> np.ndarray:
    """Fuerza bruta sobre vectores normalizados (máximo producto punto)."""
    sims = data @ q
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    return idx[np.argsort(-sims[idx], kind="stable")]


def kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0, sample: int = 100_000) -> np.ndarray:
    """Lloyd con inicialización k-means++ sobre una muestra (vectores normalizados)."""
    rng = np.random.default_rng(seed)
    xs = x[rng.choice(x.shape[0], min(sample, x.shape[0]), replace=False)]
    k = min(k, xs.shape[0])
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = xs[rng.integers(xs.shape[0])]
    d2 = np.full(xs.shape[0], np.inf, dtype=np.float32)
    for i in range(1, k):  # k-means++: prob. proporcional a la distancia^2
        d2 = np.minimum(d2, ((xs - centers[i - 1]) ** 2).sum(axis=1))
        p = d2 / d2.sum() if d2.sum() > 0 else None
        centers[i] = xs[rng.choice(xs.shape[0], p=p)]
    for _ in range(iters):
        assign = _assign(xs, centers)
        for c in range(k):
            members = xs[assign == c]
            if members.shape[0]:
                centers[c] = members.mean(axis=0)
        centers = normalize(centers)
    return centers


def _assign(x: np.ndarray, centers: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.int64)
    for s in range(0, x.shape[0], chunk):
        out[s:s + chunk] = np.argmax(x[s:s + chunk] @ centers.T, axis=1)
    return out


class IVFIndex:
    def __init__(self, nlist: int = 0, nprobe: int = 8, seed: int = 0) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed

    def build(self, data: np.ndarray) -> "IVFIndex":
        self.data = normalize(data)
        n = self.data.shape[0]
        if not self.nlist:
            self.nlist = max(1, int(4 * math.sqrt(n)))
        self.centers = kmeans(self.data, self.nlist, seed=self.seed)
        self.nlist = self.centers.shape[0]
        assign = _assign(self.data, self.centers)
        # listas invertidas en formato CSR: ids ordenados por lista
        self.ids = np.argsort(assign, kind="stable")
        counts = np.bincount(assign, minlength=self.nlist)
        self.offsets = np.zeros(self.nlist + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])
        self.list_data = self.data[self.ids]  # vectores contiguos por lista
        return self

    def query(self, q: np.ndarray, k: int = 10, nprobe: int = 0) -> np.ndarray:
        nprobe = nprobe or self.nprobe
        q = q.astype(np.float32)
        cs = self.centers @ q
        probe = np.argpartition(-cs, min(nprobe, self.nlist) - 1)[:nprobe]
        idx_parts, sim_parts = [], []
        for c in probe:
            a, b = self.offsets[c], self.offsets[c + 1]
            if b > a:
                idx_parts.append(self.ids[a:b])
                sim_parts.append(self.list_data[a:b] @ q)
        if not idx_parts:
            return np.empty(0, dtype=np.int64)
        idx = np.concatenate(idx_parts)
        sims = np.concatenate(sim_parts)
        k = min(k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        return idx[top[np.argsort(-sims[top], kind="stable")]]

    def memory_bytes(self) -> int:
        return self.list_data.nbytes + self.ids.nbytes + self.offsets.nbytes + self.centers.nbytes


class HNSWIndex:
    def __init__(self, M: int = 16, ef_construction: int = 100, ef_search: int = 50, seed: int = 0) -> None:
        self.M = M
        self.M0 = 2 * M
        self.efc = ef_construction
        self.ef = ef_search
        self.mL = 1 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.layers: List[Dict[int, List[int]]] = []
        self.entry = -1
        self.dist_evals = 0

    def _dists(self, q: np.ndarray, ids: List[int]) -> np.ndarray:
        self.dist_evals += len(ids)
        return 1.0 - self.data[ids] @ q  # distancia coseno

    def _search_layer(self, q: np.ndarray, entries: List[int], ef: int, layer: int) -> List[Tuple[float, int]]:
        adj = self.layers[layer]
        visited = set(entries)
        d0 = self._dists(q, entries)
        cand = BinaryHeap()       # min-heap por distancia: a quién expandir
        result = BinaryHeap()     # max-heap (prioridad -dist): los ef mejores
        for e, d in zip(entries, d0.tolist()):
            cand.push(d, e)
            result.push(-d, e)
        while len(cand):
            d, c = cand.pop()
            if d > -result.peek()[0] and len(result) >= ef:
                break
            neigh = [n for n in adj.get(c, ()) if n not in visited]
            if not neigh:
                continue
            visited.update(neigh)
            for n, dn in zip(neigh, self._dists(q, neigh).tolist()):
                if len(result) < ef or dn < -result.peek()[0]:
                    cand.push(dn, n)
                    result.push(-dn, n)
                    if len(result) > ef:
                        result.pop()
        return sorted((-p, i) for p, i in result.items())

    def _select(self, cands: List[Tuple[float, int]], m: int) -> List[int]:
        """Heurística de vecinos diversos (algoritmo 4 del paper)."""
        chosen: List[int] = []
        for d, c in cands:
            if len(chosen) >= m:
                break
            if not chosen or np.all(1.0 - self.data[chosen] @ self.data[c] > d):
                chosen.append(c)
        if len(chosen) < m:  # completar con los más cercanos restantes
            for _, c in cands:
                if c not in chosen:
                    chosen.append(c)
                    if len(chosen) >= m:
                        break
        return chosen

    def build(self, data: np.ndarray) -> "HNSWIndex":
        self.data = normalize(data)
        for i in range(self.data.shape[0]):
            self._insert(i)
        return self

    def _insert(self, i: int) -> None:
        q = self.data[i]
        level = int(-math.log(1.0 - self.rng.random()) * self.mL)
        while len(self.layers) <= level:
            self.layers.append({})
        for l in range(level + 1):
            self.layers[l][i] = []
        if self.entry < 0:
            self.entry = i
            self.top = level
            return
        ep = [self.entry]
        for l in range(self.top, level, -1):  # descenso greedy
            ep = [self._search_layer(q, ep, 1, l)[0][1]]
        for l in range(min(level, self.top), -1, -1):
            cands = self._search_layer(q, ep, self.efc, l)
            mmax = self.M0 if l == 0 else self.M
            neigh = self._select(cands, self.M)
            self.layers[l][i] = neigh
            for n in neigh:
                lst = self.layers[l][n]
                lst.append(i)
                if len(lst) > mmax:  # recorte manteniendo los más cercanos
                    d = self._dists(self.data[n], lst)
                    keep = np.argsort(d, kind="stable")[:mmax]
                    self.layers[l][n] = [lst[j] for j in keep.tolist()]
            ep = [c for _, c in cands]
        if level > self.top:
            self.entry, self.top = i, level

    def query(self, q: np.ndarray, k: int = 10, ef: int = 0) -> np.ndarray:
        q = normalize(q[None, :])[0]
        ef = max(ef or self.ef, k)
        ep = [self.entry]
        for l in range(self.top, 0, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]
        res = self._search_layer(q, ep, ef, 0)
        return np.array([i for _, i in res[:k]], dtype=np.int64)

    def memory_bytes(self) -> int:
        edges = sum(len(v) for layer in self.layers for v in layer.values())
        return self.data.nbytes + edges * 8


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    if truth.size == 0:
        return 1.0
    return len(set(found.tolist()) & set(truth.tolist())) / truth.size
