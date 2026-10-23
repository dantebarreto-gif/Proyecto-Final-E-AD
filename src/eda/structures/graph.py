"""Etapa 8 [FIJA] - Grafo de co-compra: PageRank y detección de comunidades.

Grafo no dirigido y ponderado: nodos = productos, peso(u,v) = nº de órdenes
(sesiones con compra) que contienen u y v. Se guarda como matriz de
adyacencia CSR (estructura de la etapa 3), simétrica.

- pagerank / personalized_pagerank: iteración de potencias (Brin y Page, 1998)
      r <- (1-alpha) * v + alpha * (P^T r + masa_colgante * v)
  con P = matriz de transición por pesos. O(iter · m). Baseline: grado ponderado.
- louvain: optimización greedy de modularidad en dos fases (Blondel et al.,
  2008): movimiento local + agregación, repetido mientras mejore Q.
- label_propagation: cada nodo adopta la etiqueta de mayor peso entre sus
  vecinos (Raghavan et al., 2007), casi lineal. Alternativa más simple.
- modularity, nmi: métricas para comparar calidad y estabilidad.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .sparse_matrix import CSRMatrix


class CoPurchaseGraph:
    """Acumula aristas desde el stream (órdenes) y construye la CSR simétrica."""

    def __init__(self, n_nodes: int) -> None:
        self.n = n_nodes
        self._edges: Dict[int, float] = {}
        self.orders = 0

    def add_order(self, items: Iterable[int], max_items: int = 20) -> int:
        items = sorted(set(items))[:max_items]
        added = 0
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                key = items[a] * self.n + items[b]
                self._edges[key] = self._edges.get(key, 0.0) + 1.0
                added += 1
        self.orders += 1
        return added

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    def to_csr(self) -> CSRMatrix:
        if not self._edges:
            return CSRMatrix(np.zeros(self.n + 1, dtype=np.int64), np.empty(0, dtype=np.int64),
                             np.empty(0), (self.n, self.n))
        keys = np.fromiter(self._edges.keys(), dtype=np.int64, count=len(self._edges))
        w = np.fromiter(self._edges.values(), dtype=np.float64, count=len(self._edges))
        u, v = keys // self.n, keys % self.n
        return symmetric_csr(u, v, w, self.n)


def symmetric_csr(u: np.ndarray, v: np.ndarray, w: np.ndarray, n: int) -> CSRMatrix:
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    vals = np.concatenate([w, w])
    return CSRMatrix.from_coo(rows, cols, vals, (n, n))


# ------------------------------------------------------------------ PageRank
def pagerank(A: CSRMatrix, alpha: float = 0.85, tol: float = 1e-9, max_iter: int = 100,
             personalization: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """Devuelve (puntajes, iteraciones). Converge en norma L1 < tol."""
    n = A.shape[0]
    out_w = A.row_sums()
    dangling = out_w == 0
    inv = np.divide(1.0, out_w, out=np.zeros(n), where=~dangling)
    v = np.full(n, 1.0 / n) if personalization is None else personalization / personalization.sum()
    rows = A.row_ids()
    # P^T r: cada arista (i->j) aporta r[i] * w_ij / out_w[i] al nodo j
    edge_w = A.data * inv[rows]
    r = v.copy()
    for it in range(1, max_iter + 1):
        contrib = np.bincount(A.indices, weights=edge_w * r[rows], minlength=n)
        r_new = alpha * (contrib + r[dangling].sum() * v) + (1 - alpha) * v
        err = np.abs(r_new - r).sum()
        r = r_new
        if err < tol:
            break
    return r, it


def personalized_pagerank(A: CSRMatrix, seeds: List[int], alpha: float = 0.85) -> np.ndarray:
    p = np.zeros(A.shape[0])
    p[seeds] = 1.0
    return pagerank(A, alpha=alpha, personalization=p)[0]


def degree_ranking(A: CSRMatrix) -> np.ndarray:
    """Baseline: grado ponderado (fuerza) de cada nodo."""
    return A.row_sums()


# ----------------------------------------------------------------- Modularidad
def modularity(A: CSRMatrix, labels: np.ndarray) -> float:
    """Q = sum_c [ L_c / m2 - (d_c / m2)^2 ], con m2 = suma total de pesos (2m)."""
    deg = A.row_sums()
    m2 = deg.sum()
    if m2 == 0:
        return 0.0
    rows = A.row_ids()
    same = labels[rows] == labels[A.indices]
    internal = np.bincount(labels[rows][same], weights=A.data[same], minlength=labels.max() + 1)
    tot = np.bincount(labels, weights=deg, minlength=labels.max() + 1)
    return float((internal / m2 - (tot / m2) ** 2).sum())


# --------------------------------------------------------------------- Louvain
def louvain(A: CSRMatrix, seed: int = 0, max_levels: int = 10, min_gain: float = 1e-7) -> np.ndarray:
    """Etiquetas de comunidad por el método de Louvain."""
    rng = np.random.default_rng(seed)
    n0 = A.shape[0]
    membership = np.arange(n0)
    G = A
    for _ in range(max_levels):
        part, moved = _louvain_local(G, rng, min_gain)
        if not moved:
            break
        # renumerar y agregar: cada comunidad pasa a ser un nodo
        _, part = np.unique(part, return_inverse=True)
        membership = part[membership]
        rows = G.row_ids()
        k = part.max() + 1
        G = CSRMatrix.from_coo(part[rows], part[G.indices], G.data, (k, k))
        if k == 1:
            break
    _, membership = np.unique(membership, return_inverse=True)
    return membership


def _louvain_local(G: CSRMatrix, rng: np.random.Generator, min_gain: float) -> Tuple[np.ndarray, bool]:
    n = G.shape[0]
    indptr, indices, data = G.indptr.tolist(), G.indices.tolist(), G.data.tolist()
    deg = G.row_sums()
    m2 = float(deg.sum())
    if m2 == 0:
        return np.arange(n), False
    deg_l = deg.tolist()
    comm = list(range(n))
    tot = deg_l[:]  # suma de grados por comunidad
    moved_any = False
    improved = True
    while improved:
        improved = False
        for i in rng.permutation(n).tolist():
            ci = comm[i]
            ki = deg_l[i]
            # pesos de i hacia cada comunidad vecina
            links: Dict[int, float] = {}
            self_loop = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if j == i:
                    self_loop += data[p]
                    continue
                cj = comm[j]
                links[cj] = links.get(cj, 0.0) + data[p]
            tot[ci] -= ki  # sacar i de su comunidad
            best_c = ci
            best_gain = links.get(ci, 0.0) - tot[ci] * ki / m2
            for c, w in links.items():
                gain = w - tot[c] * ki / m2  # ganancia de modularidad (x m2/2)
                if gain > best_gain + min_gain:
                    best_gain, best_c = gain, c
            tot[best_c] += ki
            if best_c != ci:
                comm[i] = best_c
                improved = True
                moved_any = True
    return np.array(comm), moved_any


# ------------------------------------------------------------ Label Propagation
def label_propagation(A: CSRMatrix, seed: int = 0, max_iter: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    indptr, indices, data = A.indptr.tolist(), A.indices.tolist(), A.data.tolist()
    labels = list(range(n))
    for _ in range(max_iter):
        changed = 0
        for i in rng.permutation(n).tolist():
            a, b = indptr[i], indptr[i + 1]
            if a == b:
                continue
            score: Dict[int, float] = {}
            for p in range(a, b):
                lj = labels[indices[p]]
                score[lj] = score.get(lj, 0.0) + data[p]
            best = max(score.values())
            ties = [l for l, s in score.items() if s == best]
            new = ties[0] if len(ties) == 1 else ties[int(rng.integers(len(ties)))]
            if new != labels[i] and score.get(labels[i], -1) < best:
                labels[i] = new
                changed += 1
        if changed == 0:
            break
    _, out = np.unique(np.array(labels), return_inverse=True)
    return out


# ----------------------------------------------------------------- Estabilidad
def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Información mutua normalizada entre dos particiones (estabilidad entre semillas)."""
    n = a.size
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    key = a.astype(np.int64) * (b.max() + 1) + b
    _, joint = np.unique(key, return_counts=True)
    pa = np.bincount(a) / n
    pb = np.bincount(b) / n
    pab = joint / n
    ha = -(pa * np.log(pa)).sum()
    hb = -(pb * np.log(pb)).sum()
    pairs = np.unique(key)
    ia, ib = pairs // (b.max() + 1), pairs % (b.max() + 1)
    mi = (pab * np.log(pab / (pa[ia] * pb[ib]))).sum()
    denom = (ha + hb) / 2
    return float(mi / denom) if denom > 0 else 1.0


def community_sizes(labels: np.ndarray) -> np.ndarray:
    return np.sort(np.bincount(labels))[::-1]
