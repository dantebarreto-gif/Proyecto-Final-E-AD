"""Etapa 7 [FIJA] - MinHash + LSH para similitud de Jaccard (Broder, 1997; Indyk y Motwani, 1998).

Cada producto se representa por el conjunto de usuarios que lo compraron o
agregaron al carrito. Similitud de Jaccard J(A,B) = |A ∩ B| / |A ∪ B|.

MinHash: con una permutación aleatoria h, P[min h(A) = min h(B)] = J(A,B).
Con P permutaciones, la fracción de coincidencias de las firmas estima J con
error estándar sqrt(J(1-J)/P).

LSH por bandas: la firma se corta en b bandas de r filas; dos conjuntos son
candidatos si coinciden en al menos una banda. P[candidato] = 1 - (1 - J^r)^b,
una curva en S con umbral aproximado (1/b)^(1/r). Con b=32, r=4 el umbral
es ~0.42. Cada banda se indexa como un arreglo ordenado de (hash de banda, id)
y se consulta con la búsqueda binaria propia (etapa 1): las b bandas
comparten un único arreglo ordenado de claves (hash de banda mezclado con el
número de banda), así una consulta son 2 búsquedas binarias vectorizadas.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .hashing import splitmix64_vec
from .heap import TopK
from .sorting import searchsorted_vec

_MAXU = np.iinfo(np.uint64).max


class MinHashLSH:
    def __init__(self, num_perm: int = 128, bands: int = 32, seed: int = 7) -> None:
        if num_perm % bands:
            raise ValueError("num_perm debe ser múltiplo de bands")
        self.P = num_perm
        self.b = bands
        self.r = num_perm // bands
        rng = np.random.default_rng(seed)
        self.seeds = rng.integers(1, 2 ** 63, size=num_perm, dtype=np.uint64)
        self.signatures: Optional[np.ndarray] = None
        self._keys = np.empty(0, dtype=np.uint64)
        self._ids = np.empty(0, dtype=np.int64)

    @property
    def threshold(self) -> float:
        return (1 / self.b) ** (1 / self.r)

    # ---------------------------------------------------------------- firmas
    def signature(self, elements: np.ndarray) -> np.ndarray:
        """Firma MinHash de un conjunto (arreglo de enteros)."""
        el = np.asarray(elements, dtype=np.uint64)
        if el.size == 0:
            return np.full(self.P, _MAXU, dtype=np.uint64)
        h = splitmix64_vec(el[None, :] ^ self.seeds[:, None])
        return h.min(axis=1)

    def signatures_csr(self, indptr: np.ndarray, indices: np.ndarray, chunk: int = 16) -> np.ndarray:
        """Firmas de n conjuntos dados en formato CSR (fila i = conjunto i).

        Para cada permutación se aplica el hash a todos los elementos y se toma
        el mínimo por fila con reduceat: O(P · nnz) vectorizado.
        """
        n = indptr.shape[0] - 1
        sig = np.full((n, self.P), _MAXU, dtype=np.uint64)
        nonempty = np.diff(indptr) > 0
        starts = indptr[:-1][nonempty]
        el = indices.astype(np.uint64)
        chunk = max(1, min(chunk, int(4e7 // max(1, el.size))))  # acota la memoria temporal
        for s in range(0, self.P, chunk):
            seeds = self.seeds[s:s + chunk]
            h = splitmix64_vec(el[None, :] ^ seeds[:, None])  # (chunk, nnz)
            if starts.size:
                sig[nonempty, s:s + seeds.size] = np.minimum.reduceat(h, starts, axis=1).T
        return sig

    # ------------------------------------------------------------------- LSH
    def _band_hash(self, sig: np.ndarray) -> np.ndarray:
        """Hash de cada banda: (n, b) uint64, combinando r filas con splitmix."""
        n = sig.shape[0]
        s = sig.reshape(n, self.b, self.r)
        acc = np.zeros((n, self.b), dtype=np.uint64)
        with np.errstate(over="ignore"):
            for j in range(self.r):
                acc = splitmix64_vec(acc ^ s[:, :, j])
        return acc

    def _salted(self, bh: np.ndarray) -> np.ndarray:
        """Hash de banda combinado con el número de banda: una sola tabla para las b bandas."""
        return splitmix64_vec(bh ^ self._salts[None, :])

    def index(self, signatures: np.ndarray) -> None:
        """Indexa todas las bandas en un único arreglo ordenado de (clave, id): O(n·b log(n·b))."""
        self.signatures = signatures
        n = signatures.shape[0]
        self._salts = splitmix64_vec(np.arange(1, self.b + 1, dtype=np.uint64))
        keys = self._salted(self._band_hash(signatures)).ravel()  # fila i, banda j -> i*b + j
        # los conjuntos vacíos (firma = máximo en todas las filas) no se indexan:
        # todos tendrían la misma firma y serían "similares" entre sí sin serlo
        empty = np.repeat((signatures == _MAXU).all(axis=1), self.b)
        order = np.argsort(keys, kind="stable")
        order = order[~empty[order]]
        self._keys = keys[order]
        self._ids = (order // self.b).astype(np.int64)

    def candidates(self, sig: np.ndarray) -> np.ndarray:
        """Ids que coinciden con la firma en al menos una banda (búsqueda binaria vectorizada)."""
        if (sig == _MAXU).all():
            return np.empty(0, dtype=np.int64)
        q = np.sort(self._salted(self._band_hash(sig[None, :]))[0])
        lo = searchsorted_vec(self._keys, q)
        hi = searchsorted_vec(self._keys, q + np.uint64(1))
        found = [self._ids[a:b] for a, b in zip(lo.tolist(), hi.tolist()) if b > a]
        if not found:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(found))

    def query(self, sig: np.ndarray, k: int = 10, exclude: int = -1) -> List[Tuple[int, float]]:
        """Top-K por Jaccard estimado entre los candidatos LSH."""
        cand = self.candidates(sig)
        if exclude >= 0:
            cand = cand[cand != exclude]
        if cand.size == 0:
            return []
        est = (self.signatures[cand] == sig[None, :]).mean(axis=1)
        top = TopK(k)
        for c, e in zip(cand.tolist(), est.tolist()):
            top.add(e, c)
        return top.result()

    def memory_bytes(self) -> int:
        sig = self.signatures.nbytes if self.signatures is not None else 0
        return sig + self._keys.nbytes + self._ids.nbytes


def jaccard_brute_force(indptr: np.ndarray, indices: np.ndarray, query_set: np.ndarray, k: int = 10,
                        exclude: int = -1) -> List[Tuple[int, float]]:
    """Línea base exacta: Jaccard de la consulta contra todos los conjuntos, O(nnz)."""
    n = indptr.shape[0] - 1
    q = np.unique(query_set)
    rows = np.repeat(np.arange(n), np.diff(indptr))
    inter = np.bincount(rows, weights=np.isin(indices, q), minlength=n)
    sizes = np.diff(indptr)
    union = sizes + q.size - inter
    jac = np.divide(inter, union, out=np.zeros(n), where=union > 0)
    if exclude >= 0:
        jac[exclude] = -1
    k = min(k, n)
    idx = np.argpartition(-jac, k - 1)[:k]
    idx = idx[np.argsort(-jac[idx], kind="stable")]
    return [(int(i), float(jac[i])) for i in idx if jac[i] > 0]
