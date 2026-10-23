"""Etapa 3 - Matriz dispersa (formato CSR) para interacciones usuario x producto.

Una matriz 200K x 50K densa ocuparía 80 GB en float64; en CSR solo se guardan
los nnz valores no nulos: data (nnz), indices (nnz) e indptr (filas + 1).

- DOKAccumulator: acumulador incremental (clave = fila * ncols + col) sobre
  la HashTable propia; recibe el stream evento a evento.
- CSRMatrix: estructura compacta e inmutable; fila i = data[indptr[i]:indptr[i+1]].
  matvec O(nnz), matmat O(nnz * k), transpuesta O(nnz + n).
- randomized_svd: SVD truncada aleatorizada (Halko et al., 2011) usando solo
  productos matriz-vector, para obtener embeddings de producto (etapa 7).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .hash_table import HashTable


class DOKAccumulator:
    __slots__ = ("nrows", "ncols", "_t")

    def __init__(self, nrows: int, ncols: int) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self._t = HashTable(1024)

    def add(self, row: int, col: int, value: float = 1.0) -> None:
        self._t.increment(row * self.ncols + col, value)

    def get(self, row: int, col: int) -> float:
        return self._t.get(row * self.ncols + col, 0.0)

    @property
    def nnz(self) -> int:
        return len(self._t)

    def to_csr(self) -> "CSRMatrix":
        n = len(self._t)
        keys = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=np.float64)
        for i, (k, v) in enumerate(self._t.items()):
            keys[i] = k
            vals[i] = v
        return CSRMatrix.from_coo(keys // self.ncols, keys % self.ncols, vals, (self.nrows, self.ncols))


class CSRMatrix:
    __slots__ = ("shape", "indptr", "indices", "data")

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, shape: Tuple[int, int]) -> None:
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.shape = shape

    @classmethod
    def from_coo(cls, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: Tuple[int, int],
                 sum_duplicates: bool = True) -> "CSRMatrix":
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float64)
        if sum_duplicates and rows.size:
            key = rows * shape[1] + cols
            order = np.argsort(key, kind="stable")
            key = key[order]
            vals = vals[order]
            first = np.ones(key.size, dtype=bool)
            first[1:] = key[1:] != key[:-1]
            group = np.cumsum(first) - 1
            vals = np.bincount(group, weights=vals)
            key = key[first]
            rows = key // shape[1]
            cols = key % shape[1]
        else:
            order = np.argsort(rows * shape[1] + cols, kind="stable")
            rows, cols, vals = rows[order], cols[order], vals[order]
        counts = np.bincount(rows, minlength=shape[0])
        indptr = np.zeros(shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return cls(indptr, cols, vals, shape)

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    def row(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.indptr[i], self.indptr[i + 1]
        return self.indices[a:b], self.data[a:b]

    def row_ids(self) -> np.ndarray:
        return np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.row_ids(), weights=self.data * x[self.indices], minlength=self.shape[0])

    def matmat(self, X: np.ndarray) -> np.ndarray:
        rid = self.row_ids()
        out = np.empty((self.shape[0], X.shape[1]))
        for j in range(X.shape[1]):
            out[:, j] = np.bincount(rid, weights=self.data * X[self.indices, j], minlength=self.shape[0])
        return out

    def transpose(self) -> "CSRMatrix":
        return CSRMatrix.from_coo(self.indices, self.row_ids(), self.data,
                                  (self.shape[1], self.shape[0]), sum_duplicates=False)

    def row_sums(self) -> np.ndarray:
        return np.bincount(self.row_ids(), weights=self.data, minlength=self.shape[0])

    def memory_bytes(self) -> int:
        return self.indptr.nbytes + self.indices.nbytes + self.data.nbytes

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape)
        out[self.row_ids(), self.indices] = self.data
        return out


def randomized_svd(A: CSRMatrix, k: int, n_oversample: int = 10, n_iter: int = 4,
                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD truncada aleatorizada: A ~ U diag(s) Vt con rango k."""
    rng = np.random.default_rng(seed)
    At = A.transpose()
    omega = rng.standard_normal((A.shape[1], k + n_oversample))
    Y = A.matmat(omega)
    Q, _ = np.linalg.qr(Y)
    for _ in range(n_iter):  # iteraciones de potencia: mejoran la precisión
        Z, _ = np.linalg.qr(At.matmat(Q))
        Q, _ = np.linalg.qr(A.matmat(Z))
    B = At.matmat(Q).T  # (k+p) x ncols
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], s[:k], Vt[:k]
