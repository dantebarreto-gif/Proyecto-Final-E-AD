"""Etapa 4 - B+Tree persistido en disco (índice principal).

Formato del archivo: páginas de tamaño fijo (4 KB por defecto).
- Página 0: cabecera (magic, raíz, nº de páginas, altura, nº de claves).
- Hoja:    [tipo=1][n][siguiente hoja][claves int64 x n][valores int64 x n]
- Interna: [tipo=0][n][  sin uso    ][claves int64 x n][hijos int64 x n+1]

Con páginas de 4 KB y claves/valores de 8 bytes caben 255 entradas por hoja y
254 claves por nodo interno, así que 10M claves caben en un árbol de altura 3:
una búsqueda puntual lee 3 páginas y una consulta por rango lee
O(log_B n + k/B) páginas, encadenando las hojas.

Las páginas se mantienen en un buffer pool LRU (estructura de la etapa 3):
al expulsar una página sucia se escribe a disco (write-back).
"""
from __future__ import annotations

import os
import struct
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .lru_cache import LRUCache
from .sorting import lower_bound

_MAGIC = b"BPT1"
_HDR = struct.Struct("<4sqqqqq")   # magic, root, n_pages, height, n_keys, page_size
_NODE = struct.Struct("<BxHxxxxq")  # is_leaf, n, next_leaf  (16 bytes)


class _Node:
    __slots__ = ("pid", "is_leaf", "keys", "vals", "next", "dirty")

    def __init__(self, pid: int, is_leaf: bool) -> None:
        self.pid = pid
        self.is_leaf = is_leaf
        self.keys: List[int] = []
        self.vals: List[int] = []  # valores (hoja) o hijos (interno)
        self.next = -1
        self.dirty = True


class BPlusTree:
    def __init__(self, path: str, page_size: int = 4096, cache_pages: int = 1024,
                 create: bool = True) -> None:
        self.path = path
        self.page_size = page_size
        self.max_leaf = (page_size - _NODE.size) // 16
        self.max_internal = (page_size - _NODE.size - 8) // 16
        self.reads = 0
        self.writes = 0
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        if exists and not create:
            self._f = open(path, "r+b")
            magic, self.root, self.n_pages, self.height, self.n_keys, ps = _HDR.unpack(
                self._f.read(_HDR.size))
            if magic != _MAGIC or ps != page_size:
                raise ValueError("archivo B+Tree inválido")
        else:
            self._f = open(path, "w+b")
            self.n_pages = 1
            self.n_keys = 0
            self.height = 1
            root = self._new_node(is_leaf=True)
            self.root = root.pid
        self._pool = LRUCache(cache_pages, on_evict=self._evict)

    # ---------------------------------------------------------------- páginas
    def _new_node(self, is_leaf: bool) -> _Node:
        node = _Node(self.n_pages, is_leaf)
        self.n_pages += 1
        if hasattr(self, "_pool"):
            self._pool.put(node.pid, node)
        else:
            self._write(node)
        return node

    def _evict(self, pid: int, node: _Node) -> None:
        if node.dirty:
            self._write(node)

    def _write(self, node: _Node) -> None:
        n = len(node.keys)
        buf = bytearray(self.page_size)
        _NODE.pack_into(buf, 0, 1 if node.is_leaf else 0, n, node.next)
        off = _NODE.size
        k = np.asarray(node.keys, dtype=np.int64).tobytes()
        v = np.asarray(node.vals, dtype=np.int64).tobytes()
        buf[off:off + len(k)] = k
        buf[off + len(k):off + len(k) + len(v)] = v
        self._f.seek(node.pid * self.page_size)
        self._f.write(buf)
        node.dirty = False
        self.writes += 1

    def _read(self, pid: int) -> _Node:
        node = self._pool.get(pid)
        if node is not None:
            return node
        self._f.seek(pid * self.page_size)
        buf = self._f.read(self.page_size)
        self.reads += 1
        is_leaf, n, nxt = _NODE.unpack_from(buf, 0)
        node = _Node(pid, bool(is_leaf))
        off = _NODE.size
        node.keys = np.frombuffer(buf, dtype=np.int64, count=n, offset=off).tolist()
        nv = n if is_leaf else n + 1
        node.vals = np.frombuffer(buf, dtype=np.int64, count=nv, offset=off + 8 * n).tolist()
        node.next = nxt
        node.dirty = False
        self._pool.put(pid, node)
        return node

    def _touch(self, node: _Node) -> None:
        node.dirty = True
        self._pool.put(node.pid, node)

    # -------------------------------------------------------------- consultas
    def _find_leaf(self, key: int) -> Tuple[_Node, List[Tuple[_Node, int]]]:
        path = []
        node = self._read(self.root)
        while not node.is_leaf:
            # hijo i contiene claves en [keys[i-1], keys[i])
            i = _upper_bound(node.keys, key)
            path.append((node, i))
            node = self._read(node.vals[i])
        return node, path

    def search(self, key: int) -> Optional[int]:
        leaf, _ = self._find_leaf(key)
        i = lower_bound(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return leaf.vals[i]
        return None

    def range_search(self, lo: int, hi: int, limit: Optional[int] = None) -> Iterator[Tuple[int, int]]:
        """Pares (clave, valor) con lo <= clave <= hi, en orden."""
        leaf, _ = self._find_leaf(lo)
        i = lower_bound(leaf.keys, lo)
        count = 0
        while True:
            while i < len(leaf.keys):
                k = leaf.keys[i]
                if k > hi:
                    return
                yield k, leaf.vals[i]
                count += 1
                if limit is not None and count >= limit:
                    return
                i += 1
            if leaf.next < 0:
                return
            leaf = self._read(leaf.next)
            i = 0

    # -------------------------------------------------------------- inserción
    def insert(self, key: int, value: int) -> None:
        """Inserta o actualiza. O(log_B n) páginas."""
        leaf, path = self._find_leaf(key)
        i = lower_bound(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            leaf.vals[i] = value
            self._touch(leaf)
            return
        leaf.keys.insert(i, key)
        leaf.vals.insert(i, value)
        self.n_keys += 1
        self._touch(leaf)
        if len(leaf.keys) <= self.max_leaf:
            return
        # split de hoja: la mitad derecha va a una hoja nueva
        mid = len(leaf.keys) // 2
        right = self._new_node(is_leaf=True)
        right.keys, leaf.keys = leaf.keys[mid:], leaf.keys[:mid]
        right.vals, leaf.vals = leaf.vals[mid:], leaf.vals[:mid]
        right.next, leaf.next = leaf.next, right.pid
        self._touch(right)
        self._insert_in_parent(path, right.keys[0], right.pid)

    def _insert_in_parent(self, path: List[Tuple[_Node, int]], sep: int, right_pid: int) -> None:
        while True:
            if not path:  # se divide la raíz: el árbol crece en altura
                new_root = self._new_node(is_leaf=False)
                new_root.keys = [sep]
                new_root.vals = [self.root, right_pid]
                self._touch(new_root)
                self.root = new_root.pid
                self.height += 1
                return
            parent, idx = path.pop()
            parent.keys.insert(idx, sep)
            parent.vals.insert(idx + 1, right_pid)
            self._touch(parent)
            if len(parent.keys) <= self.max_internal:
                return
            mid = len(parent.keys) // 2
            sep_up = parent.keys[mid]
            right = self._new_node(is_leaf=False)
            right.keys = parent.keys[mid + 1:]
            right.vals = parent.vals[mid + 1:]
            parent.keys = parent.keys[:mid]
            parent.vals = parent.vals[:mid + 1]
            self._touch(right)
            sep, right_pid = sep_up, right.pid

    def bulk_load(self, keys: np.ndarray, values: np.ndarray, fill: float = 0.9) -> None:
        """Construcción ascendente desde claves ordenadas y únicas: O(n), escritura secuencial."""
        if self.n_keys:
            raise ValueError("bulk_load requiere un árbol vacío")
        keys = np.asarray(keys, dtype=np.int64)
        values = np.asarray(values, dtype=np.int64)
        if keys.size == 0:
            return
        if keys.size > 1 and not np.all(keys[1:] > keys[:-1]):
            raise ValueError("bulk_load requiere claves ordenadas y únicas")
        per_leaf = max(2, int(self.max_leaf * fill))
        level: List[Tuple[int, int]] = []  # (primera clave, pid)
        self.n_pages = 1  # se descarta la raíz vacía inicial
        self._pool = LRUCache(self._pool.capacity, on_evict=self._evict)
        prev: Optional[_Node] = None
        for s in range(0, keys.size, per_leaf):
            node = _Node(self.n_pages, True)
            self.n_pages += 1
            node.keys = keys[s:s + per_leaf].tolist()
            node.vals = values[s:s + per_leaf].tolist()
            if prev is not None:
                prev.next = node.pid
                self._write(prev)
            prev = node
            level.append((node.keys[0], node.pid))
        self._write(prev)
        height = 1
        per_node = max(2, int(self.max_internal * fill))
        while len(level) > 1:
            nxt: List[Tuple[int, int]] = []
            for s in range(0, len(level), per_node + 1):
                group = level[s:s + per_node + 1]
                node = _Node(self.n_pages, False)
                self.n_pages += 1
                node.vals = [pid for _, pid in group]
                node.keys = [k for k, _ in group[1:]]
                self._write(node)
                nxt.append((group[0][0], node.pid))
            level = nxt
            height += 1
        self.root = level[0][1]
        self.height = height
        self.n_keys = int(keys.size)

    # ------------------------------------------------------------- persistencia
    def flush(self) -> None:
        for _, node in list(self._pool.items()):
            if node.dirty:
                self._write(node)
        self._f.seek(0)
        hdr = bytearray(self.page_size)
        _HDR.pack_into(hdr, 0, _MAGIC, self.root, self.n_pages, self.height, self.n_keys, self.page_size)
        self._f.write(hdr)
        self._f.flush()

    def close(self) -> None:
        self.flush()
        self._f.close()

    def __len__(self) -> int:
        return self.n_keys

    def file_size(self) -> int:
        return self.n_pages * self.page_size


def _upper_bound(arr: List[int], key: int) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] <= key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def price_key(price: float, product_idx: int) -> int:
    """Clave compuesta (precio en centavos, índice de producto) -> int64 única."""
    return (int(round(price * 100)) << 24) | (product_idx & 0xFFFFFF)


def price_range_keys(lo_price: float, hi_price: float) -> Tuple[int, int]:
    return price_key(lo_price, 0), price_key(hi_price, 0xFFFFFF)
