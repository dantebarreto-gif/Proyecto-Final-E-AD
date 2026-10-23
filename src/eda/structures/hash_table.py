"""Etapa 3 - Tabla hash con direccionamiento abierto (sondeo lineal).

- Capacidad potencia de 2 -> índice = hash & (cap - 1).
- Factor de carga máximo 0.7: al superarlo se duplica la capacidad (rehash
  O(n) amortizado, por lo que put es O(1) amortizado).
- Borrado con lápidas (tombstones) para no romper las cadenas de sondeo.
- El hash de Python se mezcla con un multiplicador de Fibonacci para
  dispersar enteros consecutivos (ids de producto/usuario).
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple

_EMPTY = object()
_TOMB = object()
_MASK64 = (1 << 64) - 1
_FIB = 11400714819323198485  # 2^64 / phi


def _mix(key: Any) -> int:
    return ((hash(key) & _MASK64) * _FIB) & _MASK64


class HashTable:
    __slots__ = ("_keys", "_vals", "_cap", "_size", "_used", "max_load", "probes")

    def __init__(self, capacity: int = 16, max_load: float = 0.7) -> None:
        cap = 8
        while cap < capacity:
            cap <<= 1
        self._cap = cap
        self._keys: List[Any] = [_EMPTY] * cap
        self._vals: List[Any] = [None] * cap
        self._size = 0   # claves vivas
        self._used = 0   # vivas + lápidas
        self.max_load = max_load
        self.probes = 0  # sondeos acumulados (para análisis)

    def __len__(self) -> int:
        return self._size

    def _slot(self, key: Any) -> Tuple[int, bool]:
        """Devuelve (índice, encontrado). Si no está, índice de inserción."""
        mask = self._cap - 1
        i = (_mix(key) >> 20) & mask
        keys = self._keys
        first_tomb = -1
        while True:
            k = keys[i]
            if k is _EMPTY:
                return (first_tomb if first_tomb >= 0 else i), False
            if k is _TOMB:
                if first_tomb < 0:
                    first_tomb = i
            elif k == key:
                return i, True
            i = (i + 1) & mask
            self.probes += 1

    def put(self, key: Any, value: Any) -> None:
        i, found = self._slot(key)
        if found:
            self._vals[i] = value
            return
        if self._keys[i] is _EMPTY:
            self._used += 1
        self._keys[i] = key
        self._vals[i] = value
        self._size += 1
        if self._used > self.max_load * self._cap:
            self._resize(self._cap * 2 if self._size > self._cap * self.max_load / 2 else self._cap)

    __setitem__ = put

    def get(self, key: Any, default: Any = None) -> Any:
        i, found = self._slot(key)
        return self._vals[i] if found else default

    def __getitem__(self, key: Any) -> Any:
        i, found = self._slot(key)
        if not found:
            raise KeyError(key)
        return self._vals[i]

    def __contains__(self, key: Any) -> bool:
        return self._slot(key)[1]

    def remove(self, key: Any) -> bool:
        i, found = self._slot(key)
        if not found:
            return False
        self._keys[i] = _TOMB
        self._vals[i] = None
        self._size -= 1
        return True

    def increment(self, key: Any, delta: int = 1) -> int:
        """Contador exacto: valor += delta (se usa como baseline del CMS)."""
        i, found = self._slot(key)
        if found:
            self._vals[i] += delta
            return self._vals[i]
        self.put(key, delta)
        return delta

    def _resize(self, new_cap: int) -> None:
        old_k, old_v = self._keys, self._vals
        self._cap = new_cap
        self._keys = [_EMPTY] * new_cap
        self._vals = [None] * new_cap
        self._size = 0
        self._used = 0
        for k, v in zip(old_k, old_v):
            if k is not _EMPTY and k is not _TOMB:
                self.put(k, v)

    def items(self) -> Iterator[Tuple[Any, Any]]:
        for k, v in zip(self._keys, self._vals):
            if k is not _EMPTY and k is not _TOMB:
                yield k, v

    def keys(self) -> Iterator[Any]:
        for k, _ in self.items():
            yield k

    def values(self) -> Iterator[Any]:
        for _, v in self.items():
            yield v

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def load_factor(self) -> float:
        return self._size / self._cap


class HashSet:
    """Conjunto exacto sobre HashTable: baseline del Bloom Filter."""

    __slots__ = ("_t",)

    def __init__(self, capacity: int = 16) -> None:
        self._t = HashTable(capacity)

    def add(self, key: Any) -> None:
        self._t.put(key, True)

    def __contains__(self, key: Any) -> bool:
        return key in self._t

    def __len__(self) -> int:
        return len(self._t)

    def discard(self, key: Any) -> None:
        self._t.remove(key)
