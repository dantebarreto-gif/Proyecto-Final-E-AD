"""Etapa 3 - Caché LRU: tabla hash + lista doblemente enlazada.

La tabla hash da acceso O(1) al nodo; la lista mantiene el orden de uso
(cabeza = más reciente, cola = menos reciente). get y put son O(1).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .hash_table import HashTable


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Any = None, value: Any = None) -> None:
        self.key = key
        self.value = value
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class LRUCache:
    __slots__ = ("capacity", "_map", "_head", "_tail", "hits", "misses", "evictions", "on_evict")

    def __init__(self, capacity: int, on_evict: Optional[Callable[[Any, Any], None]] = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity debe ser > 0")
        self.capacity = capacity
        self._map = HashTable(capacity * 2)
        # centinelas: evitan casos especiales en inserción/borrado
        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head
        self.hits = self.misses = self.evictions = 0
        self.on_evict = on_evict  # p. ej. escribir a disco una página sucia

    def __len__(self) -> int:
        return len(self._map)

    def _unlink(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node: _Node) -> None:
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def get(self, key: Any, default: Any = None) -> Any:
        node = self._map.get(key)
        if node is None:
            self.misses += 1
            return default
        self.hits += 1
        self._unlink(node)
        self._push_front(node)
        return node.value

    def __contains__(self, key: Any) -> bool:
        return key in self._map

    def put(self, key: Any, value: Any) -> None:
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self._map) >= self.capacity:
            lru = self._tail.prev
            self._unlink(lru)
            self._map.remove(lru.key)
            self.evictions += 1
            if self.on_evict is not None:
                self.on_evict(lru.key, lru.value)
        node = _Node(key, value)
        self._map.put(key, node)
        self._push_front(node)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def items(self):
        """Pares (clave, valor) del más reciente al menos reciente."""
        node = self._head.next
        while node is not self._tail:
            yield node.key, node.value
            node = node.next
