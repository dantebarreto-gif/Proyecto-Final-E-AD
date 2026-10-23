"""Etapa 0 - Cola FIFO implementada con lista simplemente enlazada.

enqueue y dequeue son O(1): se mantiene puntero a la cabeza (salida) y a la
cola (entrada). Se usa como buffer de ingesta entre el productor del stream y
el consumidor por lotes.
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional


class _Node:
    __slots__ = ("value", "next")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.next: Optional[_Node] = None


class LinkedQueue:
    """Cola FIFO con capacidad opcional (backpressure)."""

    __slots__ = ("_head", "_tail", "_size", "capacity", "dropped")

    def __init__(self, capacity: Optional[int] = None) -> None:
        self._head: Optional[_Node] = None
        self._tail: Optional[_Node] = None
        self._size = 0
        self.capacity = capacity
        self.dropped = 0  # eventos rechazados por cola llena

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def enqueue(self, value: Any) -> bool:
        """Inserta al final. Devuelve False si la cola está llena. O(1)."""
        if self.capacity is not None and self._size >= self.capacity:
            self.dropped += 1
            return False
        node = _Node(value)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1
        return True

    def dequeue(self) -> Any:
        """Extrae el primero. O(1)."""
        if self._head is None:
            raise IndexError("dequeue de cola vacía")
        node = self._head
        self._head = node.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return node.value

    def peek(self) -> Any:
        if self._head is None:
            raise IndexError("peek de cola vacía")
        return self._head.value

    def dequeue_batch(self, k: int) -> List[Any]:
        """Extrae hasta k elementos en orden de llegada. O(k)."""
        out = []
        while self._head is not None and len(out) < k:
            out.append(self.dequeue())
        return out

    def __iter__(self) -> Iterator[Any]:
        node = self._head
        while node is not None:
            yield node.value
            node = node.next
