"""Etapa 2 - Cola de prioridad con heap binario (min-heap) en arreglo.

push y pop son O(log n); peek O(1); heapify O(n). Un contador de secuencia
desempata prioridades iguales para que el orden sea FIFO dentro de cada
prioridad (estabilidad), algo importante al procesar eventos de un stream.
También se usa en la etapa 9 para mantener el Top-K (min-heap de tamaño K).
"""
from __future__ import annotations

from typing import Any, Iterable, List, Tuple


class BinaryHeap:
    __slots__ = ("_a", "_seq")

    def __init__(self) -> None:
        self._a: List[Tuple[Any, int, Any]] = []
        self._seq = 0

    def __len__(self) -> int:
        return len(self._a)

    def push(self, priority: Any, item: Any) -> None:
        a = self._a
        a.append((priority, self._seq, item))
        self._seq += 1
        self._sift_up(len(a) - 1)

    def pop(self) -> Tuple[Any, Any]:
        a = self._a
        if not a:
            raise IndexError("pop de heap vacío")
        last = a.pop()
        if not a:
            return last[0], last[2]
        top = a[0]
        a[0] = last
        self._sift_down(0)
        return top[0], top[2]

    def peek(self) -> Tuple[Any, Any]:
        if not self._a:
            raise IndexError("peek de heap vacío")
        return self._a[0][0], self._a[0][2]

    def pushpop(self, priority: Any, item: Any) -> Tuple[Any, Any]:
        """Inserta y extrae el mínimo en un solo sift (útil para Top-K)."""
        a = self._a
        entry = (priority, self._seq, item)
        self._seq += 1
        if a and a[0] < entry:
            entry, a[0] = a[0], entry
            self._sift_down(0)
        return entry[0], entry[2]

    @classmethod
    def heapify(cls, pairs: Iterable[Tuple[Any, Any]]) -> "BinaryHeap":
        """Construcción en O(n) (Floyd)."""
        h = cls()
        h._a = [(p, i, it) for i, (p, it) in enumerate(pairs)]
        h._seq = len(h._a)
        for i in range(len(h._a) // 2 - 1, -1, -1):
            h._sift_down(i)
        return h

    def _sift_up(self, i: int) -> None:
        a = self._a
        x = a[i]
        while i > 0:
            parent = (i - 1) >> 1
            if x < a[parent]:
                a[i] = a[parent]
                i = parent
            else:
                break
        a[i] = x

    def _sift_down(self, i: int) -> None:
        a = self._a
        n = len(a)
        x = a[i]
        while True:
            child = 2 * i + 1
            if child >= n:
                break
            right = child + 1
            if right < n and a[right] < a[child]:
                child = right
            if a[child] < x:
                a[i] = a[child]
                i = child
            else:
                break
        a[i] = x

    def items(self) -> List[Tuple[Any, Any]]:
        return [(p, it) for p, _, it in self._a]


class TopK:
    """Mantiene los K elementos de mayor puntaje con un min-heap de tamaño K.

    Cada actualización es O(log K); el resultado ordenado es O(K log K).
    """

    __slots__ = ("k", "_heap")

    def __init__(self, k: int) -> None:
        self.k = k
        self._heap = BinaryHeap()

    def add(self, score: float, item: Any) -> None:
        if len(self._heap) < self.k:
            self._heap.push(score, item)
        elif score > self._heap.peek()[0]:
            self._heap.pushpop(score, item)

    def result(self) -> List[Tuple[Any, float]]:
        return [(it, s) for s, it in sorted(self._heap.items(), key=lambda t: -t[0])]


def top_k(scores: Iterable[Tuple[Any, float]], k: int) -> List[Tuple[Any, float]]:
    t = TopK(k)
    for item, s in scores:
        t.add(s, item)
    return t.result()
