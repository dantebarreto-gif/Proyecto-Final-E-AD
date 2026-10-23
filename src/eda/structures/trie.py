"""Etapa 9 - Trie con pesos para autocompletado.

Cada nodo guarda sus hijos (por carácter), el peso del término que termina en
él y el peso máximo de su subárbol. Las sugerencias se obtienen con búsqueda
best-first usando el heap propio: se expande primero el subárbol con mayor
peso máximo, así que se encuentran las K mejores sin recorrer todo el
subárbol. Inserción O(L); consulta O(p + K·L·log) en la práctica.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .heap import BinaryHeap


class _TrieNode:
    __slots__ = ("children", "weight", "best", "term")

    def __init__(self) -> None:
        self.children: Dict[str, _TrieNode] = {}
        self.weight = 0.0
        self.best = 0.0  # peso máximo en el subárbol
        self.term: Optional[str] = None


class Trie:
    def __init__(self) -> None:
        self.root = _TrieNode()
        self.size = 0

    def add(self, term: str, weight: float = 1.0) -> None:
        """Inserta el término o suma `weight` a su peso."""
        term = term.lower().strip()
        if not term:
            return
        path = [self.root]
        node = self.root
        for ch in term:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = _TrieNode()
                node.children[ch] = nxt
            node = nxt
            path.append(node)
        if node.term is None:
            node.term = term
            self.size += 1
        node.weight += weight
        w = node.weight
        for n in path:  # los pesos solo crecen: basta con actualizar el máximo
            if n.best < w:
                n.best = w

    def _find(self, prefix: str) -> Optional[_TrieNode]:
        node = self.root
        for ch in prefix.lower():
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def __contains__(self, term: str) -> bool:
        node = self._find(term)
        return node is not None and node.term is not None

    def complete(self, prefix: str, k: int = 5) -> List[Tuple[str, float]]:
        start = self._find(prefix)
        if start is None:
            return []
        out: List[Tuple[str, float]] = []
        heap = BinaryHeap()
        heap.push((-start.best, 1), start)
        while len(heap) and len(out) < k:
            (neg, kind), node = heap.pop()
            if kind == 0:  # término completo: es el siguiente mejor
                out.append((node.term, node.weight))
                continue
            if node.term is not None:
                heap.push((-node.weight, 0), node)
            for child in node.children.values():
                heap.push((-child.best, 1), child)
        return out
