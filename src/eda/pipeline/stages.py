"""Las 10 etapas del pipeline. Cada una usa su estructura y alimenta a la siguiente.

    S0 Ingesta          LinkedQueue (FIFO)         -> lotes de eventos crudos
    S1 Preprocesamiento merge sort + binaria       -> eventos válidos, ordenados y sin duplicados
    S2 Prioridad        BinaryHeap                 -> eventos en orden de urgencia
    S3 Almacenamiento   HashTable, LRU, DOK/CSR    -> catálogo, usuarios, matriz de interacciones
    S4 Índice en disco  B+Tree                     -> consultas por rango de precio y de tiempo
    S5 Multidimensional KD-Tree + KNN exacto       -> vecinos por atributos y consultas por caja
    S6 Filtrado [FIJA]  Bloom Filter + CMS         -> alertas de fraude; solo pasa lo legítimo
    S7 Similitud [FIJA] MinHash+LSH, IVF (HNSW)    -> productos relacionados y recomendaciones
    S8 Grafo [FIJA]     PageRank + Louvain         -> productos influyentes y comunidades
    S9 Servicio         Trie, Top-K, panel         -> autocompletado, rankings, alertas, métricas
"""
from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..structures import (BinaryHeap, BloomFilter, BPlusTree, CoPurchaseGraph, CSRMatrix, DOKAccumulator,
                          HashSet, HashTable, IVFIndex, KDTree, LinkedQueue, LRUCache, MinHashLSH, TopK, Trie,
                          WindowedCountMin, community_sizes, contains_sorted, degree_ranking,
                          knn_brute_force, label_propagation, louvain, merge_sort, merge_sorted_arrays,
                          modularity, pagerank, price_key, price_range_keys, randomized_svd)
from .events import Event

VALID_TYPES = {"view", "cart", "remove_from_cart", "purchase"}
INTERACTION_WEIGHT = {"view": 1.0, "cart": 3.0, "remove_from_cart": -1.0, "purchase": 5.0}


class StageStats:
    def __init__(self) -> None:
        self.time = defaultdict(float)
        self.count = defaultdict(int)

    def add(self, stage: str, seconds: float, n: int = 1) -> None:
        self.time[stage] += seconds
        self.count[stage] += n


# --------------------------------------------------------------------- S0
class Ingestion:
    """S0: buffer FIFO entre el productor (lector del stream) y el consumidor por lotes."""

    def __init__(self, source: Iterable[Event], capacity: int = 200_000) -> None:
        self.source = iter(source)
        self.queue = LinkedQueue(capacity)
        self.exhausted = False
        self.received = 0

    def fill(self, n: int) -> None:
        while len(self.queue) < n and not self.exhausted:
            try:
                ev = next(self.source)
            except StopIteration:
                self.exhausted = True
                break
            if self.queue.enqueue(ev):
                self.received += 1
                ev.note("S0", f"encolado (cola={len(self.queue)})")

    def next_batch(self, size: int) -> List[Event]:
        self.fill(size)
        return self.queue.dequeue_batch(size)


# --------------------------------------------------------------------- S1
class Preprocessor:
    """S1: valida, normaliza, ordena por tiempo (merge sort) y elimina duplicados.

    Deduplicación en dos niveles: dentro del lote (ids adyacentes tras ordenar)
    y contra una ventana de ids ya vistos (arreglo ordenado + búsqueda binaria).
    """

    def __init__(self, window_ids: int = 500_000) -> None:
        self.seen = np.empty(0, dtype=np.int64)
        self.window_ids = window_ids
        self.invalid = 0
        self.duplicates = 0
        self.dup_true_positive = 0

    def _valid(self, e: Event) -> bool:
        return (e.event_type in VALID_TYPES and e.product_id is not None and e.user_id is not None
                and e.price is not None and e.price > 0 and e.ts is not None)

    def process(self, batch: List[Event]) -> List[Event]:
        clean = []
        for e in batch:
            if not self._valid(e):
                self.invalid += 1
                continue
            e.brand = (e.brand or "generic").strip().lower()
            e.category_code = (e.category_code or "unknown").strip().lower()
            e.search_query = (e.search_query or "").strip().lower()
            e.price = round(float(e.price), 2)
            clean.append(e)
        by_id = merge_sort(clean, key=lambda e: e.event_id)
        unique: List[Event] = []
        last = None
        for e in by_id:  # duplicados dentro del lote: adyacentes tras ordenar
            if e.event_id == last:
                self._drop(e)
                continue
            last = e.event_id
            unique.append(e)
        ids = np.fromiter((e.event_id for e in unique), dtype=np.int64, count=len(unique))
        seen_mask = contains_sorted(self.seen, ids)  # duplicados contra la ventana
        out = []
        for e, dup in zip(unique, seen_mask.tolist()):
            if dup:
                self._drop(e)
            else:
                out.append(e)
        self.seen = merge_sorted_arrays(self.seen, ids[~seen_mask])
        if self.seen.size > self.window_ids:
            self.seen = self.seen[-self.window_ids:]
        out = merge_sort(out, key=lambda e: e.ts)
        for e in out:
            e.note("S1", "válido, normalizado y no duplicado")
        return out

    def _drop(self, e: Event) -> None:
        self.duplicates += 1
        if e.is_duplicate:
            self.dup_true_positive += 1


# --------------------------------------------------------------------- S2
class Prioritizer:
    """S2: heap de prioridades. Menor número = más urgente; empate -> FIFO."""

    CLASSES = ["compra express", "compra premium", "compra estándar", "compra programada",
               "carrito", "vista"]

    def __init__(self) -> None:
        self.heap = BinaryHeap()
        self.by_class = [0] * len(self.CLASSES)

    @staticmethod
    def priority_of(e: Event) -> int:
        if e.event_type == "purchase":
            if e.shipping == "express":
                return 0
            if e.premium:
                return 1
            return 3 if e.shipping == "scheduled" else 2
        if e.event_type in ("cart", "remove_from_cart"):
            return 4
        return 5

    def process(self, batch: List[Event]) -> List[Event]:
        for e in batch:
            p = self.priority_of(e)
            e.priority = p
            self.by_class[p] += 1
            self.heap.push((p, e.ts), e)
        out = []
        while len(self.heap):
            _, e = self.heap.pop()
            e.note("S2", f"prioridad {e.priority} ({self.CLASSES[e.priority]}), "
                         f"posición {len(out) + 1} de {len(batch)} en el lote")
            out.append(e)
        return out


# --------------------------------------------------------------------- S3
class Store:
    """S3: catálogo y usuarios en tablas hash, caché LRU y matriz usuario x producto."""

    def __init__(self, catalog: pd.DataFrame, cache_size: int = 2_000, max_users: int = 2_000_000) -> None:
        self.catalog = catalog.reset_index(drop=True)
        self.products = HashTable(len(catalog) * 2)  # product_id -> índice denso
        for i, pid in enumerate(self.catalog["product_id"].tolist()):
            self.products.put(int(pid), i)
        n = len(self.catalog)
        self.views = np.zeros(n)
        self.carts = np.zeros(n)
        self.purchases = np.zeros(n)
        self.users = HashTable(1024)  # user_id -> índice denso
        self.user_ids: List[int] = []
        self.max_users = max_users
        self.cache = LRUCache(cache_size)
        self.interactions = DOKAccumulator(max_users, n)
        self.new_products = 0

    def product_index(self, e: Event) -> int:
        idx = self.products.get(e.product_id)
        if idx is None:  # producto nuevo en el stream: se agrega al catálogo
            idx = len(self.catalog)
            self.products.put(e.product_id, idx)
            self.catalog.loc[idx] = {"product_id": e.product_id, "category_code": e.category_code,
                                     "brand": e.brand, "price": e.price, "name": f"{e.brand} {e.product_id}",
                                     "popularity": 0.0}
            for name in ("views", "carts", "purchases"):
                setattr(self, name, np.append(getattr(self, name), 0.0))
            self.interactions.ncols += 1
            self.new_products += 1
        return idx

    def user_index(self, user_id: int) -> int:
        idx = self.users.get(user_id)
        if idx is None:
            idx = len(self.user_ids)
            self.users.put(user_id, idx)
            self.user_ids.append(user_id)
        return idx

    def process(self, batch: List[Event]) -> List[Event]:
        for e in batch:
            e.product_idx = self.product_index(e)
            e.user_idx = self.user_index(e.user_id)
            e.note("S3", f"producto #{e.product_idx} y usuario #{e.user_idx} resueltos en la tabla hash")
        return batch

    def record_interactions(self, batch: List[Event]) -> None:
        """Se llama con lo que pasa el filtro S6: solo interacciones legítimas."""
        for e in batch:
            self.interactions.add(e.user_idx, e.product_idx, INTERACTION_WEIGHT[e.event_type])
            if e.event_type == "view":
                self.views[e.product_idx] += 1
            elif e.event_type == "cart":
                self.carts[e.product_idx] += 1
            elif e.event_type == "purchase":
                self.purchases[e.product_idx] += 1

    def product_info(self, idx: int) -> dict:
        """Consulta de detalle con caché LRU (ficha de producto)."""
        key = ("info", idx)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        row = self.catalog.iloc[idx]
        info = {"idx": idx, "product_id": int(row["product_id"]), "name": row["name"],
                "brand": row["brand"], "category": row["category_code"], "price": float(row["price"]),
                "views": int(self.views[idx]), "purchases": int(self.purchases[idx])}
        self.cache.put(key, info)
        return info

    def interaction_matrix(self) -> CSRMatrix:
        A = self.interactions.to_csr()
        nu = len(self.user_ids)
        return CSRMatrix(A.indptr[: nu + 1], A.indices, np.maximum(A.data, 0), (nu, A.shape[1]))


# --------------------------------------------------------------------- S4
class DiskIndex:
    """S4: B+Tree en disco. Índice de precio (catálogo) e índice temporal de órdenes."""

    def __init__(self, store: Store, workdir: str, page_size: int = 4096) -> None:
        os.makedirs(workdir, exist_ok=True)
        self.store = store
        self.price = BPlusTree(os.path.join(workdir, "price_index.bpt"), page_size=page_size)
        prices = store.catalog["price"].to_numpy()
        keys = np.array([price_key(p, i) for i, p in enumerate(prices)], dtype=np.int64)
        order = np.argsort(keys, kind="stable")
        self.price.bulk_load(keys[order], order)  # carga masiva ordenada: O(n)
        self.orders = BPlusTree(os.path.join(workdir, "orders_index.bpt"), page_size=page_size)
        self.indexed_products = len(prices)

    def process(self, batch: List[Event]) -> List[Event]:
        for e in batch:
            if e.product_idx >= self.indexed_products:  # producto nuevo -> inserción
                self.price.insert(price_key(e.price, e.product_idx), e.product_idx)
                self.indexed_products = e.product_idx + 1
            if e.event_type == "purchase":
                # clave temporal única: segundos * 2^20 + id de evento (mod 2^20)
                self.orders.insert((int(e.ts) << 20) | (e.event_id & 0xFFFFF), e.event_id)
                e.note("S4", "orden indexada por tiempo en el B+Tree")
        return batch

    def products_in_price_range(self, lo: float, hi: float, limit: int = 20) -> List[int]:
        a, b = price_range_keys(lo, hi)
        return [v for _, v in self.price.range_search(a, b, limit=limit)]

    def orders_between(self, t0: float, t1: float) -> List[int]:
        return [v for _, v in self.orders.range_search(int(t0) << 20, (int(t1) << 20) | 0xFFFFF)]

    def close(self) -> None:
        self.price.close()
        self.orders.close()


# --------------------------------------------------------------------- S5
class AttributeIndex:
    """S5: KD-Tree sobre (log precio, log vistas, tasa de conversión)."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.tree: Optional[KDTree] = None
        self.features: Optional[np.ndarray] = None

    def refresh(self) -> None:
        s = self.store
        price = np.log1p(s.catalog["price"].to_numpy(dtype=float))
        views = np.log1p(s.views)
        conv = (s.purchases + 1) / (s.views + 10)
        f = np.column_stack([price, views, conv])
        self.features = (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-9)  # misma escala por eje
        self.tree = KDTree(self.features, leaf_size=16)

    def similar(self, idx: int, k: int = 5) -> List[int]:
        ids, _ = self.tree.query(self.features[idx], k + 1)
        return [int(i) for i in ids if i != idx][:k]

    def similar_exact(self, idx: int, k: int = 5) -> List[int]:
        ids, _ = knn_brute_force(self.features, self.features[idx], k + 1)
        return [int(i) for i in ids if i != idx][:k]

    def process(self, batch: List[Event]) -> List[Event]:
        if self.tree is not None:
            for e in batch:
                if e.trace is not None:
                    e.note("S5", f"vecinos por atributos (KD-Tree): {self.similar(e.product_idx, 3)}")
        return batch


# --------------------------------------------------------------------- S6
class FraudFilter:
    """S6: Bloom Filter de la lista negra + Count-Min Sketch por ventana temporal.

    - Bloom: tarjeta o dispositivo "posiblemente" en la lista negra. Como puede
      haber falsos positivos, se verifica contra el conjunto exacto antes de
      bloquear (la decisión es crítica).
    - CMS con ventana de 10 min (10 intervalos de 1 min): intentos de compra o
      carrito por tarjeta y por dispositivo. Alerta si supera el umbral.
    """

    def __init__(self, blacklist: Dict[str, np.ndarray], window_seconds: int = 600,
                 card_threshold: int = 12, device_threshold: int = 20) -> None:
        keys = np.concatenate([blacklist.get("card_id", np.empty(0, np.int64)),
                               blacklist.get("device_id", np.empty(0, np.int64))]).astype(np.int64)
        self.bloom = BloomFilter(max(1000, keys.size * 2), fp_rate=0.001)
        self.bloom.add_many(keys)
        self.exact = HashSet(max(16, keys.size * 2))
        for k in keys.tolist():
            self.exact.add(k)
        self.card_cms = WindowedCountMin(window_seconds, 10, eps=0.0005, delta=0.01)
        self.device_cms = WindowedCountMin(window_seconds, 10, eps=0.0005, delta=0.01)
        self.card_threshold = card_threshold
        self.device_threshold = device_threshold
        self.alerts = BinaryHeap()  # severidad: 0 lista negra, 1 frecuencia de tarjeta, 2 de dispositivo
        self.flagged_cards = HashSet()
        self.bloom_positives = 0
        self.bloom_false_positives = 0
        self.blocked = 0
        self.tp = self.fp = self.fn = 0

    def process(self, batch: List[Event]) -> List[Event]:
        passed = []
        for e in batch:
            blocked = False
            if e.event_type in ("purchase", "cart"):
                for key in (e.card_id, e.device_id):
                    if key in self.bloom:
                        self.bloom_positives += 1
                        if key in self.exact:  # verificación exacta antes de bloquear
                            self._alert(0, e, "tarjeta/dispositivo en lista negra")
                            blocked = True
                            break
                        self.bloom_false_positives += 1
                if not blocked:
                    self.card_cms.update(e.card_id, e.ts)
                    self.device_cms.update(e.device_id, e.ts)
                    c = self.card_cms.estimate(e.card_id, e.ts)
                    d = self.device_cms.estimate(e.device_id, e.ts)
                    if c > self.card_threshold or e.card_id in self.flagged_cards:
                        self.flagged_cards.add(e.card_id)
                        self._alert(1, e, f"{c} intentos con la tarjeta en 10 min")
                        blocked = True
                    elif d > self.device_threshold:
                        self._alert(2, e, f"{d} intentos desde el dispositivo en 10 min")
                        blocked = True
            if blocked:
                self.blocked += 1
                e.flags.append("fraude")
                self.tp += bool(e.is_fraud)
                self.fp += not e.is_fraud
            else:
                self.fn += bool(e.is_fraud)
                e.note("S6", "pasa el filtro (Bloom: ausente; CMS bajo el umbral)")
                passed.append(e)
        return passed

    def _alert(self, severity: int, e: Event, reason: str) -> None:
        self.alerts.push((severity, e.ts), {"event_id": e.event_id, "card_id": e.card_id,
                                            "device_id": e.device_id, "reason": reason, "ts": e.ts})
        e.note("S6", f"BLOQUEADO: {reason}")

    def metrics(self) -> dict:
        prec = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        rec = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        return {"blocked": self.blocked, "precision": prec, "recall": rec,
                "bloom_positives": self.bloom_positives, "bloom_false_positives": self.bloom_false_positives,
                "bloom_memory_bytes": self.bloom.memory_bytes(),
                "cms_memory_bytes": self.card_cms.memory_bytes() + self.device_cms.memory_bytes()}


# --------------------------------------------------------------------- S7
class Similarity:
    """S7: MinHash+LSH sobre compradores de cada producto e IVF sobre embeddings SVD."""

    def __init__(self, store: Store, dim: int = 32) -> None:
        self.store = store
        self.dim = dim
        self.lsh: Optional[MinHashLSH] = None
        self.ivf: Optional[IVFIndex] = None
        self.emb: Optional[np.ndarray] = None
        self.sets_indptr = self.sets_indices = None

    def refresh(self) -> None:
        A = self.store.interaction_matrix()  # usuarios x productos
        if A.nnz == 0:
            return
        # conjuntos de MinHash: usuarios con carrito o compra (peso >= 3) por producto
        strong = A.data >= 3
        At = CSRMatrix.from_coo(A.indices[strong], A.row_ids()[strong], np.ones(strong.sum()),
                                (A.shape[1], A.shape[0]))
        self.sets_indptr, self.sets_indices = At.indptr, At.indices
        self.lsh = MinHashLSH(num_perm=128, bands=32)
        self.lsh.index(self.lsh.signatures_csr(At.indptr, At.indices))
        # embeddings: SVD truncada de la matriz de interacciones (log escala)
        Al = CSRMatrix(A.indptr, A.indices, np.log1p(A.data), A.shape)
        k = min(self.dim, min(A.shape) - 1)
        _, s, Vt = randomized_svd(Al, k)
        self.emb = (Vt.T * np.sqrt(s)).astype(np.float32)
        active = np.where(np.abs(self.emb).sum(axis=1) > 0)[0]
        self.active = active
        self.ivf = IVFIndex(nprobe=8).build(self.emb[active])

    def related(self, idx: int, k: int = 5) -> List[Tuple[int, float]]:
        if self.lsh is None:
            return []
        return self.lsh.query(self.lsh.signatures[idx], k=k, exclude=idx)

    def neighbors(self, vec: np.ndarray, k: int = 10, exclude: Iterable[int] = ()) -> List[int]:
        if self.ivf is None or not np.any(vec):
            return []
        q = vec / (np.linalg.norm(vec) + 1e-12)
        ex = set(exclude)
        res = [int(self.active[i]) for i in self.ivf.query(q, k + len(ex))]
        return [i for i in res if i not in ex][:k]

    def recommend_for_user(self, user_idx: int, A: CSRMatrix, k: int = 10) -> List[int]:
        items, w = A.row(user_idx)
        if items.size == 0 or self.emb is None:
            return []
        vec = (self.emb[items] * w[:, None]).sum(axis=0)
        return self.neighbors(vec, k, exclude=items.tolist())

    def process(self, batch: List[Event]) -> List[Event]:
        if self.lsh is not None:
            for e in batch:
                if e.trace is not None:
                    e.note("S7", f"relacionados LSH: {[i for i, _ in self.related(e.product_idx, 3)]}; "
                                 f"vecinos IVF: {self.neighbors(self.emb[e.product_idx], 3, [e.product_idx])}")
        return batch


# --------------------------------------------------------------------- S8
class GraphStage:
    """S8: grafo de co-compra incremental; PageRank y Louvain en cada refresco."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.graph = CoPurchaseGraph(len(store.catalog))
        self.session_items = HashTable(1024)  # sesión -> productos comprados (orden en curso)
        self.rank: Optional[np.ndarray] = None
        self.communities: Optional[np.ndarray] = None
        self.adj: Optional[CSRMatrix] = None
        self.stats: dict = {}

    def process(self, batch: List[Event]) -> List[Event]:
        for e in batch:
            if e.event_type != "purchase":
                continue
            items = self.session_items.get(e.user_session)
            if items is None:
                items = []
                self.session_items.put(e.user_session, items)
            if e.product_idx not in items:
                for other in items[-20:]:  # aristas con lo ya comprado en la misma orden
                    a, b = min(other, e.product_idx), max(other, e.product_idx)
                    key = a * self.graph.n + b
                    self.graph._edges[key] = self.graph._edges.get(key, 0.0) + 1.0
                prev = items[-20:]
                items.append(e.product_idx)
                e.note("S8", f"co-compra con {prev[-5:]} en la misma orden" if prev
                       else "primera compra de la orden: nodo listo para nuevas aristas")
        return batch

    def refresh(self) -> None:
        self.graph.n = max(self.graph.n, len(self.store.catalog))
        A = self.graph.to_csr()
        self.adj = A
        if A.nnz == 0:
            return
        t = time.perf_counter()
        self.rank, iters = pagerank(A)
        t_pr = time.perf_counter() - t
        # comunidades sobre los nodos con aristas (los aislados no aportan)
        active = np.where(np.diff(A.indptr) > 0)[0]
        remap = -np.ones(A.shape[0], dtype=np.int64)
        remap[active] = np.arange(active.size)
        rows = A.row_ids()
        sub = CSRMatrix.from_coo(remap[rows], remap[A.indices], A.data, (active.size, active.size))
        t = time.perf_counter()
        lv = louvain(sub, seed=0)
        t_lv = time.perf_counter() - t
        t = time.perf_counter()
        lp = label_propagation(sub, seed=0)
        t_lp = time.perf_counter() - t
        comm = -np.ones(A.shape[0], dtype=np.int64)
        comm[active] = lv
        self.communities = comm
        deg = degree_ranking(A)
        top_pr = set(np.argsort(-self.rank)[:20].tolist())
        top_deg = set(np.argsort(-deg)[:20].tolist())
        self.stats = {"nodes_with_edges": int(active.size), "edges": int(A.nnz // 2),
                      "pagerank_iters": iters, "pagerank_s": t_pr,
                      "louvain_s": t_lv, "louvain_Q": modularity(sub, lv), "louvain_k": int(lv.max() + 1),
                      "lpa_s": t_lp, "lpa_Q": modularity(sub, lp), "lpa_k": int(lp.max() + 1),
                      "top20_overlap_pagerank_vs_degree": len(top_pr & top_deg) / 20,
                      "largest_communities": community_sizes(lv)[:5].tolist()}

    def neighbors(self, idx: int, k: int = 5) -> List[int]:
        if self.adj is None or idx >= self.adj.shape[0]:
            return []
        cols, w = self.adj.row(idx)
        order = np.argsort(-w, kind="stable")[:k]
        return cols[order].tolist()


# --------------------------------------------------------------------- S9
class Service:
    """S9: autocompletado (Trie), Top-K de ventas, recomendaciones y panel."""

    def __init__(self, store: Store, k: int = 10) -> None:
        self.store = store
        self.trie = Trie()
        for name, pop in zip(store.catalog["name"].tolist(), store.catalog["popularity"].tolist()):
            self.trie.add(name, float(pop))
        self.k = k

    def process(self, batch: List[Event]) -> List[Event]:
        for e in batch:
            if e.search_query:
                self.trie.add(e.search_query, 1.0)
                e.note("S9", f"consulta '{e.search_query}' suma peso en el Trie")
        return batch

    def top_products(self, k: int = 10) -> List[Tuple[int, float]]:
        t = TopK(k)
        for i, v in enumerate(self.store.purchases.tolist()):
            if v > 0:
                t.add(v, i)
        return t.result()

    def autocomplete(self, prefix: str, k: int = 5) -> List[Tuple[str, float]]:
        return self.trie.complete(prefix, k)

    def recommend(self, idx: int, sim: Similarity, graph: GraphStage, attrs: AttributeIndex,
                  k: int = 10) -> List[Tuple[int, float, str]]:
        """Combina las salidas de S5, S7 y S8 en un solo ranking Top-K.

        puntaje = Jaccard estimado (LSH) + similitud de embeddings (IVF)
                  + co-compra (grafo) + 0.5 · PageRank normalizado
                  + 0.1 si comparte comunidad.
        """
        score: Dict[int, float] = defaultdict(float)
        why: Dict[int, List[str]] = defaultdict(list)
        cache_key = ("rec", idx)
        cached = self.store.cache.get(cache_key)
        if cached is not None:
            return cached
        for j, s in sim.related(idx, k):
            score[j] += s
            why[j].append("LSH")
        if sim.emb is not None:
            for r, j in enumerate(sim.neighbors(sim.emb[idx], k, [idx])):
                score[j] += 1.0 - r / (2 * k)
                why[j].append("IVF")
        for r, j in enumerate(graph.neighbors(idx, k)):
            score[j] += 1.0 - r / (2 * k)
            why[j].append("grafo")
        if attrs.tree is not None:
            for j in attrs.similar(idx, 3):
                score[j] += 0.2
                why[j].append("KD-Tree")
        if graph.rank is not None:
            pr_max = graph.rank.max()
            for j in list(score):
                if j < graph.rank.size:
                    score[j] += 0.5 * graph.rank[j] / pr_max
                if graph.communities is not None and j < graph.communities.size and \
                        graph.communities[j] >= 0 and graph.communities[j] == graph.communities[idx]:
                    score[j] += 0.1
        top = TopK(k)
        for j, s in score.items():
            top.add(s, j)
        res = [(j, round(s, 3), "+".join(why[j])) for j, s in top.result()]
        self.store.cache.put(cache_key, res)
        return res
