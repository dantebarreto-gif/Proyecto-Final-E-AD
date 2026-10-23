"""Orquestador: conecta las 10 etapas y produce las salidas observables.

Orden de ejecución por lote:
    S0 -> S1 -> S2 -> S3 -> S6 -> S4 -> S5 -> S7 -> S8 -> S9

S6 (filtrado) se ejecuta justo después de resolver ids en S3 para que los
eventos bloqueados no lleguen a los índices, a la similitud ni al grafo: lo
que pasa el filtro actualiza la matriz de interacciones (S3), el índice de
órdenes (S4), y en cada refresco el KD-Tree (S5), LSH/IVF (S7) y el grafo
(S8). S9 consume las salidas de todas las etapas.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..data.generator import blacklist
from .events import Event, iter_events
from .stages import (AttributeIndex, DiskIndex, FraudFilter, GraphStage, Ingestion, Prioritizer,
                     Preprocessor, Service, Similarity, StageStats, Store)


class StreamPipeline:
    def __init__(self, events_path: str, catalog_path: str, workdir: str = "data/work",
                 batch_size: int = 5_000, refresh_every: int = 100_000, known_fraud_share: float = 0.5,
                 trace_event: Optional[int] = None, seed: int = 0, limit: Optional[int] = None) -> None:
        self.events_path = events_path
        self.batch_size = batch_size
        self.refresh_every = refresh_every
        self.stats = StageStats()
        t = time.perf_counter()
        catalog = pd.read_parquet(catalog_path)
        labels = pd.read_parquet(events_path, columns=["event_id", "event_type", "is_fraud", "is_duplicate",
                                                       "card_id", "device_id"])
        # lista negra "conocida": se asume que una parte de los fraudes se reportó antes
        self.blacklist = blacklist(labels, known_fraud_share, seed)
        window = labels if limit is None else labels.iloc[:limit]
        self.trace_event = trace_event if trace_event is not None else self._pick_trace(window)
        self.store = Store(catalog)
        self.s0 = Ingestion(self._source(), capacity=4 * batch_size)
        self.s1 = Preprocessor()
        self.s2 = Prioritizer()
        self.s4 = DiskIndex(self.store, workdir)
        self.s5 = AttributeIndex(self.store)
        self.s6 = FraudFilter(self.blacklist)
        self.s7 = Similarity(self.store)
        self.s8 = GraphStage(self.store)
        self.s9 = Service(self.store)
        self.stats.add("setup", time.perf_counter() - t)
        self.processed = 0
        self.passed = 0
        self.refreshes = 0
        self.trace: List[str] = []
        self.traced: Optional[Event] = None

    @staticmethod
    def _pick_trace(labels: pd.DataFrame) -> int:
        """Una compra legítima ubicada al 70 % del stream (índices ya construidos)."""
        c = labels[(labels.event_type == "purchase") & ~labels.is_fraud & ~labels.is_duplicate]
        return int(c.event_id.iloc[int(len(c) * 0.7)]) if len(c) else -1

    def _source(self):
        for e in iter_events(self.events_path):
            if e.event_id == self.trace_event and self.traced is None and not e.is_duplicate:
                e.trace = []
                self.traced = e
            yield e

    def _timed(self, name: str, fn, batch):
        t = time.perf_counter()
        out = fn(batch)
        self.stats.add(name, time.perf_counter() - t, len(out) if isinstance(out, list) else 0)
        return out

    def refresh(self) -> None:
        for name, fn in (("S5 refresco KD-Tree", self.s5.refresh), ("S7 refresco LSH+IVF", self.s7.refresh),
                         ("S8 refresco PageRank+Louvain", self.s8.refresh)):
            t = time.perf_counter()
            fn()
            self.stats.add(name, time.perf_counter() - t)
        self.store.cache = type(self.store.cache)(self.store.cache.capacity)  # invalida resultados viejos
        self.refreshes += 1

    def run(self, limit: Optional[int] = None, verbose: bool = True) -> dict:
        next_refresh = self.refresh_every
        t0 = time.perf_counter()
        while True:
            batch = self._timed("S0 ingesta", self.s0.next_batch, self.batch_size)
            if not batch:
                break
            batch = self._timed("S1 preprocesamiento", self.s1.process, batch)
            batch = self._timed("S2 prioridad", self.s2.process, batch)
            batch = self._timed("S3 almacenamiento", self.s3_process, batch)
            batch = self._timed("S6 filtrado", self.s6.process, batch)
            self._timed("S3 interacciones", self.store.record_interactions, batch)
            batch = self._timed("S4 B+Tree", self.s4.process, batch)
            batch = self._timed("S5 KD-Tree", self.s5.process, batch)
            batch = self._timed("S7 similitud", self.s7.process, batch)
            batch = self._timed("S8 grafo", self.s8.process, batch)
            batch = self._timed("S9 servicio", self.s9.process, batch)
            self.passed += len(batch)
            self.processed = self.s0.received
            if self.processed >= next_refresh:
                self.refresh()
                next_refresh += self.refresh_every
                if verbose:
                    rate = self.processed / (time.perf_counter() - t0)
                    print(f"  {self.processed:>10,} eventos | {rate:,.0f} ev/s | refresco #{self.refreshes}",
                          flush=True)
            if limit is not None and self.processed >= limit:
                break
        self.refresh()
        self.elapsed = time.perf_counter() - t0
        return self.summary()

    def s3_process(self, batch: List[Event]) -> List[Event]:
        return self.store.process(batch)

    # -------------------------------------------------------------- salidas
    def summary(self) -> dict:
        st = self.store
        info = st.product_info
        out: Dict = {
            "eventos_recibidos": self.processed,
            "eventos_que_pasan_el_filtro": self.passed,
            "segundos": round(self.elapsed, 2),
            "eventos_por_segundo": round(self.processed / max(self.elapsed, 1e-9)),
            "S1": {"invalidos": self.s1.invalid, "duplicados_eliminados": self.s1.duplicates,
                   "duplicados_reales_detectados": self.s1.dup_true_positive},
            "S2": dict(zip(self.s2.CLASSES, self.s2.by_class)),
            "S3": {"productos": len(st.products), "usuarios": len(st.users),
                   "interacciones_nnz": st.interactions.nnz, "productos_nuevos_en_stream": st.new_products},
            "S4": {"altura_indice_precio": self.s4.price.height, "paginas_precio": self.s4.price.n_pages,
                   "ordenes_indexadas": len(self.s4.orders), "altura_indice_ordenes": self.s4.orders.height,
                   "lecturas_disco": self.s4.price.reads + self.s4.orders.reads,
                   "escrituras_disco": self.s4.price.writes + self.s4.orders.writes},
            "S6": self.s6.metrics(),
            "S8": self.s8.stats,
            "tiempos_por_etapa_s": {k: round(v, 3) for k, v in self.stats.time.items()},
        }
        out["top_ventas"] = [{**info(i), "ventas": int(v)} for i, v in self.s9.top_products(10)]
        if self.s8.rank is not None:
            pr = np.argsort(-self.s8.rank)[:10]
            out["top_pagerank"] = [{**info(int(i)), "pagerank": round(float(self.s8.rank[i]), 6)} for i in pr]
        alerts = []
        h = self.s6.alerts
        snapshot = sorted(h.items(), key=lambda t: t[0])[:10]
        for (sev, _), a in snapshot:
            alerts.append({"severidad": sev, **a})
        out["alertas_prioritarias"] = alerts
        out["autocompletado"] = {p: self.s9.autocomplete(p, 5) for p in ("sam", "app", "lg ", "nik", "tv")}
        out["rango_precio_100_120"] = [info(i)["name"] for i in self.s4.products_in_price_range(100, 120, 5)]
        if self.traced is not None:
            e = self.traced
            recs = self.s9.recommend(e.product_idx, self.s7, self.s8, self.s5, k=5)
            out["traza_evento"] = {"evento": repr(e), "pasos": e.trace,
                                   "producto": info(e.product_idx),
                                   "recomendaciones": [{**info(j), "puntaje": s, "fuentes": w}
                                                       for j, s, w in recs],
                                   "recomendaciones_usuario": [info(j)["name"] for j in
                                                               self.s7.recommend_for_user(
                                                                   e.user_idx, st.interaction_matrix(), 5)]}
        return out

    def close(self) -> None:
        self.s4.close()


def save_summary(summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
