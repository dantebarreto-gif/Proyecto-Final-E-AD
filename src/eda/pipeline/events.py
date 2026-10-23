"""Representación de un evento del stream y lectura por lotes desde Parquet."""
from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np
import pyarrow.parquet as pq

FIELDS = ("event_id", "ts", "event_type", "product_id", "category_code", "brand", "price", "user_id",
          "user_session", "device_id", "ip", "card_id", "premium", "shipping", "search_query",
          "is_fraud", "is_duplicate")


class Event:
    """Evento con __slots__ (menos memoria que un dict) y campos que agregan las etapas."""

    __slots__ = FIELDS + ("priority", "product_idx", "user_idx", "flags", "trace")

    def __init__(self, **kw) -> None:
        for f in FIELDS:
            setattr(self, f, kw.get(f))
        self.priority = None
        self.product_idx = -1
        self.user_idx = -1
        self.flags: List[str] = []
        self.trace: Optional[List[str]] = None

    def note(self, stage: str, msg: str) -> None:
        if self.trace is not None:
            self.trace.append(f"[{stage}] {msg}")

    def __repr__(self) -> str:
        return (f"Event(id={self.event_id}, {self.event_type}, product={self.product_id}, "
                f"user={self.user_id}, price={self.price})")


def iter_events(path: str, batch_rows: int = 50_000, limit: Optional[int] = None) -> Iterator[Event]:
    """Lee el Parquet por row groups (sin cargarlo completo) y emite eventos en orden."""
    pf = pq.ParquetFile(path)
    emitted = 0
    for rb in pf.iter_batches(batch_size=batch_rows, columns=list(FIELDS)):
        cols = {name: rb.column(name).to_numpy(zero_copy_only=False) for name in FIELDS}
        n = rb.num_rows
        for i in range(n):
            if limit is not None and emitted >= limit:
                return
            yield Event(**{f: _py(cols[f][i]) for f in FIELDS})
            emitted += 1


def _py(v):
    if isinstance(v, np.generic):
        return v.item()
    return v
