"""Experimento E1: escalabilidad de las estructuras base (etapas 0 a 5)."""
from __future__ import annotations

import os
import random
import tempfile

import numpy as np

from common import Recorder, SEED, memory, reps_for, timeit
from eda.structures import (BinaryHeap, BPlusTree, HashTable, KDTree, LinkedQueue, LRUCache, knn_brute_force,
                            merge_sort)


def run(sizes) -> None:
    rec = Recorder("base_structures")
    rng = np.random.default_rng(SEED)
    for n in sizes:
        reps = reps_for(n)
        keys = rng.integers(0, 2 ** 62, n).tolist()
        print(f"[E1] n={n:,} reps={reps}")

        # S0 cola FIFO enlazada: n enqueue + n dequeue
        def fifo():
            q = LinkedQueue()
            for k in keys:
                q.enqueue(k)
            while len(q):
                q.dequeue()
        m, s = timeit(fifo, reps)
        cur, peak, _ = memory(lambda: _fill_queue(keys))
        rec.add(structure="LinkedQueue", op="enqueue+dequeue", n=n, reps=reps, mean_s=m, std_s=s,
                per_op_us=m / (2 * n) * 1e6, mem_current_B=cur, mem_peak_B=peak, expected="O(1)")

        # S1 merge sort (Python puro) sobre n claves
        r = max(1, reps // 10) if n >= 100_000 else reps
        m, s = timeit(lambda: merge_sort(keys), r)
        rec.add(structure="MergeSort", op="sort", n=n, reps=r, mean_s=m, std_s=s, per_op_us=m / n * 1e6,
                expected="O(n log n)")

        # S2 heap: n push + n pop
        def heap():
            h = BinaryHeap()
            for k in keys:
                h.push(k, None)
            while len(h):
                h.pop()
        m, s = timeit(heap, reps)
        rec.add(structure="BinaryHeap", op="push+pop", n=n, reps=reps, mean_s=m, std_s=s,
                per_op_us=m / (2 * n) * 1e6, expected="O(log n)")

        # S3 tabla hash: n put + n get
        def hashtable():
            t = HashTable()
            for k in keys:
                t.put(k, 1)
            for k in keys:
                t.get(k)
        m, s = timeit(hashtable, reps)
        cur, peak, t = memory(lambda: _fill_hash(keys))
        rec.add(structure="HashTable", op="put+get", n=n, reps=reps, mean_s=m, std_s=s,
                per_op_us=m / (2 * n) * 1e6, mem_current_B=cur, mem_peak_B=peak, expected="O(1)",
                extra=f"probes_per_op={t.probes / n:.3f}")
        del t

        # S3 LRU: n operaciones get/put con claves Zipf, capacidad n/10
        zipf = (rng.zipf(1.2, n) % max(10, n // 2)).tolist()

        def lru():
            c = LRUCache(max(1, n // 10))
            for k in zipf:
                if c.get(k) is None:
                    c.put(k, k)
            return c
        m, s = timeit(lru, reps)
        c = lru()
        rec.add(structure="LRUCache", op="get/put", n=n, reps=reps, mean_s=m, std_s=s, per_op_us=m / n * 1e6,
                expected="O(1)", extra=f"hit_rate={c.hit_rate:.3f}")

        # S4 B+Tree en disco: carga masiva + 1000 búsquedas + inserciones aleatorias (hasta 1M)
        sk = np.unique(np.array(keys, dtype=np.int64) >> 1)
        q = [int(x) for x in rng.choice(sk, 1000)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.bpt")

            def bulk(_=None):
                t = BPlusTree(path)
                t.bulk_load(sk, np.arange(sk.size))
                t.close()
            m, s = timeit(bulk, max(1, reps // 10) if n >= 100_000 else min(reps, 20))
            rec.add(structure="BPlusTree", op="bulk_load", n=n, reps=max(1, reps // 10) if n >= 100_000 else min(reps, 20),
                    mean_s=m, std_s=s, per_op_us=m / n * 1e6, expected="O(n)")
            t = BPlusTree(path, create=False, cache_pages=64)
            m, s = timeit(lambda: [t.search(x) for x in q], min(reps, 20))
            rec.add(structure="BPlusTree", op="search(1000)", n=n, reps=min(reps, 20), mean_s=m, std_s=s,
                    per_op_us=m / 1000 * 1e6, expected="O(log_B n)",
                    extra=f"height={t.height};pages={t.n_pages};file_MB={t.file_size() / 2**20:.1f}")
            lo = int(np.quantile(sk, 0.5))
            m, s = timeit(lambda: list(t.range_search(lo, lo + (1 << 61) // 1000)), min(reps, 20))
            rec.add(structure="BPlusTree", op="range(0.1%)", n=n, reps=min(reps, 20), mean_s=m, std_s=s,
                    per_op_us=m * 1e6, expected="O(log_B n + k/B)")
            t.close()
            if n <= 1_000_000:
                ins = keys[: n]

                def inserts(_=None):
                    tt = BPlusTree(os.path.join(d, "i.bpt"), cache_pages=256)
                    for k in ins:
                        tt.insert(k, 1)
                    tt.close()
                    return tt
                r = 1 if n >= 100_000 else min(reps, 10)
                m, s = timeit(inserts, r)
                rec.add(structure="BPlusTree", op="insert", n=n, reps=r, mean_s=m, std_s=s, per_op_us=m / n * 1e6,
                        expected="O(log_B n)")

        # S5 KD-Tree (3D): construcción + 100 KNN, contra fuerza bruta
        pts = rng.random((n, 3))
        qs = rng.random((100, 3))
        rb = max(1, reps // 10) if n >= 100_000 else min(reps, 20)
        m, s = timeit(lambda: KDTree(pts), rb)
        tree = KDTree(pts)
        rec.add(structure="KDTree", op="build", n=n, reps=rb, mean_s=m, std_s=s, per_op_us=m / n * 1e6,
                expected="O(n log n)", mem_current_B=tree.memory_bytes())
        m, s = timeit(lambda: [tree.query(x, 10) for x in qs], min(reps, 10))
        rec.add(structure="KDTree", op="knn10(100 consultas)", n=n, reps=min(reps, 10), mean_s=m, std_s=s,
                per_op_us=m / 100 * 1e6, expected="O(log n)")
        m, s = timeit(lambda: [knn_brute_force(pts, x, 10) for x in qs], min(reps, 10))
        rec.add(structure="KNN fuerza bruta", op="knn10(100 consultas)", n=n, reps=min(reps, 10), mean_s=m,
                std_s=s, per_op_us=m / 100 * 1e6, expected="O(n)")
        del pts, tree, keys, zipf


def _fill_queue(keys):
    q = LinkedQueue()
    for k in keys:
        q.enqueue(k)
    return q


def _fill_hash(keys):
    t = HashTable()
    for k in keys:
        t.put(k, 1)
    return t
