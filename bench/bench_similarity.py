"""Experimentos E4 y E5: MinHash+LSH contra fuerza bruta e IVF/HNSW contra KNN exacto."""
from __future__ import annotations

import time

import numpy as np

from common import Recorder, SEED, memory, timeit
from eda.structures import (HNSWIndex, IVFIndex, KDTree, MinHashLSH, exact_knn, jaccard_brute_force, normalize,
                            recall_at_k)


def make_sets(n: int, rng: np.random.Generator, universe: int = 1_000_000, avg: int = 12):
    """n conjuntos (compradores por producto) con grupos de casi duplicados.

    Un 20 % de los conjuntos son variaciones de otro (cambian ~20 % de los
    elementos), así existen vecinos con Jaccard alto que el LSH debe hallar.
    """
    sizes = np.maximum(3, rng.poisson(avg, n))
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    indices = rng.integers(0, universe, indptr[-1])
    variants = rng.random(n) < 0.2
    src = rng.integers(0, n, n)
    for i in np.where(variants)[0][: n // 5]:
        j = src[i]
        a, b = indptr[i], indptr[i + 1]
        sa, sb = indptr[j], indptr[j + 1]
        m = min(b - a, sb - sa)
        keep = rng.random(m) < 0.8
        indices[a:a + m] = np.where(keep, indices[sa:sa + m], indices[a:a + m])
    # normalizar cada conjunto (ordenado y sin repetidos) para una CSR válida
    rows = np.repeat(np.arange(n), sizes)
    key = rows * universe + indices
    key = np.unique(key)
    rows, indices = key // universe, key % universe
    counts = np.bincount(rows, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, indices


def run_minhash(sizes) -> None:
    rec = Recorder("minhash_vs_bruteforce")
    rng = np.random.default_rng(SEED + 2)
    for n in sizes:
        if n > 1_000_000:
            print("[E4] 10M conjuntos: firmas de 10 GB (128 x 8 B x 10M); se omite por memoria")
            continue
        indptr, indices = make_sets(n, rng)
        qs = rng.choice(n, 100, replace=False)
        print(f"[E4] n={n:,} conjuntos, nnz={indices.size:,}")
        for bands, num_perm in ((32, 128),) if n != 100_000 else ((16, 128), (32, 128), (64, 128)):
            def build():
                l = MinHashLSH(num_perm=num_perm, bands=bands)
                l.index(l.signatures_csr(indptr, indices))
                return l
            reps = 3 if n <= 100_000 else 1
            m_b, s_b = timeit(build, reps)
            cur, peak, lsh = memory(build)
            t = time.perf_counter()
            res = [lsh.query(lsh.signatures[q], k=10, exclude=int(q)) for q in qs]
            t_q = (time.perf_counter() - t) / len(qs)
            t = time.perf_counter()
            truth = [jaccard_brute_force(indptr, indices, indices[indptr[q]:indptr[q + 1]], k=10, exclude=int(q))
                     for q in qs]
            t_bf = (time.perf_counter() - t) / len(qs)
            # recall@10 sobre los vecinos verdaderos con Jaccard >= umbral del LSH
            rec_all, rec_thr, cands = [], [], []
            for q, r, tr in zip(qs, res, truth):
                found = {i for i, _ in r}
                rel = [i for i, j in tr if j >= 0.5]
                if rel:
                    rec_thr.append(len(found & set(rel)) / len(rel))
                if tr:
                    rec_all.append(len(found & {i for i, _ in tr}) / len(tr))
                cands.append(lsh.candidates(lsh.signatures[q]).size)
            rec.add(structure="MinHash+LSH", n=n, nnz=int(indices.size), num_perm=num_perm, bands=bands,
                    rows=num_perm // bands, threshold=lsh.threshold, build_s=m_b, build_std=s_b,
                    query_ms=t_q * 1e3, bruteforce_query_ms=t_bf * 1e3, speedup=t_bf / t_q,
                    recall_at10_J05=float(np.mean(rec_thr)) if rec_thr else float("nan"),
                    recall_at10_all=float(np.mean(rec_all)) if rec_all else float("nan"),
                    avg_candidates=float(np.mean(cands)), memory_B=lsh.memory_bytes(), mem_peak_B=peak,
                    bruteforce_memory_B=int(indptr.nbytes + indices.nbytes))
            del lsh


def make_vectors(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Mezcla de gaussianas (embeddings agrupados por categoría)."""
    k = max(10, int(np.sqrt(n) / 4))
    centers = rng.standard_normal((k, d)).astype(np.float32)
    out = np.empty((n, d), dtype=np.float32)
    for s in range(0, n, 1_000_000):
        e = min(n, s + 1_000_000)
        out[s:e] = centers[rng.integers(0, k, e - s)] + 0.6 * rng.standard_normal((e - s, d)).astype(np.float32)
    return out


def run_ann(sizes, d: int = 32) -> None:
    rec = Recorder("ann_vs_exact")
    rng = np.random.default_rng(SEED + 3)
    for n in sizes:
        x = make_vectors(n, d, rng)
        xn = normalize(x)
        qs = normalize(rng.standard_normal((100, d)).astype(np.float32) + x[rng.integers(0, n, 100)])
        print(f"[E5] n={n:,} d={d}")
        t = time.perf_counter()
        truth = [exact_knn(xn, q, 10) for q in qs]
        t_exact = (time.perf_counter() - t) / 100
        rec.add(method="Fuerza bruta (exacto)", n=n, d=d, build_s=0.0, query_ms=t_exact * 1e3, recall_at10=1.0,
                memory_B=int(xn.nbytes))
        # IVF: nlist = 4 sqrt(n) acotado a 4096
        nlist = int(min(4096, max(8, 4 * np.sqrt(n))))
        t = time.perf_counter()
        ivf = IVFIndex(nlist=nlist, nprobe=8).build(x)
        t_build = time.perf_counter() - t
        for nprobe in (1, 4, 8, 16, 32) if n == 1_000_000 else (8, 16):
            t = time.perf_counter()
            found = [ivf.query(q, 10, nprobe=nprobe) for q in qs]
            tq = (time.perf_counter() - t) / 100
            r = np.mean([recall_at_k(f, tr) for f, tr in zip(found, truth)])
            rec.add(method="IVF", n=n, d=d, nlist=nlist, nprobe=nprobe, build_s=t_build, query_ms=tq * 1e3,
                    recall_at10=float(r), speedup=t_exact / tq, memory_B=ivf.memory_bytes())
        del ivf
        # KD-Tree exacto en 32 dimensiones: muestra la maldición de la dimensionalidad
        if n <= 100_000:
            t = time.perf_counter()
            kd = KDTree(xn)
            t_build = time.perf_counter() - t
            t = time.perf_counter()
            found = [kd.query(q, 10)[0] for q in qs[:20]]
            tq = (time.perf_counter() - t) / 20
            r = np.mean([recall_at_k(f, tr) for f, tr in zip(found, truth[:20])])
            rec.add(method="KD-Tree (exacto, d=32)", n=n, d=d, build_s=t_build, query_ms=tq * 1e3,
                    recall_at10=float(r), speedup=t_exact / tq, memory_B=kd.memory_bytes())
        # HNSW en Python puro: construcción O(n log n) pero con constante alta
        if n <= 100_000:
            t = time.perf_counter()
            h = HNSWIndex(M=16, ef_construction=100, ef_search=50).build(x)
            t_build = time.perf_counter() - t
            for ef in (10, 50, 100) if n == 100_000 else (50,):
                h.dist_evals = 0
                t = time.perf_counter()
                found = [h.query(q, 10, ef=ef) for q in qs]
                tq = (time.perf_counter() - t) / 100
                r = np.mean([recall_at_k(f, tr) for f, tr in zip(found, truth)])
                rec.add(method="HNSW", n=n, d=d, ef=ef, M=16, build_s=t_build, query_ms=tq * 1e3,
                        recall_at10=float(r), speedup=t_exact / tq, memory_B=h.memory_bytes(),
                        dist_evals=h.dist_evals / 100)
            del h
        del x, xn
