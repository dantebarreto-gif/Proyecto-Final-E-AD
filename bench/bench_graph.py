"""Experimentos E6 y E7: PageRank frente a grado, y Louvain frente a Label Propagation."""
from __future__ import annotations

import time

import numpy as np

from common import Recorder, SEED, timeit
from eda.structures import (community_sizes, degree_ranking, label_propagation, louvain, modularity, nmi, pagerank,
                            symmetric_csr)


def planted_graph(m: int, rng: np.random.Generator, avg_degree: int = 10, mix: float = 0.2):
    """Grafo con comunidades plantadas de tamaño heterogéneo y m aristas (aprox.)."""
    n = max(20, 2 * m // avg_degree)
    k = max(2, int(np.sqrt(n) / 2))
    w = rng.pareto(1.5, k) + 1
    truth = rng.choice(k, n, p=w / w.sum())
    members = [np.where(truth == c)[0] for c in range(k)]
    sizes = np.array([len(x) for x in members])
    # nodos con grado heterogéneo (ley de potencia) dentro de cada comunidad
    fitness = rng.pareto(2.0, n) + 1
    intra = rng.random(m) >= mix
    u = rng.choice(n, m, p=fitness / fitness.sum())
    v = np.empty(m, dtype=np.int64)
    idx_intra = np.where(intra)[0]
    ci = truth[u[idx_intra]]
    order = np.argsort(ci, kind="stable")
    bounds = np.searchsorted(ci[order], np.arange(k + 1))
    for c in range(k):
        sel = idx_intra[order[bounds[c]:bounds[c + 1]]]
        if sel.size == 0:
            continue
        mem = members[c]
        f = fitness[mem]
        v[sel] = mem[rng.choice(mem.size, sel.size, p=f / f.sum())]
    v[~intra] = rng.integers(0, n, (~intra).sum())
    keep = u != v
    a, b = np.minimum(u[keep], v[keep]), np.maximum(u[keep], v[keep])
    key, w8 = np.unique(a.astype(np.int64) * n + b, return_counts=True)
    return symmetric_csr(key // n, key % n, w8.astype(float), n), truth


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run_pagerank(sizes) -> None:
    rec = Recorder("pagerank_vs_degree")
    rng = np.random.default_rng(SEED + 4)
    for m in sizes:
        A, _ = planted_graph(m, rng)
        reps = 10 if m <= 100_000 else 3 if m <= 1_000_000 else 1
        m_pr, s_pr = timeit(lambda: pagerank(A), reps)
        r, iters = pagerank(A)
        m_dg, s_dg = timeit(lambda: degree_ranking(A), reps)
        deg = degree_ranking(A)
        k = 100
        overlap = len(set(np.argsort(-r)[:k]) & set(np.argsort(-deg)[:k])) / k
        rec.add(method="PageRank", edges=int(A.nnz // 2), nodes=A.shape[0], reps=reps, time_s=m_pr, std_s=s_pr,
                iterations=iters, spearman_vs_degree=spearman(r, deg), top100_overlap=overlap,
                memory_B=A.memory_bytes())
        rec.add(method="Grado ponderado", edges=int(A.nnz // 2), nodes=A.shape[0], reps=reps, time_s=m_dg,
                std_s=s_dg, iterations=1)
        print(f"[E6] m={m:,} PageRank {m_pr:.3f}s, {iters} iteraciones")


LOUVAIN_MAX_EDGES = 1_000_000


def run_communities(sizes, max_louvain_edges: int = None) -> None:
    max_louvain_edges = max_louvain_edges or LOUVAIN_MAX_EDGES
    rec = Recorder("communities")
    rng = np.random.default_rng(SEED + 5)
    for m in sizes:
        A, truth = planted_graph(m, rng)
        seeds = range(5) if m <= 1_000_000 else range(1)
        for name, fn in (("Louvain", louvain), ("Label Propagation", label_propagation)):
            if name == "Louvain" and m > max_louvain_edges:
                print(f"[E7] m={m:,} Louvain omitido: supera {max_louvain_edges:,} aristas en Python puro")
                continue
            labels, times = [], []
            for sd in seeds:
                t = time.perf_counter()
                labels.append(fn(A, seed=sd))
                times.append(time.perf_counter() - t)
            lab = labels[0]
            sz = community_sizes(lab)
            stab = [nmi(labels[0], l) for l in labels[1:]]
            rec.add(method=name, edges=int(A.nnz // 2), nodes=A.shape[0], runs=len(labels),
                    time_s=float(np.mean(times)), std_s=float(np.std(times)), modularity=modularity(A, lab),
                    modularity_truth=modularity(A, truth), n_communities=int(sz.size), largest=int(sz[0]),
                    median_size=float(np.median(sz)), singletons=int((sz == 1).sum()),
                    stability_nmi=float(np.mean(stab)) if stab else float("nan"), nmi_vs_truth=nmi(lab, truth))
            print(f"[E7] m={m:,} {name}: {np.mean(times):.2f}s")
