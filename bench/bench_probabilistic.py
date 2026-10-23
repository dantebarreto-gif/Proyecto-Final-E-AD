"""Experimentos E2 y E3: Bloom Filter y Count-Min Sketch frente a sus alternativas exactas."""
from __future__ import annotations

import numpy as np

from common import Recorder, SEED, memory, reps_for, timeit
from eda.structures import BloomFilter, CountMinSketch, HashSet, HashTable


def run_bloom(sizes) -> None:
    rec = Recorder("bloom_vs_exact")
    rng = np.random.default_rng(SEED)
    for n in sizes:
        reps = reps_for(n)
        keys = rng.choice(2 ** 62, n, replace=False).astype(np.int64) * 2  # pares: presentes
        neg = rng.integers(0, 2 ** 61, n).astype(np.int64) * 2 + 1          # impares: ausentes
        queries = np.concatenate([keys[: n // 2], neg[: n // 2]])
        print(f"[E2] n={n:,} reps={reps}")
        for p in (0.01,) if n != 1_000_000 else (0.1, 0.01, 0.001):
            def build():
                b = BloomFilter(n, p)
                b.add_many(keys)
                return b
            m_ins, s_ins = timeit(build, reps)
            bf = build()
            m_q, s_q = timeit(lambda: bf.contains_many(queries), reps)
            fpr = float(bf.contains_many(neg).mean())
            cur, peak, _ = memory(build)
            rec.add(structure="BloomFilter", p=p, n=n, reps=reps, insert_s=m_ins, insert_std=s_ins,
                    query_s=m_q, query_std=s_q, insert_us_op=m_ins / n * 1e6, query_us_op=m_q / n * 1e6,
                    memory_B=bf.memory_bytes(), mem_peak_B=peak, fp_rate=fpr,
                    fp_theoretical=bf.expected_fp_rate(), k=bf.k, bits_per_item=bf.m / n,
                    false_negatives=int((~bf.contains_many(keys)).sum()))
        # escalar (Python puro, un elemento a la vez) para comparar justo con el set propio
        kl, ql = keys.tolist(), queries.tolist()
        r = max(1, reps // 10) if n >= 1_000_000 else reps
        if n <= 1_000_000:
            def build_scalar():
                b = BloomFilter(n, 0.01)
                for k in kl:
                    b.add(k)
                return b
            m_ins, s_ins = timeit(build_scalar, max(1, r // 5) if n >= 100_000 else r)
            b = build_scalar()
            m_q, s_q = timeit(lambda: [k in b for k in ql], max(1, r // 5) if n >= 100_000 else r)
            rec.add(structure="BloomFilter (escalar)", p=0.01, n=n, reps=r, insert_s=m_ins, insert_std=s_ins,
                    query_s=m_q, query_std=s_q, insert_us_op=m_ins / n * 1e6, query_us_op=m_q / n * 1e6,
                    memory_B=b.memory_bytes(), fp_rate=float(np.mean([int(x) in b for x in neg[:20000]])))

        def build_exact():
            s = HashSet()
            for k in kl:
                s.add(k)
            return s
        m_ins, s_ins = timeit(build_exact, r)
        cur, peak, hs = memory(build_exact)
        m_q, s_q = timeit(lambda: [k in hs for k in ql], r)
        rec.add(structure="HashSet exacto", p=0.0, n=n, reps=r, insert_s=m_ins, insert_std=s_ins, query_s=m_q,
                query_std=s_q, insert_us_op=m_ins / n * 1e6, query_us_op=m_q / n * 1e6, memory_B=cur,
                mem_peak_B=peak, fp_rate=0.0, false_negatives=0)
        cur, peak, ps = memory(lambda: set(kl))
        rec.add(structure="set de Python", p=0.0, n=n, reps=1, memory_B=cur, mem_peak_B=peak, fp_rate=0.0)
        del hs, ps, kl, ql


def run_cms(sizes) -> None:
    rec = Recorder("cms_vs_exact")
    rng = np.random.default_rng(SEED + 1)
    for n in sizes:
        reps = reps_for(n)
        universe = max(100, n // 10)
        stream = (rng.zipf(1.2, n) % universe).astype(np.int64)
        true = np.bincount(stream, minlength=universe)
        keys = np.arange(universe)
        heavy = np.argsort(-true)[:100]
        print(f"[E3] n={n:,} reps={reps}")
        for eps in ((0.001,) if n != 1_000_000 else (0.01, 0.001, 0.0001)):
            def build():
                c = CountMinSketch(eps=eps, delta=0.01)
                c.update_many(stream)
                return c
            m_u, s_u = timeit(build, reps)
            c = build()
            m_q, s_q = timeit(lambda: c.estimate_many(keys), reps)
            est = c.estimate_many(keys).astype(np.int64)
            err = est - true
            cur, peak, _ = memory(build)
            rec.add(structure="CountMinSketch", eps=eps, delta=0.01, n=n, distinct=universe, reps=reps,
                    update_s=m_u, update_std=s_u, query_s=m_q, query_std=s_q, update_us_op=m_u / n * 1e6,
                    memory_B=c.memory_bytes(), mem_peak_B=peak, width=c.w, depth=c.d,
                    mean_abs_err=float(err.mean()), max_err=int(err.max()), bound_epsN=c.eps * n,
                    within_bound=float((err <= c.eps * n).mean()), underestimates=int((err < 0).sum()),
                    heavy_rel_err=float((err[heavy] / np.maximum(true[heavy], 1)).mean()))
        sl = stream.tolist()
        r = max(1, reps // 10) if n >= 1_000_000 else reps

        def exact():
            t = HashTable()
            for k in sl:
                t.increment(k)
            return t
        m_u, s_u = timeit(exact, r)
        cur, peak, t = memory(exact)
        rec.add(structure="HashTable exacta", eps=0.0, n=n, distinct=universe, reps=r, update_s=m_u,
                update_std=s_u, update_us_op=m_u / n * 1e6, memory_B=cur, mem_peak_B=peak, mean_abs_err=0.0,
                max_err=0, within_bound=1.0, underestimates=0)
        if n <= 1_000_000:
            def scalar():
                c = CountMinSketch(eps=0.001, delta=0.01)
                for k in sl:
                    c.update(k)
                return c
            rs = 1 if n >= 100_000 else min(r, 10)
            m_u, s_u = timeit(scalar, rs)
            rec.add(structure="CountMinSketch (escalar)", eps=0.001, delta=0.01, n=n, distinct=universe, reps=rs,
                    update_s=m_u, update_std=s_u, update_us_op=m_u / n * 1e6)
        del t, sl
