"""Cada estructura se valida contra su versión exacta o de referencia."""
import os
import random

import numpy as np
import pytest

from eda.structures import (BinaryHeap, BloomFilter, BPlusTree, CountMinSketch, CSRMatrix, DOKAccumulator,
                            HashTable, HNSWIndex, IVFIndex, KDTree, LinkedQueue, LRUCache, MinHashLSH, Trie,
                            WindowedCountMin, binary_search, community_sizes, contains_sorted, exact_knn,
                            jaccard_brute_force, knn_brute_force, label_propagation, louvain,
                            merge_sort, merge_sorted_arrays, modularity, nmi, pagerank, price_key,
                            price_range_keys, randomized_svd, recall_at_k, searchsorted_vec, symmetric_csr, top_k)


def test_linked_queue_fifo_and_capacity():
    q = LinkedQueue(capacity=3)
    for i in range(5):
        q.enqueue(i)
    assert len(q) == 3 and q.dropped == 2
    assert q.dequeue_batch(10) == [0, 1, 2]
    with pytest.raises(IndexError):
        q.dequeue()


def test_merge_sort_stable():
    rng = random.Random(0)
    items = [(rng.randint(0, 20), i) for i in range(2000)]
    assert merge_sort(items, key=lambda t: t[0]) == sorted(items, key=lambda t: t[0])


def test_binary_search_variants():
    arr = np.array(sorted(random.Random(1).sample(range(100000), 5000)))
    q = np.random.default_rng(2).integers(0, 100000, 3000)
    assert np.array_equal(searchsorted_vec(arr, q), np.searchsorted(arr, q))
    assert np.array_equal(contains_sorted(arr, q), np.isin(q, arr))
    assert binary_search(arr.tolist(), int(arr[77])) == 77
    assert binary_search(arr.tolist(), -5) == -1
    b = np.array([3, 50, 99999, 7])
    b.sort()
    assert np.array_equal(merge_sorted_arrays(arr, b), np.sort(np.concatenate([arr, b])))


def test_heap_order_and_fifo_ties():
    h = BinaryHeap()
    data = [(random.Random(3).randint(0, 5), i) for i in range(1000)]
    for p, i in data:
        h.push(p, i)
    out = [h.pop() for _ in range(len(data))]
    assert out == sorted(data, key=lambda t: t[0])  # estable: FIFO dentro de cada prioridad
    assert [x for x, _ in top_k([(i, float(i % 97)) for i in range(1000)], 3)] == [96, 193, 290]


def test_hash_table_matches_dict():
    t, d = HashTable(), {}
    rng = random.Random(4)
    for _ in range(20000):
        k = rng.randint(0, 3000)
        op = rng.random()
        if op < 0.6:
            t.put(k, k * 2)
            d[k] = k * 2
        elif op < 0.8:
            assert t.remove(k) == (d.pop(k, None) is not None)
        else:
            assert t.get(k) == d.get(k)
    assert len(t) == len(d) and dict(t.items()) == d


def test_lru_eviction_order():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")
    c.put("c", 3)  # expulsa b (menos reciente)
    assert "b" not in c and c.get("a") == 1 and c.get("c") == 3


def test_sparse_matrix_and_svd():
    rng = np.random.default_rng(5)
    dense = (rng.random((60, 40)) < 0.1) * rng.integers(1, 5, (60, 40))
    r, c = np.nonzero(dense)
    A = CSRMatrix.from_coo(r, c, dense[r, c], dense.shape)
    x = rng.random(40)
    assert np.allclose(A.matvec(x), dense @ x)
    assert np.allclose(A.transpose().to_dense(), dense.T)
    acc = DOKAccumulator(60, 40)
    for i, j in zip(r, c):
        acc.add(int(i), int(j), float(dense[i, j]))
    assert np.allclose(acc.to_csr().to_dense(), dense)
    U, s, Vt = randomized_svd(A, 5)
    assert np.allclose(s, np.linalg.svd(dense, compute_uv=False)[:5], rtol=1e-2)


def test_bplus_tree_insert_range_and_reopen(tmp_path):
    path = str(tmp_path / "idx.bpt")
    t = BPlusTree(path, page_size=512, cache_pages=8)
    rng = random.Random(6)
    keys = rng.sample(range(10 ** 6), 5000)
    for k in keys:
        t.insert(k, k + 1)
    assert t.height >= 3
    assert all(t.search(k) == k + 1 for k in keys[:500])
    assert t.search(-1) is None
    srt = sorted(keys)
    got = list(t.range_search(srt[100], srt[400]))
    assert [k for k, _ in got] == srt[100:401]
    t.close()
    t2 = BPlusTree(path, page_size=512, create=False)
    assert t2.search(keys[0]) == keys[0] + 1 and len(t2) == 5000


def test_bplus_tree_bulk_load(tmp_path):
    t = BPlusTree(str(tmp_path / "b.bpt"), page_size=512)
    keys = np.arange(0, 200000, 7)
    t.bulk_load(keys, keys * 3)
    assert t.search(7 * 1234) == 7 * 1234 * 3 and t.search(8) is None
    assert [k for k, _ in t.range_search(70, 140)] == list(range(70, 141, 7))
    t.insert(8, 1)
    assert t.search(8) == 1
    lo, hi = price_range_keys(10.0, 20.0)
    assert lo <= price_key(15.5, 42) <= hi


def test_kdtree_matches_brute_force():
    rng = np.random.default_rng(7)
    pts = rng.random((3000, 3))
    tree = KDTree(pts, leaf_size=8)
    for q in rng.random((20, 3)):
        a, _ = tree.query(q, 10)
        b, _ = knn_brute_force(pts, q, 10)
        assert set(a.tolist()) == set(b.tolist())
    lo, hi = np.array([0.2, 0.2, 0.2]), np.array([0.5, 0.6, 0.7])
    inside = np.where(np.all((pts >= lo) & (pts <= hi), axis=1))[0]
    assert set(tree.range_query(lo, hi).tolist()) == set(inside.tolist())


def test_bloom_no_false_negatives_and_fp_rate():
    bf = BloomFilter(20000, 0.01)
    keys = np.arange(0, 40000, 2)
    bf.add_many(keys)
    assert bf.contains_many(keys).all()
    fp = bf.contains_many(np.arange(1, 200001, 2)).mean()
    assert fp < 0.02
    bf2 = BloomFilter(100, 0.01)
    bf2.add("tarjeta-123")
    assert "tarjeta-123" in bf2
    assert np.array_equal(BloomFilter(1000, 0.01).contains_many([5]), [False])


def test_count_min_never_underestimates():
    cms = CountMinSketch(eps=0.01, delta=0.01)
    rng = np.random.default_rng(8)
    keys = rng.zipf(1.3, 50000) % 5000
    cms.update_many(keys)
    true = np.bincount(keys, minlength=5000)
    est = cms.estimate_many(np.arange(5000))
    assert np.all(est >= true)
    assert np.mean(est - true) <= cms.eps * cms.total
    c2 = CountMinSketch(eps=0.01, delta=0.01)
    for k in keys[:2000].tolist():
        c2.update(k)
    assert c2.estimate(int(keys[0])) >= int((keys[:2000] == keys[0]).sum())


def test_windowed_cms_expires():
    w = WindowedCountMin(window_seconds=60, buckets=6)
    for t in range(0, 30):
        w.update("card-1", ts=t)
    assert w.estimate("card-1", ts=30) >= 30
    assert w.estimate("card-1", ts=200) == 0


def test_minhash_lsh_finds_similar_sets():
    rng = np.random.default_rng(9)
    base = [np.unique(rng.integers(0, 5000, 60)) for _ in range(200)]
    sets = []
    for b in base:  # pares casi duplicados (J alto)
        sets.append(b)
        sets.append(np.unique(np.concatenate([b[:55], rng.integers(0, 5000, 5)])))
    indptr = np.concatenate([[0], np.cumsum([len(s) for s in sets])])
    indices = np.concatenate(sets)
    lsh = MinHashLSH(128, 32)
    sig = lsh.signatures_csr(indptr, indices)
    assert np.array_equal(sig[3], lsh.signature(sets[3]))
    lsh.index(sig)
    hits = 0
    for i in range(0, 400, 2):
        res = lsh.query(sig[i], k=1, exclude=i)
        exact = jaccard_brute_force(indptr, indices, sets[i], k=1, exclude=i)
        hits += bool(res) and res[0][0] == exact[0][0]
    assert hits / 200 > 0.95


def test_ann_recall():
    rng = np.random.default_rng(10)
    centers = rng.standard_normal((20, 16))
    data = centers[rng.integers(0, 20, 3000)] + 0.3 * rng.standard_normal((3000, 16))
    ivf = IVFIndex(nprobe=8).build(data)
    hnsw = HNSWIndex(M=12, ef_construction=80, ef_search=60).build(data[:1500])
    r_ivf, r_h = [], []
    for q in rng.standard_normal((20, 16)):
        qn = q / np.linalg.norm(q)
        r_ivf.append(recall_at_k(ivf.query(qn, 10), exact_knn(ivf.data, qn, 10)))
        r_h.append(recall_at_k(hnsw.query(q, 10), exact_knn(hnsw.data, qn.astype(np.float32), 10)))
    assert np.mean(r_ivf) > 0.85 and np.mean(r_h) > 0.85


def _planted_graph(seed=11, groups=8, size=40):
    rng = np.random.default_rng(seed)
    n = groups * size
    g = np.arange(n) // size
    u, v = np.triu_indices(n, 1)
    p = np.where(g[u] == g[v], 0.3, 0.005)
    keep = rng.random(u.size) < p
    return symmetric_csr(u[keep], v[keep], np.ones(keep.sum()), n), g


def test_pagerank_matches_networkx():
    nx = pytest.importorskip("networkx")
    A, _ = _planted_graph()
    r, _ = pagerank(A)
    G = nx.Graph()
    rows = A.row_ids()
    G.add_weighted_edges_from(zip(rows.tolist(), A.indices.tolist(), A.data.tolist()))
    ref = nx.pagerank(G, alpha=0.85, tol=1e-12)
    assert np.allclose(r, [ref[i] for i in range(A.shape[0])], atol=1e-6)
    assert abs(r.sum() - 1) < 1e-9


def test_communities_recover_planted_partition():
    A, truth = _planted_graph()
    lv = louvain(A, seed=0)
    lp = label_propagation(A, seed=0)
    assert nmi(lv, truth) > 0.9
    assert modularity(A, lv) >= modularity(A, lp) - 0.05
    assert modularity(A, truth) > 0.6
    assert community_sizes(lv).sum() == A.shape[0]


def test_trie_autocomplete():
    t = Trie()
    for term, w in [("samsung galaxy", 50), ("samsung tv", 80), ("sony tv", 30), ("samsonite", 5)]:
        t.add(term, w)
    t.add("samsung galaxy", 40)
    assert t.complete("sam", 2) == [("samsung galaxy", 90), ("samsung tv", 80)]
    assert t.complete("x") == [] and "sony tv" in t
