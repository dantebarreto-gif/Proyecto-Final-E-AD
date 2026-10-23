"""Genera las figuras del informe (report/figures/*.png) a partir de results/bench/*.csv.

Todas las escalas de tamaño son log-log, con la pendiente ajustada (exponente
empírico b en t ~ n^b) para contrastarla con la complejidad esperada.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results", "bench")
FIG = os.path.join(ROOT, "report", "figures")

# paleta categórica en orden fijo; el color sigue al método en todas las figuras
C = {"ours": "#2a78d6", "exact": "#eb6834", "third": "#1baf7a", "fourth": "#6250d6", "fifth": "#e87ba4"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
})


def load(name: str) -> pd.DataFrame:
    p = os.path.join(RES, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)[0])


def line(ax, x, y, color, label, err=None, ls="-", marker="o"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = slope(x, y)
    lab = f"{label} (pend. {b:.2f})" if np.isfinite(b) else label
    ax.plot(x, y, ls, color=color, marker=marker, label=lab)
    if err is not None:
        e = np.asarray(err, float)
        ax.fill_between(x, np.maximum(y - e, y * 0.01), y + e, color=color, alpha=0.15, linewidth=0)
    return b


def loglog(ax, title, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("figura:", name)


def fig_base(slopes):
    df = load("base_structures")
    if df.empty:
        return
    panels = [
        ("S0 Cola FIFO", [("LinkedQueue", "enqueue+dequeue", C["ours"])], "O(n)"),
        ("S1 Ordenación", [("MergeSort", "sort", C["ours"])], "O(n log n)"),
        ("S2 Heap", [("BinaryHeap", "push+pop", C["ours"])], "O(n log n)"),
        ("S3 Hash y LRU", [("HashTable", "put+get", C["ours"]), ("LRUCache", "get/put", C["third"])], "O(n)"),
        ("S4 B+Tree", [("BPlusTree", "bulk_load", C["ours"]), ("BPlusTree", "insert", C["third"])], "O(n) / O(n log n)"),
        ("S5 KNN (100 consultas)", [("KDTree", "knn10(100 consultas)", C["ours"]),
                                    ("KNN fuerza bruta", "knn10(100 consultas)", C["exact"])], "O(log n) vs O(n)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (title, series, exp) in zip(axes.ravel(), panels):
        for st, op, col in series:
            d = df[(df.structure == st) & (df.op == op)].sort_values("n")
            if d.empty:
                continue
            b = line(ax, d.n, d.mean_s, col, f"{st} {op}", d.std_s)
            slopes.append({"Estructura": st, "Operación": op, "Pendiente": b, "Esperado": exp})
        loglog(ax, title, "n", "tiempo total (s)")
        ax.legend(fontsize=7, loc="upper left")
    save(fig, "fig_base_scaling.png")
    d = df[(df.structure == "BPlusTree") & (df.op == "search(1000)")].sort_values("n")
    if not d.empty:
        slopes.append({"Estructura": "BPlusTree", "Operación": "search (por consulta)",
                       "Pendiente": slope(d.n, d.per_op_us), "Esperado": "O(log_B n) ≈ 0"})


def fig_bloom(slopes):
    df = load("bloom_vs_exact")
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    b = df[(df.structure == "BloomFilter") & (df.p == 0.01)].sort_values("n")
    h = df[df.structure == "HashSet exacto"].sort_values("n")
    s = df[df.structure == "set de Python"].sort_values("n")
    bs = df[df.structure == "BloomFilter (escalar)"].sort_values("n")
    ax = axes[0]
    line(ax, b.n, b.memory_B, C["ours"], "Bloom p=1 %")
    line(ax, h.n, h.memory_B, C["exact"], "HashSet propio")
    line(ax, s.n, s.memory_B, C["third"], "set de Python", ls="--", marker="s")
    loglog(ax, "Memoria", "n elementos", "bytes")
    ax.legend(fontsize=7)
    ax = axes[1]
    line(ax, b.n, b.insert_s + b.query_s, C["ours"], "Bloom vectorizado")
    if not bs.empty:
        line(ax, bs.n, bs.insert_s + bs.query_s, C["fifth"], "Bloom escalar", ls="--", marker="s")
    line(ax, h.n, h.insert_s + h.query_s, C["exact"], "HashSet propio")
    loglog(ax, "Tiempo: n inserciones + n consultas", "n elementos", "s")
    ax.legend(fontsize=7)
    ax = axes[2]
    p = df[(df.structure == "BloomFilter") & (df.n == 1_000_000)].sort_values("p")
    if not p.empty:
        x = np.arange(len(p))
        ax.bar(x - 0.18, p.fp_theoretical, 0.34, color=C["exact"], label="teórico")
        ax.bar(x + 0.18, p.fp_rate, 0.34, color=C["ours"], label="medido")
        ax.set_xticks(x, [f"p={v:g}\nk={k}, {bpi:.1f} bits/elem" for v, k, bpi in zip(p.p, p.k, p.bits_per_item)],
                      fontsize=7)
        ax.set_yscale("log")
        ax.set_title("Falsos positivos (n = 1M)", loc="left")
        ax.legend(fontsize=7)
    save(fig, "fig_bloom.png")
    slopes.append({"Estructura": "BloomFilter", "Operación": "insert+query (vect.)",
                   "Pendiente": slope(b.n, b.insert_s + b.query_s), "Esperado": "O(n·k)"})


def fig_cms(slopes):
    df = load("cms_vs_exact")
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    c = df[(df.structure == "CountMinSketch") & (df.eps == 0.001)].sort_values("n")
    h = df[df.structure == "HashTable exacta"].sort_values("n")
    cs = df[df.structure == "CountMinSketch (escalar)"].sort_values("n")
    ax = axes[0]
    line(ax, c.n, c.memory_B, C["ours"], "CMS ε=0.001")
    line(ax, h.n, h.memory_B, C["exact"], "HashTable exacta")
    loglog(ax, "Memoria (claves distintas = n/10)", "n actualizaciones", "bytes")
    ax.legend(fontsize=7)
    ax = axes[1]
    line(ax, c.n, c.update_s, C["ours"], "CMS vectorizado")
    if not cs.empty:
        line(ax, cs.n, cs.update_s, C["fifth"], "CMS escalar", ls="--", marker="s")
    line(ax, h.n, h.update_s, C["exact"], "HashTable exacta")
    loglog(ax, "Tiempo de actualización", "n actualizaciones", "s")
    ax.legend(fontsize=7)
    ax = axes[2]
    e = df[(df.structure == "CountMinSketch") & (df.n == 1_000_000)].sort_values("eps", ascending=False)
    if not e.empty:
        x = np.arange(len(e))
        ax.bar(x - 0.18, e.bound_epsN, 0.34, color=C["exact"], label="cota ε·N")
        ax.bar(x + 0.18, e.mean_abs_err, 0.34, color=C["ours"], label="error medio medido")
        ax.set_xticks(x, [f"ε={v:g}\n{w:,} x {d}" for v, w, d in zip(e.eps, e.width, e.depth)], fontsize=7)
        ax.set_yscale("log")
        ax.set_title("Sobreestimación (n = 1M)", loc="left")
        ax.legend(fontsize=7)
    save(fig, "fig_cms.png")
    slopes.append({"Estructura": "CountMinSketch", "Operación": "update (vect.)",
                   "Pendiente": slope(c.n, c.update_s), "Esperado": "O(n·d)"})


def fig_minhash(slopes):
    df = load("minhash_vs_bruteforce")
    if df.empty:
        return
    d = df[df.bands == 32].sort_values("n")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    ax = axes[0]
    line(ax, d.n, d.query_ms, C["ours"], "MinHash+LSH")
    line(ax, d.n, d.bruteforce_query_ms, C["exact"], "Fuerza bruta")
    loglog(ax, "Tiempo por consulta Top-10", "n conjuntos", "ms")
    ax.legend(fontsize=7)
    ax = axes[1]
    line(ax, d.n, d.build_s, C["ours"], "construcción LSH")
    loglog(ax, "Construcción (firmas + índice)", "n conjuntos", "s")
    ax.legend(fontsize=7)
    ax = axes[2]
    v = df[df.n == 100_000].sort_values("bands")
    if not v.empty:
        x = np.arange(len(v))
        ax.bar(x, v.recall_at10_J05, 0.5, color=C["ours"])
        ax.set_xticks(x, [f"b={b}, r={r}\numbral {t:.2f}" for b, r, t in zip(v.bands, v.rows, v.threshold)],
                      fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title("Recall@10 (vecinos con J ≥ 0.5), n = 100K", loc="left")
        for xi, r in zip(x, v.recall_at10_J05):
            ax.text(xi, r + 0.02, f"{r:.2f}", ha="center", fontsize=8, color=INK)
    save(fig, "fig_minhash.png")
    slopes.append({"Estructura": "MinHash+LSH", "Operación": "consulta",
                   "Pendiente": slope(d.n, d.query_ms), "Esperado": "≈ O(1) + candidatos"})
    slopes.append({"Estructura": "Jaccard fuerza bruta", "Operación": "consulta",
                   "Pendiente": slope(d.n, d.bruteforce_query_ms), "Esperado": "O(nnz)"})


def fig_ann(slopes):
    df = load("ann_vs_exact")
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    ax = axes[0]
    ex = df[df.method == "Fuerza bruta (exacto)"].sort_values("n")
    iv = df[(df.method == "IVF") & (df.nprobe == 8)].sort_values("n")
    hn = df[(df.method == "HNSW") & (df.ef == 50)].sort_values("n")
    kd = df[df.method == "KD-Tree (exacto, d=32)"].sort_values("n")
    line(ax, ex.n, ex.query_ms, C["exact"], "Fuerza bruta")
    line(ax, iv.n, iv.query_ms, C["ours"], "IVF nprobe=8")
    if not hn.empty:
        line(ax, hn.n, hn.query_ms, C["third"], "HNSW ef=50")
    if not kd.empty:
        line(ax, kd.n, kd.query_ms, C["fourth"], "KD-Tree d=32", ls="--", marker="s")
    loglog(ax, "Tiempo por consulta (d = 32)", "n vectores", "ms")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(iv.n, iv.recall_at10, "-o", color=C["ours"], label="IVF nprobe=8")
    if not hn.empty:
        ax.plot(hn.n, hn.recall_at10, "-o", color=C["third"], label="HNSW ef=50")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall@10 frente al exacto", loc="left")
    ax.set_xlabel("n vectores")
    ax.legend(fontsize=7)
    ax = axes[2]
    tr = df[(df.method == "IVF") & (df.n == 1_000_000)].sort_values("nprobe")
    if not tr.empty:
        ax.plot(tr.query_ms, tr.recall_at10, "-o", color=C["ours"], label="IVF (nprobe 1 a 32), n=1M")
        for _, r in tr.iterrows():
            ax.annotate(f"{int(r.nprobe)}", (r.query_ms, r.recall_at10), textcoords="offset points",
                        xytext=(4, -10), fontsize=7, color=MUTED)
    th = df[(df.method == "HNSW") & (df.n == 100_000)].sort_values("ef")
    if not th.empty:
        ax.plot(th.query_ms, th.recall_at10, "-s", color=C["third"], label="HNSW (ef 10 a 100), n=100K")
    ax.set_xscale("log")
    ax.set_title("Compromiso recall / latencia", loc="left")
    ax.set_xlabel("ms por consulta")
    ax.set_ylabel("recall@10")
    ax.legend(fontsize=7)
    save(fig, "fig_ann.png")
    slopes.append({"Estructura": "IVF", "Operación": "consulta",
                   "Pendiente": slope(iv.n, iv.query_ms), "Esperado": "O(√n) con nlist ∝ √n"})
    slopes.append({"Estructura": "KNN fuerza bruta (d=32)", "Operación": "consulta",
                   "Pendiente": slope(ex.n, ex.query_ms), "Esperado": "O(n·d)"})


def fig_graph(slopes):
    pr = load("pagerank_vs_degree")
    cm = load("communities")
    if pr.empty and cm.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    ax = axes[0]
    p = pr[pr.method == "PageRank"].sort_values("edges")
    g = pr[pr.method == "Grado ponderado"].sort_values("edges")
    line(ax, p.edges, p.time_s, C["ours"], "PageRank", p.std_s)
    line(ax, g.edges, g.time_s, C["exact"], "Grado (baseline)")
    loglog(ax, "PageRank vs. grado", "aristas", "s")
    ax.legend(fontsize=7)
    ax = axes[1]
    lv = cm[cm.method == "Louvain"].sort_values("edges")
    lp = cm[cm.method == "Label Propagation"].sort_values("edges")
    line(ax, lv.edges, lv.time_s, C["ours"], "Louvain")
    line(ax, lp.edges, lp.time_s, C["third"], "Label Propagation")
    loglog(ax, "Comunidades: tiempo", "aristas", "s")
    ax.legend(fontsize=7)
    ax = axes[2]
    ax.plot(lv.edges, lv.modularity, "-o", color=C["ours"], label="Louvain")
    ax.plot(lp.edges, lp.modularity, "-o", color=C["third"], label="Label Propagation")
    ax.plot(lv.edges, lv.modularity_truth, "--", color=MUTED, label="partición plantada")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_title("Modularidad Q", loc="left")
    ax.set_xlabel("aristas")
    ax.legend(fontsize=7)
    save(fig, "fig_graph.png")
    slopes.append({"Estructura": "PageRank", "Operación": "iteración de potencias",
                   "Pendiente": slope(p.edges, p.time_s), "Esperado": "O(iter·m)"})
    slopes.append({"Estructura": "Louvain", "Operación": "detección",
                   "Pendiente": slope(lv.edges, lv.time_s), "Esperado": "≈ O(m log n)"})
    slopes.append({"Estructura": "Label Propagation", "Operación": "detección",
                   "Pendiente": slope(lp.edges, lp.time_s), "Esperado": "≈ O(m)"})


def main() -> None:
    slopes = []
    for f in (fig_base, fig_bloom, fig_cms, fig_minhash, fig_ann, fig_graph):
        f(slopes)
    out = pd.DataFrame(slopes)
    out["Pendiente"] = out["Pendiente"].round(2)
    out.to_csv(os.path.join(RES, "slopes.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
