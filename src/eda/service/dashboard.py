"""Etapa 9 - Panel del servicio (Streamlit).

    streamlit run src/eda/service/dashboard.py -- --data data/synthetic_200000

Pestañas:
- Demo en vivo: corre el pipeline sobre el dataset elegido y muestra el
  recorrido de un evento por las 10 etapas, alertas y métricas.
- Buscar: autocompletado (Trie) y recomendaciones Top-K para el producto elegido.
- Resultados: el resumen guardado de la corrida de 10M y los experimentos.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eda.pipeline.pipeline import StreamPipeline  # noqa: E402


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    a, _ = ap.parse_known_args()
    return a


@st.cache_resource(show_spinner="Procesando el stream por las 10 etapas...")
def run_pipeline(data_dir: str, limit: int):
    p = StreamPipeline(os.path.join(data_dir, "events.parquet"), os.path.join(data_dir, "catalog.parquet"),
                       workdir=os.path.join(data_dir, "work_dashboard"), batch_size=5_000,
                       refresh_every=max(50_000, limit // 4), limit=limit)
    summary = p.run(limit=limit, verbose=False)
    return p, summary


def main() -> None:
    st.set_page_config(page_title="Pipeline EDA - e-commerce", layout="wide")
    st.title("Pipeline de data streams: comercio electrónico")
    datasets = sorted(glob.glob(os.path.join(ROOT, "data", "*", "events.parquet")))
    default = args().data
    options = [os.path.dirname(d) for d in datasets]
    if default:
        default = os.path.abspath(default)
        if default in options:
            options.remove(default)
        options.insert(0, default)
    tab_live, tab_search, tab_results = st.tabs(["Demo en vivo", "Buscar y recomendar", "Resultados guardados"])

    with tab_live:
        if not options:
            st.info("No hay datos. Ejecuta: python scripts/prepare_data.py --events 200000")
        else:
            c1, c2 = st.columns([3, 1])
            data_dir = c1.selectbox("Dataset", options, format_func=os.path.basename)
            limit = c2.number_input("Eventos a procesar", 10_000, 10_000_000, 200_000, step=50_000)
            if st.button("Procesar stream", type="primary"):
                st.session_state["run"] = (data_dir, int(limit))
            if "run" in st.session_state:
                p, s = run_pipeline(*st.session_state["run"])
                k = st.columns(5)
                k[0].metric("Eventos", f"{s['eventos_recibidos']:,}")
                k[1].metric("Eventos/s", f"{s['eventos_por_segundo']:,}")
                k[2].metric("Duplicados eliminados", f"{s['S1']['duplicados_eliminados']:,}")
                k[3].metric("Bloqueados por fraude", f"{s['S6']['blocked']:,}")
                k[4].metric("Precisión del filtro", f"{s['S6']['precision']:.1%}")
                t = s.get("traza_evento")
                if t:
                    st.subheader("Recorrido de un evento por las etapas")
                    st.code(t["evento"] + "\n" + "\n".join(t["pasos"]), language="text")
                    st.dataframe(pd.DataFrame(t["recomendaciones"])[["name", "category", "price", "puntaje", "fuentes"]],
                                 hide_index=True)
                a, b = st.columns(2)
                a.subheader("Alertas por prioridad (heap)")
                a.dataframe(pd.DataFrame(s["alertas_prioritarias"]), hide_index=True)
                b.subheader("Top ventas (Top-K)")
                b.dataframe(pd.DataFrame(s["top_ventas"])[["name", "category", "price", "ventas"]], hide_index=True)
                st.subheader("Tiempo por etapa (s)")
                st.bar_chart(pd.Series(s["tiempos_por_etapa_s"]).drop("setup", errors="ignore"))
                st.subheader("Grafo de co-compra")
                st.json(s["S8"])
                st.session_state["pipeline"] = p

    with tab_search:
        p = st.session_state.get("pipeline")
        if p is None:
            st.info("Primero procesa el stream en la pestaña Demo en vivo.")
        else:
            q = st.text_input("Buscar producto (autocompletado con Trie)", "sam")
            sugg = p.s9.autocomplete(q, 8) if q else []
            st.write([f"{t} ({w:.0f})" for t, w in sugg])
            cat = p.store.catalog
            names = cat["name"]
            match = cat[names.str.startswith(sugg[0][0])] if sugg else cat.iloc[:0]
            if match.empty and q:
                match = cat[names.str.contains(q, case=False, regex=False)]
            if not match.empty:
                idx = int(match.index[0])
                info = p.store.product_info(idx)
                st.write(f"**{info['name']}** · {info['category']} · ${info['price']:.2f}")
                recs = p.s9.recommend(idx, p.s7, p.s8, p.s5, k=10)
                st.dataframe(pd.DataFrame([{**p.store.product_info(j), "puntaje": s, "fuentes": w}
                                           for j, s, w in recs])[["name", "category", "price", "puntaje", "fuentes"]],
                             hide_index=True)
            lo, hi = st.slider("Productos por rango de precio (B+Tree en disco)", 0, 2000, (100, 150))
            st.write([p.store.product_info(i)["name"] for i in p.s4.products_in_price_range(lo, hi, 15)])

    with tab_results:
        for path in sorted(glob.glob(os.path.join(ROOT, "results", "demo", "summary*.json"))):
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
            st.subheader(os.path.basename(path))
            st.write({k: s[k] for k in ("eventos_recibidos", "segundos", "eventos_por_segundo")})
            st.json({k: s[k] for k in ("S1", "S4", "S6", "S8")}, expanded=False)
        for fig in sorted(glob.glob(os.path.join(ROOT, "report", "figures", "fig_*.png"))):
            st.image(fig, caption=os.path.basename(fig))


main()
