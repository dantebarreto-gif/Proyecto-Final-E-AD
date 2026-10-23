"""Ejecuta el pipeline de punta a punta y muestra las salidas observables.

    python scripts/run_pipeline.py --data data/synthetic_1000000
"""
import argparse
import json
import os

import _path  # noqa: F401
from eda.pipeline.pipeline import StreamPipeline, save_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="carpeta con events.parquet y catalog.parquet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=5_000)
    ap.add_argument("--refresh", type=int, default=250_000)
    ap.add_argument("--out", default="results/demo/summary.json")
    a = ap.parse_args()
    p = StreamPipeline(os.path.join(a.data, "events.parquet"), os.path.join(a.data, "catalog.parquet"),
                       workdir=os.path.join(a.data, "work"), batch_size=a.batch, refresh_every=a.refresh, limit=a.limit)
    print(f"Evento trazado: {p.trace_event}")
    s = p.run(limit=a.limit)
    p.close()
    save_summary(s, a.out)
    t = s.get("traza_evento")
    if t:
        print("\n=== Recorrido de un evento por las 10 etapas ===")
        print(t["evento"])
        for step in t["pasos"]:
            print("  " + step)
        print("  [S9] recomendaciones Top-5 para su producto:")
        for r in t["recomendaciones"]:
            print(f"       {r['name']:<40} puntaje={r['puntaje']:<6} fuentes={r['fuentes']}")
    print("\n=== Resumen ===")
    for k in ("eventos_recibidos", "eventos_que_pasan_el_filtro", "segundos", "eventos_por_segundo",
              "S1", "S2", "S4", "S6", "S8"):
        print(f"{k}: {json.dumps(s[k], ensure_ascii=False, default=str)}")
    print(f"\nResumen completo en {a.out}")


if __name__ == "__main__":
    main()
