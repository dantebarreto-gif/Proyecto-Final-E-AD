"""Carga del dataset real REES46 y preparación de los archivos del pipeline.

Dataset: "eCommerce behavior data from multi category store" (REES46 Marketing
Platform), https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
Archivos mensuales 2019-Oct.csv, 2019-Nov.csv, ... (~285M eventos en total).

Uso (requiere haber descargado los CSV en data/raw/):
    kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store -p data/raw --unzip
    python scripts/prepare_data.py --source rees46 --events 10000000

DuckDB lee los CSV en streaming sin cargarlos completos en memoria y toma los
primeros N eventos en orden temporal. Luego se agrega el componente
sintético (enrich) y se guarda todo en Parquet.
"""
from __future__ import annotations

import glob
import os
from typing import Tuple

import pandas as pd

from .generator import GenConfig, catalog_from_events, enrich, generate_events

REES46_COLUMNS = ["event_time", "event_type", "product_id", "category_id", "category_code",
                  "brand", "price", "user_id", "user_session"]


def load_rees46(raw_dir: str, n_events: int) -> pd.DataFrame:
    import duckdb

    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv"))) + sorted(glob.glob(os.path.join(raw_dir, "*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"No hay CSV de REES46 en {raw_dir}")
    con = duckdb.connect()
    file_list = ", ".join("'" + f.replace("'", "''") + "'" for f in files)
    df = con.execute(f"""
        SELECT event_time::TIMESTAMPTZ AS event_time, event_type, product_id, category_id,
               category_code, brand, price, user_id,
               -- unas pocas filas no traen sesión: se agrupan por usuario para no mezclar órdenes
               COALESCE(user_session, 'sin-sesion-' || user_id::VARCHAR) AS user_session
        FROM read_csv_auto([{file_list}], header=true)
        WHERE price > 0
        ORDER BY event_time
        LIMIT {int(n_events)}
    """).df()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df


def prepare(source: str, n_events: int, out_dir: str, raw_dir: str = "data/raw",
            seed: int = 42) -> Tuple[str, str]:
    """Genera data/<nombre>/events.parquet y catalog.parquet. Devuelve sus rutas."""
    cfg = GenConfig(n_events=n_events, seed=seed)
    if source == "rees46":
        raw = load_rees46(raw_dir, n_events)
        catalog = catalog_from_events(raw)
    elif source == "synthetic":
        raw, catalog = generate_events(cfg)
    else:
        raise ValueError("source debe ser 'rees46' o 'synthetic'")
    events = enrich(raw, cfg)
    os.makedirs(out_dir, exist_ok=True)
    ev_path = os.path.join(out_dir, "events.parquet")
    cat_path = os.path.join(out_dir, "catalog.parquet")
    events.to_parquet(ev_path, index=False, row_group_size=500_000)
    catalog.to_parquet(cat_path, index=False)
    return ev_path, cat_path
