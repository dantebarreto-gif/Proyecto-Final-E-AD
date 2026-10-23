"""Generador sintético de eventos de e-commerce con el esquema de REES46.

REES46 ("eCommerce behavior data from multi category store", Kaggle) trae
columnas event_time, event_type, product_id, category_id, category_code,
brand, price, user_id, user_session, pero no trae prioridades, fraude,
duplicados ni dispositivos. Este módulo:

1. `generate_events` simula un stream con la misma estructura que REES46
   (embudo view -> cart -> purchase, popularidad Zipf, sesiones, categorías
   agrupadas en "clusters de afinidad" que generan co-compra), para poder
   correr el pipeline y los experimentos sin descargar el dataset.
2. `enrich` agrega a cualquier tabla con esquema REES46 (real o sintética)
   el componente de stream que pide el proyecto: event_id, ts, dispositivo,
   IP, tarjeta, tipo de envío, usuario premium, consultas de búsqueda,
   ráfagas de fraude y duplicados, todo con semilla fija.

Las etiquetas is_fraud e is_duplicate son la verdad conocida para medir la
precisión de las etapas 1 y 6; el pipeline nunca las lee para decidir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# (category_code, cluster de afinidad, precio mediano USD, marcas)
CATEGORIES: List[Tuple[str, int, float, List[str]]] = [
    ("electronics.smartphone", 0, 320, ["samsung", "apple", "xiaomi", "huawei", "oppo", "nokia"]),
    ("electronics.audio.headphone", 0, 60, ["sony", "jbl", "apple", "samsung", "xiaomi"]),
    ("electronics.clocks", 0, 150, ["apple", "samsung", "garmin", "xiaomi", "casio"]),
    ("electronics.tablet", 0, 280, ["apple", "samsung", "lenovo", "huawei"]),
    ("computers.notebook", 1, 650, ["lenovo", "hp", "asus", "acer", "apple", "dell"]),
    ("computers.peripherals.mouse", 1, 20, ["logitech", "razer", "hp", "a4tech"]),
    ("computers.peripherals.keyboard", 1, 35, ["logitech", "razer", "hp", "a4tech"]),
    ("computers.components.videocards", 1, 400, ["msi", "asus", "gigabyte", "palit"]),
    ("computers.desktop", 1, 700, ["hp", "dell", "lenovo", "asus"]),
    ("electronics.video.tv", 2, 450, ["samsung", "lg", "sony", "philips", "tcl"]),
    ("electronics.audio.acoustic", 2, 120, ["jbl", "sony", "bose", "yamaha"]),
    ("electronics.audio.subwoofer", 2, 180, ["jbl", "sony", "yamaha", "pioneer"]),
    ("appliances.kitchen.refrigerators", 3, 600, ["lg", "samsung", "bosch", "indesit", "beko"]),
    ("appliances.kitchen.washer", 3, 380, ["lg", "samsung", "bosch", "indesit", "beko"]),
    ("appliances.kitchen.microwave", 3, 90, ["samsung", "lg", "panasonic", "midea"]),
    ("appliances.kitchen.oven", 3, 300, ["bosch", "electrolux", "gorenje", "beko"]),
    ("appliances.environment.vacuum", 4, 160, ["dyson", "samsung", "philips", "xiaomi", "bosch"]),
    ("appliances.environment.air_conditioner", 4, 420, ["lg", "samsung", "midea", "gree"]),
    ("appliances.iron", 4, 45, ["philips", "tefal", "bosch", "braun"]),
    ("appliances.personal.hair_cutter", 5, 40, ["philips", "braun", "remington", "panasonic"]),
    ("beauty.skin.care", 5, 25, ["nivea", "loreal", "garnier", "vichy"]),
    ("beauty.fragrance", 5, 55, ["dior", "chanel", "versace", "hugo"]),
    ("apparel.shoes", 6, 70, ["nike", "adidas", "puma", "reebok", "skechers"]),
    ("apparel.shoes.keds", 6, 55, ["nike", "adidas", "puma", "vans", "converse"]),
    ("apparel.costume", 6, 40, ["adidas", "nike", "puma", "zara"]),
    ("sport.bicycle", 7, 350, ["trek", "giant", "merida", "stels"]),
    ("sport.tennis", 7, 60, ["wilson", "head", "babolat"]),
    ("sport.trainer", 7, 250, ["kettler", "torneo", "nordictrack"]),
    ("kids.toys", 8, 25, ["lego", "hasbro", "mattel", "playmobil"]),
    ("kids.carriage", 8, 280, ["chicco", "cybex", "peg-perego"]),
    ("kids.dolls", 8, 30, ["mattel", "hasbro", "barbie"]),
    ("furniture.living_room.sofa", 9, 520, ["ikea", "hoff", "askona"]),
    ("furniture.bedroom.bed", 9, 450, ["ikea", "askona", "ormatek"]),
    ("furniture.kitchen.table", 9, 160, ["ikea", "hoff"]),
    ("auto.accessories.player", 10, 90, ["pioneer", "sony", "kenwood", "jvc"]),
    ("auto.accessories.videoregister", 10, 70, ["xiaomi", "neoline", "playme"]),
    ("construction.tools.drill", 11, 85, ["bosch", "makita", "dewalt", "metabo"]),
    ("construction.tools.saw", 11, 110, ["bosch", "makita", "dewalt"]),
    ("country_yard.lawn_mower", 11, 260, ["bosch", "husqvarna", "makita"]),
]
N_CLUSTERS = 12
SHIPPING = np.array(["", "express", "standard", "scheduled"])
EVENT_TYPES = np.array(["view", "cart", "remove_from_cart", "purchase"])
START = pd.Timestamp("2019-10-01", tz="UTC")


@dataclass
class GenConfig:
    n_events: int = 1_000_000
    n_products: int = 0          # 0 = automático según n_events
    n_users: int = 0
    seed: int = 42
    days: int = 31
    buyer_session_rate: float = 0.10
    duplicate_rate: float = 0.015
    fraud_burst_rate: float = 1e-4   # ráfagas por evento legítimo
    known_fraud_share: float = 0.5   # fracción de tarjetas fraudulentas ya en la lista negra


def build_catalog(n_products: int, rng: np.random.Generator) -> pd.DataFrame:
    n_cat = len(CATEGORIES)
    weights = rng.dirichlet(np.full(n_cat, 2.0))
    cat_of = rng.choice(n_cat, size=n_products, p=weights)
    brand = np.empty(n_products, dtype=object)
    price = np.empty(n_products)
    for c, (code, cluster, med, brands) in enumerate(CATEGORIES):
        m = cat_of == c
        k = int(m.sum())
        bw = 1 / np.arange(1, len(brands) + 1)
        brand[m] = np.array(brands)[rng.choice(len(brands), size=k, p=bw / bw.sum())]
        price[m] = np.round(rng.lognormal(np.log(med), 0.45, size=k), 2)
    model = rng.integers(100, 9999, size=n_products)
    df = pd.DataFrame({
        "product_id": 1_000_000 + np.arange(n_products) * 7 + rng.integers(0, 7, n_products),
        "category_id": 2_053_013_550_000_000_000 + cat_of.astype(np.int64) * 1_000_003,
        "category_code": np.array([c[0] for c in CATEGORIES])[cat_of],
        "brand": brand,
        "price": np.maximum(price, 0.99),
        "cluster": np.array([c[1] for c in CATEGORIES])[cat_of],
    })
    leaf = df["category_code"].str.split(".").str[-1].str.replace("_", " ")
    df["name"] = df["brand"] + " " + leaf + " " + model.astype(str)
    # popularidad Zipf dentro del catálogo (pocos productos concentran las vistas)
    df["popularity"] = 1.0 / np.power(rng.permutation(n_products) + 1, 0.9)
    return df


def generate_events(cfg: GenConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (eventos con esquema REES46, catálogo de productos)."""
    rng = np.random.default_rng(cfg.seed)
    n_products = cfg.n_products or int(min(200_000, max(2_000, cfg.n_events // 50)))
    n_users = cfg.n_users or int(max(1_000, cfg.n_events // 20))
    catalog = build_catalog(n_products, rng)

    # productos de cada cluster con probabilidad proporcional a su popularidad
    cl_products: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for c in range(N_CLUSTERS):
        idx = np.where(catalog["cluster"].to_numpy() == c)[0]
        p = catalog["popularity"].to_numpy()[idx]
        cl_products[c] = (idx, p / p.sum())

    # sesiones: largo geométrico, cada una enfocada en un cluster (afinidad)
    mean_len = 8
    n_sessions = int(cfg.n_events / mean_len * 1.05) + 1
    lens = rng.geometric(1 / mean_len, size=n_sessions)
    lens = lens[np.cumsum(lens) <= cfg.n_events]
    n_sessions = lens.size
    user_pref = rng.integers(0, N_CLUSTERS, size=n_users)
    user_w = 1.0 / np.power(np.arange(1, n_users + 1), 0.6)
    s_user = rng.choice(n_users, size=n_sessions, p=user_w / user_w.sum())
    # 75 % de las sesiones en el cluster preferido del usuario
    s_cluster = np.where(rng.random(n_sessions) < 0.75, user_pref[s_user],
                         rng.integers(0, N_CLUSTERS, n_sessions))
    s_buyer = rng.random(n_sessions) < cfg.buyer_session_rate
    s_start = rng.random(n_sessions) * cfg.days * 86400

    ev_session = np.repeat(np.arange(n_sessions), lens)
    n = ev_session.size
    pos_in_session = np.arange(n) - np.repeat(np.cumsum(lens) - lens, lens)
    gaps = rng.exponential(35.0, size=n)
    gaps[pos_in_session == 0] = 0
    # tiempo = inicio de sesión + suma acumulada de gaps dentro de la sesión
    cum = np.cumsum(gaps)
    cum -= np.repeat(cum[np.cumsum(lens) - lens], lens)
    ts = s_start[ev_session] + cum

    product = np.empty(n, dtype=np.int64)
    ev_cluster = s_cluster[ev_session]
    for c in range(N_CLUSTERS):
        m = ev_cluster == c
        idx, p = cl_products[c]
        product[m] = idx[rng.choice(idx.size, size=int(m.sum()), p=p)]

    # embudo: las sesiones compradoras convierten mucho más
    buyer = s_buyer[ev_session]
    u = rng.random(n)
    p_purchase = np.where(buyer, 0.22, 0.003)
    p_cart = np.where(buyer, 0.18, 0.04)
    etype = np.zeros(n, dtype=np.int8)                     # view
    etype[u < p_purchase + p_cart + 0.01] = 2              # remove_from_cart
    etype[u < p_purchase + p_cart] = 1                     # cart
    etype[u < p_purchase] = 3                              # purchase

    sessions_uuid = _uuid_like(rng, n_sessions)
    events = pd.DataFrame({
        "event_time": START + pd.to_timedelta(np.round(ts).astype(np.int64), unit="s"),
        "event_type": EVENT_TYPES[etype],
        "product_id": catalog["product_id"].to_numpy()[product],
        "category_id": catalog["category_id"].to_numpy()[product],
        "category_code": catalog["category_code"].to_numpy()[product],
        "brand": catalog["brand"].to_numpy()[product],
        "price": catalog["price"].to_numpy()[product],
        "user_id": 512_000_000 + s_user[ev_session].astype(np.int64),
        "user_session": sessions_uuid[ev_session],
    })
    events = events.sort_values("event_time", kind="stable").reset_index(drop=True)
    return events, catalog


def _uuid_like(rng: np.random.Generator, n: int) -> np.ndarray:
    hi = rng.integers(0, 2 ** 63, size=n, dtype=np.int64)
    lo = rng.integers(0, 2 ** 63, size=n, dtype=np.int64)
    return np.array([f"{a:016x}-{b:016x}" for a, b in zip(hi, lo)])


def catalog_from_events(events: pd.DataFrame) -> pd.DataFrame:
    """Catálogo derivado de un dataset real (REES46): un registro por producto."""
    g = events.groupby("product_id", sort=True)
    cat = g.agg(category_id=("category_id", "first"), category_code=("category_code", "first"),
                brand=("brand", "first"), price=("price", "median"), views=("event_type", "size")).reset_index()
    cat["category_code"] = cat["category_code"].fillna("unknown")
    cat["brand"] = cat["brand"].fillna("generic")
    leaf = cat["category_code"].str.split(".").str[-1].str.replace("_", " ")
    cat["name"] = cat["brand"] + " " + leaf + " " + (cat["product_id"] % 10000).astype(str)
    cat["popularity"] = cat["views"] / cat["views"].max()
    cat["cluster"] = cat["category_code"].str.split(".").str[0].astype("category").cat.codes
    return cat.drop(columns="views")


def enrich(events: pd.DataFrame, cfg: GenConfig) -> pd.DataFrame:
    """Agrega el componente de stream sintético a eventos con esquema REES46."""
    rng = np.random.default_rng(cfg.seed + 1)
    ev = events.reset_index(drop=True).copy()
    n = len(ev)
    ev["ts"] = (ev["event_time"] - START).dt.total_seconds().astype(np.float64)
    users = ev["user_id"].to_numpy()
    uniq_users, uinv = np.unique(users, return_inverse=True)
    nu = uniq_users.size
    # cada usuario tiene 1-2 dispositivos, una IP y una tarjeta; 6 % son premium
    dev_a = rng.integers(10 ** 9, 2 * 10 ** 9, nu)
    dev_b = rng.integers(10 ** 9, 2 * 10 ** 9, nu)
    two = rng.random(nu) < 0.3
    use_b = two[uinv] & (rng.random(n) < 0.4)
    ev["device_id"] = np.where(use_b, dev_b[uinv], dev_a[uinv])
    ev["ip"] = rng.integers(0, 2 ** 32, nu, dtype=np.int64)[uinv]
    ev["card_id"] = rng.integers(4 * 10 ** 15, 5 * 10 ** 15, nu, dtype=np.int64)[uinv]
    ev["premium"] = (rng.random(nu) < 0.06)[uinv]
    is_purchase = ev["event_type"].to_numpy() == "purchase"
    ship = np.zeros(n, dtype=np.int8)
    ship[is_purchase] = rng.choice([1, 2, 3], size=int(is_purchase.sum()), p=[0.15, 0.7, 0.15])
    ev["shipping"] = SHIPPING[ship]
    # 30 % de las vistas llegan desde una búsqueda (alimenta el autocompletado)
    is_view = ev["event_type"].to_numpy() == "view"
    from_search = is_view & (rng.random(n) < 0.3)
    leaf = ev["category_code"].fillna("").str.split(".").str[-1].str.replace("_", " ")
    brand = ev["brand"].fillna("")
    kind = rng.integers(0, 3, n)
    q = np.where(kind == 0, brand, np.where(kind == 1, leaf, brand + " " + leaf))
    ev["search_query"] = np.where(from_search, q, "")
    ev["is_fraud"] = False
    ev["is_duplicate"] = False

    ev = pd.concat([ev, _fraud_bursts(ev, cfg, rng)], ignore_index=True)
    ev = ev.sort_values("ts", kind="stable").reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(len(ev), dtype=np.int64))

    # duplicados: reenvío exacto del mismo evento (mismo event_id) segundos después
    nd = int(len(ev) * cfg.duplicate_rate)
    dup_idx = rng.choice(len(ev), size=nd, replace=False)
    dups = ev.iloc[dup_idx].copy()
    dups["ts"] = dups["ts"] + rng.exponential(3.0, nd)
    dups["is_duplicate"] = True
    ev = pd.concat([ev, dups], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    return ev


def _fraud_bursts(ev: pd.DataFrame, cfg: GenConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Ráfagas de intentos de compra desde dispositivos/tarjetas fraudulentas."""
    n_bursts = max(3, int(len(ev) * cfg.fraud_burst_rate))
    t_max = float(ev["ts"].max())
    rows = []
    sample = ev[ev["event_type"].isin(["view", "cart"])]
    picks = sample.iloc[rng.integers(0, len(sample), n_bursts)]
    for b, (_, base) in enumerate(picks.iterrows()):
        size = int(rng.integers(25, 120))
        t0 = rng.random() * t_max
        r = pd.DataFrame([base] * size)
        r["ts"] = t0 + np.sort(rng.random(size)) * 300  # todo en < 5 minutos
        r["event_type"] = np.where(rng.random(size) < 0.7, "purchase", "cart")
        r["device_id"] = 3 * 10 ** 9 + b
        r["card_id"] = 6 * 10 ** 15 + b
        r["ip"] = 3_000_000_000 + b
        r["user_id"] = 900_000_000 + b
        r["premium"] = False
        r["shipping"] = np.where(r["event_type"] == "purchase", "express", "")
        r["search_query"] = ""
        r["is_fraud"] = True
        r["is_duplicate"] = False
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    out["event_time"] = START + pd.to_timedelta(np.round(out["ts"]).astype(np.int64), unit="s")
    return out


def blacklist(events: pd.DataFrame, share: float, seed: int = 0) -> Dict[str, np.ndarray]:
    """Lista negra conocida (carga el Bloom Filter): una fracción de los fraudes."""
    rng = np.random.default_rng(seed)
    fr = events[events["is_fraud"]]
    cards = np.unique(fr["card_id"].to_numpy())
    devices = np.unique(fr["device_id"].to_numpy())
    keep_c = cards[rng.random(cards.size) < share]
    keep_d = devices[np.isin(devices, fr[fr["card_id"].isin(keep_c)]["device_id"].to_numpy())]
    return {"card_id": keep_c, "device_id": keep_d}
