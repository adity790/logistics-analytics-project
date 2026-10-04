"""Generate a realistic *simulated* logistics dataset (12,000 shipments, India network, 2025).

Why simulate? The public datasets cited in the reports (DataCo Smart Supply Chain, Olist, etc.) are
large, have licence/terms of their own, and are not shipped inside this repo. This generator produces a
dataset with the *same kinds of fields and the same kinds of data-quality problems* so the whole
pipeline runs end to end and is fully reproducible (fixed seed).

Output: data/raw/logistics_shipments_raw.csv  (deliberately dirty - cleaned in Week 2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (DESTINATIONS, MODE_RATE, MODE_SPEED, RAW_FILE, ROAD_FACTOR, SEED, WAREHOUSES,
                    ensure_dirs, haversine_km)

N = 12_000
rng = np.random.default_rng(SEED)


def build_clean_truth() -> pd.DataFrame:
    # --- order dates with seasonality (festive peak Oct-Dec) -------------------------------
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    month_w = np.array([0.8, 0.8, 0.85, 0.9, 0.9, 0.9, 0.95, 1.0, 1.05, 1.25, 1.45, 1.35])
    dow_w = np.array([1.1, 1.1, 1.05, 1.05, 1.0, 0.8, 0.7])
    w = np.array([month_w[d.month - 1] * dow_w[d.dayofweek] for d in days])
    order_date = rng.choice(days, size=N, p=w / w.sum())
    order_date = pd.to_datetime(order_date)

    # --- network ----------------------------------------------------------------------------
    hubs = list(DESTINATIONS)
    hub_p = np.array([DESTINATIONS[h][3] for h in hubs])
    dest = rng.choice(hubs, size=N, p=hub_p / hub_p.sum())
    wh_names = list(WAREHOUSES)
    base_share = np.array([0.30, 0.22, 0.20, 0.18, 0.10])

    origin = np.empty(N, dtype=object)
    dist = np.empty(N)
    for i, h in enumerate(dest):
        hlat, hlon = DESTINATIONS[h][0], DESTINATIONS[h][1]
        d_all = np.array([haversine_km(WAREHOUSES[w][0], WAREHOUSES[w][1], hlat, hlon) for w in wh_names]) * ROAD_FACTOR
        # dispatchers prefer nearer warehouses but stock availability breaks the rule often
        score = base_share * np.exp(-d_all / 1500.0)
        p = score / score.sum()
        k = rng.choice(len(wh_names), p=p)
        origin[i] = wh_names[k]
        dist[i] = d_all[k] * rng.normal(1.0, 0.04)
    dist = np.clip(dist, 40, None)

    zone = np.array([DESTINATIONS[h][2] for h in dest])

    # --- shipment attributes ----------------------------------------------------------------
    category = rng.choice(["Electronics", "Apparel", "Grocery", "Furniture", "Pharma", "Books"], size=N,
                          p=[0.22, 0.24, 0.20, 0.10, 0.12, 0.12])
    wt_scale = {"Electronics": 6, "Apparel": 3, "Grocery": 12, "Furniture": 38, "Pharma": 2.5, "Books": 4}
    weight = np.array([rng.gamma(2.2, wt_scale[c] / 2.2) for c in category])
    weight = np.clip(weight, 0.3, 160)
    volume = weight / rng.uniform(80, 220, size=N)  # m3 (density kg/m3 varies)
    val_scale = {"Electronics": 24000, "Apparel": 2800, "Grocery": 1800, "Furniture": 16000, "Pharma": 3500, "Books": 900}
    order_value = np.array([rng.lognormal(np.log(val_scale[c]), 0.55) for c in category]).round(0)

    mode = rng.choice(["Economy", "Standard", "Express"], size=N, p=[0.30, 0.50, 0.20])
    priority = np.where(mode == "Express", rng.choice(["High", "Medium"], size=N, p=[0.7, 0.3]),
                        rng.choice(["High", "Medium", "Low"], size=N, p=[0.12, 0.48, 0.40]))
    carrier = rng.choice(["BlueDart Lite", "SwiftTrans", "RoadRunner", "CityLink"], size=N, p=[0.28, 0.32, 0.22, 0.18])
    carrier_delay = {"BlueDart Lite": 0.10, "SwiftTrans": 0.25, "RoadRunner": 0.55, "CityLink": 0.35}
    carrier_cost = {"BlueDart Lite": 1.12, "SwiftTrans": 1.00, "RoadRunner": 0.88, "CityLink": 0.94}

    month = order_date.month.values
    peak = np.isin(month, [10, 11, 12]).astype(float)
    monsoon = np.isin(month, [6, 7, 8, 9]).astype(float)
    weather = np.clip(rng.beta(2, 5, size=N) + 0.35 * monsoon * rng.beta(2, 3, size=N), 0, 1)  # 0 calm .. 1 severe
    traffic = np.clip(rng.beta(2.2, 3.5, size=N) + 0.10 * peak, 0, 1)
    ne_penalty = (zone == "North-East").astype(float) * 0.9  # hill terrain / limited road network

    speed = np.array([MODE_SPEED[m] for m in mode])
    rate = np.array([MODE_RATE[m] for m in mode])
    transit = dist / speed
    handling = 0.6 + 0.25 * (weight > 40)
    planned = np.ceil(transit + handling + 0.15).astype(int)

    delay_core = (np.array([carrier_delay[c] for c in carrier]) + 1.4 * weather + 1.0 * traffic + 0.5 * peak
                  + ne_penalty + 0.004 * np.maximum(weight - 30, 0))
    noise = rng.lognormal(mean=-1.1, sigma=0.7, size=N)
    actual = transit + handling + (delay_core * noise) - 0.10 + rng.normal(0, 0.18, size=N)
    actual = np.clip(actual, 0.5, None).round(1)

    cost = (dist * rate * 0.42 + weight * dist * 0.0016 + 45) * np.array([carrier_cost[c] for c in carrier])
    cost = cost * (1 + 0.14 * peak) * rng.normal(1.0, 0.06, size=N)
    cost = np.clip(cost, 35, None).round(2)

    late_days = np.maximum(actual - planned, 0)
    rating = np.clip(4.7 - 0.45 * late_days - 0.6 * (late_days > 2) + rng.normal(0, 0.45, size=N), 1, 5).round(0)

    df = pd.DataFrame(
        {
            "shipment_id": [f"SH{100000 + i}" for i in range(N)],
            "order_date": order_date,
            "origin_warehouse": origin,
            "destination_hub": dest,
            "destination_zone": zone,
            "product_category": category,
            "shipping_mode": mode,
            "priority": priority,
            "carrier": carrier,
            "distance_km": dist.round(1),
            "weight_kg": weight.round(2),
            "volume_m3": volume.round(4),
            "order_value_inr": order_value,
            "weather_index": weather.round(3),
            "traffic_index": traffic.round(3),
            "planned_days": planned,
            "actual_days": actual,
            "delivery_cost_inr": cost,
            "customer_rating": rating,
        }
    )
    return df.sort_values("order_date").reset_index(drop=True)


def make_dirty(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic data-quality problems so Week 2 has genuine work to do."""
    d = df.copy()
    n = len(d)

    def pick(frac):
        return rng.choice(n, size=int(n * frac), replace=False)

    # 1. missing values (different mechanisms)
    d.loc[pick(0.050), "weight_kg"] = np.nan
    d.loc[pick(0.040), "distance_km"] = np.nan
    d.loc[pick(0.030), "carrier"] = np.nan
    d.loc[pick(0.025), "weather_index"] = np.nan
    d.loc[pick(0.030), "traffic_index"] = np.nan
    d.loc[pick(0.035), "customer_rating"] = np.nan
    d.loc[pick(0.010), "volume_m3"] = np.nan
    # 2. outliers / impossible values
    idx = pick(0.015); d.loc[idx, "weight_kg"] = d.loc[idx, "weight_kg"] * rng.uniform(12, 40, size=len(idx))
    idx = pick(0.010); d.loc[idx, "delivery_cost_inr"] = d.loc[idx, "delivery_cost_inr"] * rng.uniform(8, 25, size=len(idx))
    idx = pick(0.006); d.loc[idx, "actual_days"] = rng.uniform(35, 90, size=len(idx)).round(1)
    idx = pick(0.004); d.loc[idx, "actual_days"] = -rng.uniform(1, 5, size=len(idx)).round(1)
    idx = pick(0.003); d.loc[idx, "distance_km"] = 0.0
    idx = pick(0.004); d.loc[idx, "delivery_cost_inr"] = -d.loc[idx, "delivery_cost_inr"]
    # 3. inconsistent categorical labels
    d["shipping_mode"] = d["shipping_mode"].astype(object)
    idx = pick(0.08); d.loc[idx, "shipping_mode"] = d.loc[idx, "shipping_mode"].str.upper()
    idx = pick(0.06); d.loc[idx, "shipping_mode"] = d.loc[idx, "shipping_mode"].str.lower() + " "
    d["carrier"] = d["carrier"].astype(object)
    idx = pick(0.05)
    d.loc[idx, "carrier"] = d.loc[idx, "carrier"].astype(str).str.replace("nan", "").str.lower()
    d.loc[d["carrier"] == "", "carrier"] = np.nan
    idx = pick(0.04); d.loc[idx, "origin_warehouse"] = d.loc[idx, "origin_warehouse"].str.upper()
    # 4. mixed date formats
    d["order_date"] = d["order_date"].dt.strftime("%Y-%m-%d")
    idx = pick(0.10); d.loc[idx, "order_date"] = pd.to_datetime(d.loc[idx, "order_date"]).dt.strftime("%d/%m/%Y")
    idx = pick(0.04); d.loc[idx, "order_date"] = pd.to_datetime(d.loc[idx, "order_date"], dayfirst=True, format="mixed").dt.strftime("%b %d, %Y")
    # 5. duplicated records (system retries)
    dups = d.sample(160, random_state=SEED)
    d = pd.concat([d, dups], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return d


if __name__ == "__main__":
    ensure_dirs()
    truth = build_clean_truth()
    raw = make_dirty(truth)
    raw.to_csv(RAW_FILE, index=False)
    print(f"Clean ground-truth rows: {len(truth):,} | raw rows written: {len(raw):,}")
    print(f"Saved -> {RAW_FILE}")
    print(raw.head(3).to_string())
