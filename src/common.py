"""Shared helpers: paths, plotting style, JSON I/O, geo maths and a light-touch pilot loader."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, safe on servers / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
RES = ROOT / "results"
REPORTS = ROOT / "reports"

RAW_FILE = DATA_RAW / "logistics_shipments_raw.csv"
CLEAN_FILE = DATA_PROC / "logistics_shipments_clean.csv"

SEED = 42

# Fixed-order categorical palette (blue, orange, aqua, yellow, magenta, green, violet, red).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROC, FIG, RES, REPORTS):
        d.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 170,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        }
    )


def save_fig(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_json(obj, name: str) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(name: str):
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised)."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# Network definition (shared by generator and optimisation) -------------------------------
WAREHOUSES = {  # name: (lat, lon)
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
}
DESTINATIONS = {  # hub: (lat, lon, zone, demand weight)
    "Ahmedabad": (23.0225, 72.5714, "West", 0.13),
    "Jaipur": (26.9124, 75.7873, "North", 0.11),
    "Lucknow": (26.8467, 80.9462, "North", 0.12),
    "Hyderabad": (17.3850, 78.4867, "South", 0.15),
    "Chennai": (13.0827, 80.2707, "South", 0.14),
    "Nagpur": (21.1458, 79.0882, "Central", 0.10),
    "Bhubaneswar": (20.2961, 85.8245, "East", 0.13),
    "Guwahati": (26.1445, 91.7362, "North-East", 0.12),
}
ROAD_FACTOR = 1.3  # road distance ~ 1.3 x great-circle distance

MODE_SPEED = {"Economy": 330.0, "Standard": 480.0, "Express": 750.0}  # km/day
MODE_RATE = {"Economy": 0.85, "Standard": 1.20, "Express": 1.95}  # INR per km base rate


CARRIER_CANON = {"bluedart lite": "BlueDart Lite", "swifttrans": "SwiftTrans", "roadrunner": "RoadRunner",
                 "citylink": "CityLink"}


def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """Parse the three date layouts found in the raw feed without guessing day/month order.

    ISO 'YYYY-MM-DD', Indian-style 'DD/MM/YYYY' and 'Mon DD, YYYY'. Anything else becomes NaT.
    (A blind pd.to_datetime(..., format='mixed') would read 05/10/2025 as May 10 - a silent error.)
    """
    s = s.astype("string").str.strip()
    layouts = [(r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"), (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
               (r"^[A-Za-z]{3} \d{1,2}, \d{4}$", "%b %d, %Y")]
    parts = []
    for pattern, fmt in layouts:
        m = s.str.match(pattern).fillna(False).astype(bool)
        if m.any():
            parts.append(pd.to_datetime(s[m], format=fmt))
    if not parts:
        return pd.Series(pd.NaT, index=s.index)
    return pd.concat(parts).reindex(s.index)


def load_pilot() -> pd.DataFrame:
    """Light-touch loader used in Week 1 (planning).

    Only removes exact duplicates and rows with impossible values so KPI definitions can be
    prototyped. The *full* cleaning pipeline is the subject of Week 2 (02_cleaning_pipeline.py).
    """
    df = pd.read_csv(RAW_FILE)
    df = df.drop_duplicates(subset="shipment_id").copy()
    for col in ("shipping_mode", "carrier", "origin_warehouse", "destination_hub", "product_category", "priority"):
        df[col] = df[col].astype("string").str.strip().str.title()
    df["carrier"] = df["carrier"].str.lower().map(CARRIER_CANON)
    df["order_date"] = parse_mixed_dates(df["order_date"])
    for col in ("distance_km", "weight_kg", "delivery_cost_inr", "planned_days", "actual_days", "customer_rating"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["actual_days"] > 0) & (df["actual_days"] < 30) & (df["distance_km"] > 0)]
    df = df.dropna(subset=["order_date", "carrier", "weight_kg", "distance_km", "actual_days", "planned_days"])
    df = df[df["weight_kg"] < 500]
    df["delayed"] = (df["actual_days"] > df["planned_days"]).astype(int)
    return df.reset_index(drop=True)
