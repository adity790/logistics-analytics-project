"""Week 1 - Strategic planning: scenario KPIs, lane clustering, inventory policy, cost regression preview.

Runs on a *pilot* sample (light cleaning only). Full cleaning is Week 2.
Outputs: figures/w1_*.png and results/week1_results.json
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import (INK, INK_2, MUTED, PALETTE, SEED, SEQ_BLUES, ensure_dirs, load_pilot, save_fig,
                    save_json, set_style)

ensure_dirs()
set_style()
df = load_pilot()
df["cost_per_km"] = df["delivery_cost_inr"].where(df["delivery_cost_inr"] > 0) / df["distance_km"]
df["cost_per_kg"] = df["delivery_cost_inr"].where(df["delivery_cost_inr"] > 0) / df["weight_kg"]
df["delay_days"] = (df["actual_days"] - df["planned_days"]).clip(lower=0)

# ------------------------------------------------------------------ 1. KPI definitions ---------
def kpi_block(g: pd.DataFrame) -> dict:
    late = g[g["delayed"] == 1]
    return {
        "shipments": int(len(g)),
        "on_time_rate_pct": round(100 * (1 - g["delayed"].mean()), 1),
        "avg_lead_time_days": round(g["actual_days"].mean(), 2),
        "avg_delay_days_when_late": round(late["delay_days"].mean(), 2) if len(late) else 0.0,
        "median_cost_per_km_inr": round(g["cost_per_km"].median(), 3),
        "median_cost_per_kg_inr": round(g["cost_per_kg"].median(), 2),
        "avg_customer_rating": round(g["customer_rating"].mean(), 2),
    }

overall = kpi_block(df)
by_zone = {k: kpi_block(g) for k, g in df.groupby("destination_zone")}
by_carrier = {k: kpi_block(g) for k, g in df.groupby("carrier")}
by_wh = {k: kpi_block(g) for k, g in df.groupby("origin_warehouse")}
by_mode = {k: kpi_block(g) for k, g in df.groupby("shipping_mode")}

# Figure 1: on-time rate by destination zone (horizontal bars, sorted, direct labels)
z = pd.DataFrame(by_zone).T.sort_values("on_time_rate_pct")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
colors = [PALETTE[1] if v == z["on_time_rate_pct"].min() else PALETTE[0] for v in z["on_time_rate_pct"]]
bars = ax.barh(z.index, z["on_time_rate_pct"], color=colors, height=0.55)
ax.axvline(overall["on_time_rate_pct"], color=MUTED, lw=1, ls="--")
ax.text(overall["on_time_rate_pct"] + 0.4, -0.62, f"network avg {overall['on_time_rate_pct']}%", color=INK_2, fontsize=8.5)
for b, v in zip(bars, z["on_time_rate_pct"]):
    ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9, color=INK)
ax.set_xlim(0, 100); ax.set_xlabel("On-time delivery rate (%)"); ax.grid(axis="y", visible=False)
ax.set_title("On-time delivery rate by destination zone (pilot sample)")
save_fig(fig, "w1_on_time_by_zone.png")

# ------------------------------------------------------------------ 2. Lane clustering ----------
lanes = (df.groupby(["origin_warehouse", "destination_hub"])
         .agg(shipments=("shipment_id", "count"), distance_km=("distance_km", "mean"),
              delay_rate=("delayed", "mean"), cost_per_km=("cost_per_km", "median"),
              lead_time=("actual_days", "mean"), avg_weight=("weight_kg", "mean"),
              weather=("weather_index", "mean"))
         .reset_index())
lanes["lane"] = lanes["origin_warehouse"] + " -> " + lanes["destination_hub"]
feat = ["distance_km", "delay_rate", "cost_per_km", "lead_time", "shipments"]
Xs = StandardScaler().fit_transform(lanes[feat])

sil, inertia = {}, {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xs)
    sil[k] = float(silhouette_score(Xs, km.labels_)); inertia[k] = float(km.inertia_)
# choose the best silhouette among k=3..6 (k=2 is usually too coarse to act on)
best_k = max(range(3, 7), key=lambda k: sil[k])
km = KMeans(n_clusters=best_k, n_init=50, random_state=SEED).fit(Xs)
lanes["cluster"] = km.labels_

prof = lanes.groupby("cluster")[feat].mean()
zs = (prof - lanes[feat].mean()) / lanes[feat].std()

def describe(row_z: pd.Series) -> str:
    bits = []
    if row_z["distance_km"] > 0.6: bits.append("long-haul")
    elif row_z["distance_km"] < -0.6: bits.append("short-haul")
    if row_z["delay_rate"] > 0.6: bits.append("high-delay")
    elif row_z["delay_rate"] < -0.6: bits.append("reliable")
    if row_z["cost_per_km"] > 0.6: bits.append("high-cost/km")
    elif row_z["cost_per_km"] < -0.6: bits.append("low-cost/km")
    if row_z["shipments"] > 0.6: bits.append("high-volume")
    elif row_z["shipments"] < -0.6: bits.append("low-volume")
    return ", ".join(bits) if bits else "mid-range / balanced"

names = {c: describe(zs.loc[c]) for c in zs.index}
lanes["segment"] = lanes["cluster"].map(names)

cluster_table = []
for c in prof.index:
    members = lanes[lanes["cluster"] == c]
    cluster_table.append({
        "cluster": int(c), "segment": names[c], "lanes": int(len(members)),
        "shipments": int(members["shipments"].sum()),
        "avg_distance_km": round(prof.loc[c, "distance_km"], 0),
        "delay_rate_pct": round(100 * prof.loc[c, "delay_rate"], 1),
        "cost_per_km_inr": round(prof.loc[c, "cost_per_km"], 3),
        "lead_time_days": round(prof.loc[c, "lead_time"], 2),
        "example_lanes": ", ".join(members.sort_values("shipments", ascending=False)["lane"].head(3)),
    })

# Figure 2: silhouette + elbow
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
ks = list(sil)
axes[0].plot(ks, [inertia[k] for k in ks], marker="o", color=PALETTE[0], lw=2, ms=6)
axes[0].set_title("Elbow (inertia)"); axes[0].set_xlabel("k"); axes[0].set_ylabel("within-cluster SS")
axes[1].plot(ks, [sil[k] for k in ks], marker="o", color=PALETTE[0], lw=2, ms=6)
axes[1].scatter([best_k], [sil[best_k]], s=110, color=PALETTE[1], zorder=5)
axes[1].annotate(f"chosen k={best_k}", (best_k, sil[best_k]), xytext=(8, 8), textcoords="offset points", fontsize=9, color=INK)
axes[1].set_title("Silhouette score"); axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette")
save_fig(fig, "w1_cluster_selection.png")

# Figure 3: PCA projection of lanes coloured by cluster
pca = PCA(n_components=2, random_state=SEED).fit(Xs)
P = pca.transform(Xs)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for c in sorted(lanes["cluster"].unique()):
    m = lanes["cluster"].values == c
    ax.scatter(P[m, 0], P[m, 1], s=70, color=PALETTE[c % len(PALETTE)], edgecolor="#fcfcfb", linewidth=1.5,
               label=f"C{c}: {names[c]}")
for i, row in lanes.iterrows():
    if row["delay_rate"] == lanes["delay_rate"].max() or row["shipments"] == lanes["shipments"].max():
        ax.annotate(row["lane"], (P[i, 0], P[i, 1]), xytext=(6, 5), textcoords="offset points", fontsize=8, color=INK_2)
ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.0f}% variance)")
ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.0f}% variance)")
ax.set_title("Delivery lanes segmented by K-Means (PCA view)")
ax.legend(fontsize=8, loc="best")
save_fig(fig, "w1_lane_clusters.png")

# ------------------------------------------------------------------ 3. Inventory policy (EOQ) --
daily = df.groupby(["origin_warehouse", "order_date"]).size().rename("units").reset_index()
S_ORDER, H_RATE, LEAD, LEAD_SD, Z95 = 4000.0, 0.22, 5.0, 1.2, 1.645
inv = []
for wh, g in daily.groupby("origin_warehouse"):
    mu, sd = g["units"].mean(), g["units"].std()
    annual = mu * 365
    unit_cost = 0.6 * df.loc[df["origin_warehouse"] == wh, "order_value_inr"].mean()
    eoq = np.sqrt(2 * annual * S_ORDER / (H_RATE * unit_cost))
    ss = Z95 * np.sqrt(LEAD * sd ** 2 + (mu ** 2) * LEAD_SD ** 2)
    inv.append({"warehouse": wh, "mean_daily_units": round(mu, 1), "sd_daily_units": round(sd, 1),
                "annual_units": int(annual), "unit_cost_inr": round(unit_cost, 0), "EOQ_units": int(round(eoq)),
                "safety_stock_units": int(round(ss)), "reorder_point_units": int(round(mu * LEAD + ss))})

# ------------------------------------------------------------------ 4. Regression preview ------
reg = df[(df["delivery_cost_inr"] > 0)].copy()
# Pilot-stage robustness: the raw feed still contains billing outliers (Week 2 treats them properly).
# For this preview only, trim the top 1% of costs so the exploratory fit is not dominated by them.
cost_cap = reg["delivery_cost_inr"].quantile(0.99)
reg_untrimmed = reg.copy()
r2_untrimmed = stats.linregress(reg_untrimmed["distance_km"], reg_untrimmed["delivery_cost_inr"]).rvalue ** 2
reg = reg[reg["delivery_cost_inr"] <= cost_cap].copy()
reg["wd"] = reg["weight_kg"] * reg["distance_km"]
lr = stats.linregress(reg["distance_km"], reg["delivery_cost_inr"])
from sklearn.linear_model import LinearRegression
mlr = LinearRegression().fit(reg[["distance_km", "weight_kg", "wd"]], reg["delivery_cost_inr"])
r2_multi = mlr.score(reg[["distance_km", "weight_kg", "wd"]], reg["delivery_cost_inr"])

fig, ax = plt.subplots(figsize=(7.2, 4.0))
s = reg.sample(2500, random_state=SEED)
ax.scatter(s["distance_km"], s["delivery_cost_inr"], s=9, alpha=0.35, color=PALETTE[0], edgecolor="none")
xs = np.linspace(reg["distance_km"].min(), reg["distance_km"].max(), 50)
ax.plot(xs, lr.intercept + lr.slope * xs, color=PALETTE[1], lw=2.2)
ax.text(0.03, 0.92, f"cost = {lr.intercept:,.0f} + {lr.slope:.3f} x km    (R² = {lr.rvalue**2:.2f})",
        transform=ax.transAxes, fontsize=9, color=INK)
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Delivery cost (INR)")
ax.set_title("Regression preview: delivery cost versus distance")
save_fig(fig, "w1_cost_regression_preview.png")

# ------------------------------------------------------------------ 5. Roadmap diagram ---------
steps = [("1  Collect", "shipments, carriers,\nweather, inventory"), ("2  Clean", "missing, outliers,\nduplicates, scaling"),
         ("3  Explore", "KPIs, distributions,\ncorrelations"), ("4  Model", "regression, clustering,\nclassification"),
         ("5  Optimise", "carrier mix, lane\nallocation (LP)"), ("6  Decide", "dashboards, KPI\ntargets, pilots")]
fig, ax = plt.subplots(figsize=(10, 2.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis("off"); ax.grid(False)
w, h, gap = 1.40, 1.35, 0.22
for i, (t, sub) in enumerate(steps):
    x = 0.12 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.45), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=SEQ_BLUES[min(i, 5)] if i < 3 else SEQ_BLUES[min(i, 5)], ec="none"))
    tc = INK if i < 3 else "#ffffff"
    ax.text(x + w / 2, 1.43, t, ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc)
    ax.text(x + w / 2, 0.92, sub, ha="center", va="center", fontsize=8, color=tc)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.12), (x + w + gap - 0.02, 1.12), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.4))
ax.text(5.0, 0.12, "Iterate: monitor KPIs after each decision and feed new data back into step 1", ha="center", fontsize=8.5, color=INK_2)
save_fig(fig, "w1_roadmap.png")

save_json({
    "overall": overall, "by_zone": by_zone, "by_carrier": by_carrier, "by_warehouse": by_wh, "by_mode": by_mode,
    "n_pilot_rows": int(len(df)), "n_lanes": int(len(lanes)), "silhouette": sil, "best_k": int(best_k),
    "cluster_table": cluster_table, "inventory": inv,
    "regression": {"slope_per_km": lr.slope, "intercept": lr.intercept, "r2_simple": lr.rvalue ** 2, "r2_multi": r2_multi,
                   "r2_untrimmed": r2_untrimmed, "cost_cap_inr": float(cost_cap)},
    "eoq_params": {"order_cost_inr": S_ORDER, "holding_rate": H_RATE, "lead_time_days": LEAD, "lead_time_sd": LEAD_SD, "z": Z95},
}, "week1_results.json")

print("overall:", overall)
print("best k:", best_k, {k: round(v, 3) for k, v in sil.items()})
for r in cluster_table: print(r)
print(pd.DataFrame(inv).to_string(index=False))
print("regression", round(lr.rvalue ** 2, 3), round(r2_multi, 3))
