"""Week 3 - Exploratory data analysis and visualisation on the cleaned logistics data.

Input : data/processed/logistics_shipments_clean.csv
Output: figures/w3_*.png and results/week3_results.json
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from common import (CLEAN_FILE, INK, INK_2, MUTED, PALETTE, SEQ_BLUES, ensure_dirs, save_fig, save_json, set_style)

ensure_dirs()
set_style()
df = pd.read_csv(CLEAN_FILE, parse_dates=["order_date"])
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
MODE_COL = {"Economy": PALETTE[0], "Standard": PALETTE[1], "Express": PALETTE[2]}
insights: dict = {}

# ------------------------------------------------------------------ 1. Descriptive statistics --
cols = ["distance_km", "weight_kg", "order_value_inr", "weather_index", "traffic_index", "planned_days",
        "actual_days", "delay_days", "delivery_cost_inr", "cost_per_km", "customer_rating"]
desc = df[cols].describe().T[["mean", "50%", "std", "min", "max"]].rename(columns={"50%": "median"})
desc["skew"] = df[cols].skew()
desc["IQR"] = df[cols].quantile(0.75) - df[cols].quantile(0.25)
desc = desc.round(3)
overall = {
    "shipments": int(len(df)),
    "on_time_pct": round(100 * (1 - df["delayed"].mean()), 1),
    "mean_actual_days": round(df["actual_days"].mean(), 2),
    "median_actual_days": round(df["actual_days"].median(), 2),
    "mean_cost_inr": round(df["delivery_cost_inr"].mean(), 1),
    "median_cost_per_km": round(df["cost_per_km"].median(), 3),
    "mean_rating": round(df["customer_rating"].mean(), 2),
    "total_cost_inr": float(df["delivery_cost_inr"].sum().round(0)),
}

# ------------------------------------------------------------------ 2. Distributions -----------
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))
ax = axes[0, 0]
ax.hist(df["actual_days"], bins=40, color=PALETTE[0], alpha=0.9)
ax.axvline(df["actual_days"].median(), color=PALETTE[1], lw=2)
ax.text(df["actual_days"].median() + 0.2, ax.get_ylim()[1] * 0.9, f"median {df['actual_days'].median():.1f} d", color=INK, fontsize=9)
ax.set_title("Delivery time (days)"); ax.set_xlabel("actual_days"); ax.set_ylabel("Shipments")
ax = axes[0, 1]
ax.hist(df["cost_per_km"].clip(upper=df["cost_per_km"].quantile(0.99)), bins=40, color=PALETTE[0], alpha=0.9)
ax.axvline(df["cost_per_km"].median(), color=PALETTE[1], lw=2)
ax.set_title("Cost per km (INR)"); ax.set_xlabel("cost_per_km (99th pct cap)")
ax = axes[1, 0]
ax.hist(np.log10(df["weight_kg"]), bins=40, color=PALETTE[0], alpha=0.9)
ticks = [0.5, 1, 2, 5, 10, 25, 50, 100]
ax.set_xticks(np.log10(ticks)); ax.set_xticklabels(ticks)
ax.set_title("Parcel weight (kg, log scale)"); ax.set_xlabel("weight_kg"); ax.set_ylabel("Shipments")
ax = axes[1, 1]
order = ["Economy", "Standard", "Express"]
data = [df.loc[df["shipping_mode"] == m, "actual_days"] for m in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.5, showfliers=False,
                medianprops=dict(color=INK, lw=1.6))
for patch, m in zip(bp["boxes"], order):
    patch.set(facecolor=MODE_COL[m], alpha=0.85, edgecolor="none")
ax.set_title("Delivery time by shipping mode"); ax.set_ylabel("actual_days")
save_fig(fig, "w3_distributions.png")

# ------------------------------------------------------------------ 3. Correlation -------------
corr_cols = ["distance_km", "weight_kg", "order_value_inr", "weather_index", "traffic_index", "is_peak_season",
             "planned_days", "actual_days", "delay_days", "delivery_cost_inr", "customer_rating"]
corr = df[corr_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7.6, 6.2))
im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8.5)
ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, fontsize=8.5)
ax.grid(False)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2, color="#ffffff" if abs(v) > 0.65 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03); cb.set_label("Spearman correlation", fontsize=9)
ax.set_title("Spearman correlation of key logistics variables")
save_fig(fig, "w3_correlation.png")

tri = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().reset_index()
tri.columns = ["a", "b", "rho"]
tri["abs"] = tri["rho"].abs()
top_pairs = tri.sort_values("abs", ascending=False).head(8)[["a", "b", "rho"]].round(2).to_dict("records")
delay_corr = corr["delay_days"].drop("delay_days").sort_values(key=np.abs, ascending=False).round(2).head(5).to_dict()
actual_corr = corr["actual_days"].drop("actual_days").sort_values(key=np.abs, ascending=False).round(2).head(5).to_dict()

# ------------------------------------------------------------------ 4. Monthly trend (two panels) --
monthly = df.groupby("order_month").agg(shipments=("shipment_id", "count"), delay_rate=("delayed", "mean"),
                                        cost=("cost_per_km", "median")).reset_index()
monthly["delay_rate"] *= 100
mnames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, axes = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True)
cols_bar = [PALETTE[1] if m in (10, 11, 12) else PALETTE[0] for m in monthly["order_month"]]
axes[0].bar(monthly["order_month"], monthly["shipments"], color=cols_bar, width=0.62)
axes[0].set_ylabel("Shipments"); axes[0].set_title("Monthly volume (orange = festive peak Oct-Dec)")
axes[1].plot(monthly["order_month"], monthly["delay_rate"], color=PALETTE[0], lw=2.2, marker="o", ms=6)
for m, v in zip(monthly["order_month"], monthly["delay_rate"]):
    if m in (1, 6, 10, 11, 12):
        axes[1].annotate(f"{v:.0f}%", (m, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
axes[1].set_ylabel("Late deliveries (%)"); axes[1].set_title("Share of shipments delivered late")
axes[1].set_xticks(range(1, 13)); axes[1].set_xticklabels(mnames); axes[1].set_ylim(0, monthly["delay_rate"].max() * 1.25)
save_fig(fig, "w3_monthly_trend.png")

peak = df[df["is_peak_season"] == 1]["delayed"]; off = df[df["is_peak_season"] == 0]["delayed"]
p1, p2, n1, n2 = peak.mean(), off.mean(), len(peak), len(off)
pp = (peak.sum() + off.sum()) / (n1 + n2)
zstat = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
insights["peak_season"] = {"peak_delay_pct": round(100 * p1, 1), "offpeak_delay_pct": round(100 * p2, 1),
                           "z": round(float(zstat), 2), "p_value": float(2 * (1 - stats.norm.cdf(abs(zstat)))),
                           "volume_uplift_pct": round(100 * (len(df[df.is_peak_season == 1]) / 3) / (len(df[df.is_peak_season == 0]) / 9) - 100, 1)}

# ------------------------------------------------------------------ 5. Carrier x zone heatmap ----
hm = df.pivot_table(index="carrier", columns="destination_zone", values="delayed", aggfunc="mean") * 100
hm = hm.loc[hm.mean(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(7.6, 3.8))
im = ax.imshow(hm.values, cmap=SEQ, aspect="auto")
ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns, fontsize=9)
ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index, fontsize=9); ax.grid(False)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="#ffffff" if v > hm.values.max() * 0.65 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03); cb.set_label("Late deliveries (%)", fontsize=9)
ax.set_title("Late-delivery rate by carrier and destination zone")
save_fig(fig, "w3_delay_heatmap.png")

car = df.groupby("carrier").agg(n=("shipment_id", "count"), delay=("delayed", "mean"), cpk=("cost_per_km", "median"),
                                 days=("actual_days", "mean"), rating=("customer_rating", "mean"))
chi_car = stats.chi2_contingency(pd.crosstab(df["carrier"], df["delayed"]))
insights["carrier"] = {"table": {k: {"shipments": int(r.n), "late_pct": round(100 * r.delay, 1), "median_cost_per_km": round(r.cpk, 3),
                                     "mean_days": round(r.days, 2), "mean_rating": round(r.rating, 2)} for k, r in car.iterrows()},
                       "chi2_p_value": float(chi_car[1])}
zone = df.groupby("destination_zone").agg(n=("shipment_id", "count"), delay=("delayed", "mean"), days=("actual_days", "mean"),
                                           cpk=("cost_per_km", "median"))
insights["zone"] = {k: {"shipments": int(r.n), "late_pct": round(100 * r.delay, 1), "mean_days": round(r.days, 2),
                        "median_cost_per_km": round(r.cpk, 3)} for k, r in zone.iterrows()}
ne = df[df["destination_zone"] == "North-East"]["delayed"]; rest = df[df["destination_zone"] != "North-East"]["delayed"]
insights["north_east"] = {"ne_late_pct": round(100 * ne.mean(), 1), "rest_late_pct": round(100 * rest.mean(), 1),
                          "ne_share_of_shipments_pct": round(100 * len(ne) / len(df), 1),
                          "ne_share_of_late_pct": round(100 * ne.sum() / df["delayed"].sum(), 1)}

# ------------------------------------------------------------------ 6. Cost drivers --------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
samp = df.sample(2500, random_state=1)
for m in order:
    s = samp[samp["shipping_mode"] == m]
    ax.scatter(s["distance_km"], s["delivery_cost_inr"], s=9, alpha=0.35, color=MODE_COL[m], edgecolor="none", label=m)
    g = df[df["shipping_mode"] == m]
    sl = stats.linregress(g["distance_km"], g["delivery_cost_inr"])
    xs = np.linspace(40, df["distance_km"].max(), 40)
    ax.plot(xs, sl.intercept + sl.slope * xs, color=MODE_COL[m], lw=2.4)
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Delivery cost (INR)"); ax.set_title("Cost rises with distance, faster for Express")
ax.legend(markerscale=2.5, fontsize=8.5, loc="upper left")
ax = axes[1]
by_dist = df.assign(band=pd.cut(df["distance_km"], [0, 500, 1000, 1500, 2000, 4000],
                                labels=["<500", "500-1000", "1000-1500", "1500-2000", ">2000"]))
bd = by_dist.groupby("band", observed=True)["cost_per_km"].median()
bars = ax.bar(bd.index.astype(str), bd.values, color=PALETTE[0], width=0.6)
for b, v in zip(bars, bd.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5)
ax.set_xlabel("Distance band (km)"); ax.set_ylabel("Median cost per km (INR)"); ax.set_title("Short hauls cost more per km")
ax.tick_params(axis="x", labelsize=8); plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.grid(axis="x", visible=False)
save_fig(fig, "w3_cost_drivers.png")

mode_stats = df.groupby("shipping_mode").agg(n=("shipment_id", "count"), cpk=("cost_per_km", "median"), days=("actual_days", "mean"),
                                              late=("delayed", "mean"), cost=("delivery_cost_inr", "mean"))
kw = stats.kruskal(*[df.loc[df.shipping_mode == m, "cost_per_km"] for m in order])
insights["mode"] = {k: {"shipments": int(r.n), "median_cost_per_km": round(r.cpk, 3), "mean_days": round(r.days, 2),
                        "late_pct": round(100 * r.late, 1), "mean_cost_inr": round(r.cost, 1)} for k, r in mode_stats.iterrows()}
insights["mode_kruskal_p"] = float(kw.pvalue)
insights["cost_distance"] = {"spearman_rho": round(float(stats.spearmanr(df["distance_km"], df["delivery_cost_inr"])[0]), 3),
                             "short_haul_cpk": round(float(bd.iloc[0]), 3), "long_haul_cpk": round(float(bd.iloc[-1]), 3)}

# ------------------------------------------------------------------ 7. Bottleneck lanes ----------
lane = df.groupby("lane").agg(n=("shipment_id", "count"), late=("delayed", "mean"), days=("actual_days", "mean"),
                              cost=("delivery_cost_inr", "sum")).query("n >= 150")
lane["late"] *= 100
top = lane.sort_values("late", ascending=False).head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
bars = ax.barh(top.index, top["late"], color=PALETTE[0], height=0.58)
ax.axvline(100 * df["delayed"].mean(), color=MUTED, ls="--", lw=1)
ax.text(100 * df["delayed"].mean() + 0.4, -0.75, "network avg", fontsize=8.5, color=INK_2)
for b, v, n in zip(bars, top["late"], top["n"]):
    ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.0f}%  (n={int(n)})", va="center", fontsize=8.5)
ax.set_xlim(0, top["late"].max() * 1.22); ax.set_xlabel("Late deliveries (%)"); ax.grid(axis="y", visible=False)
ax.set_title("Ten slowest delivery lanes (lanes with 150+ shipments)")
save_fig(fig, "w3_bottleneck_lanes.png")
insights["worst_lanes"] = [{"lane": k, "late_pct": round(r.late, 1), "shipments": int(r.n)} for k, r in lane.sort_values("late", ascending=False).head(5).iterrows()]
insights["best_lanes"] = [{"lane": k, "late_pct": round(r.late, 1), "shipments": int(r.n)} for k, r in lane.sort_values("late").head(3).iterrows()]
wh = df.groupby("origin_warehouse").agg(n=("shipment_id", "count"), late=("delayed", "mean"), days=("actual_days", "mean"), cpk=("cost_per_km", "median"))
insights["warehouse"] = {k: {"shipments": int(r.n), "late_pct": round(100 * r.late, 1), "mean_days": round(r.days, 2),
                             "median_cost_per_km": round(r.cpk, 3)} for k, r in wh.iterrows()}

# ------------------------------------------------------------------ 8. Weather / traffic ----------
fig, ax = plt.subplots(figsize=(7.2, 3.8))
qlab = ["Q1 low", "Q2", "Q3", "Q4 high"]
res = {}
for name, col, c in (("Weather severity", "weather_index", PALETTE[0]), ("Traffic congestion", "traffic_index", PALETTE[1])):
    q = pd.qcut(df[col], 4, labels=qlab)
    rate = df.groupby(q, observed=True)["delayed"].mean() * 100
    ax.plot(qlab, rate.values, marker="o", ms=7, lw=2.4, color=c, label=name)
    ax.annotate(name, (3, rate.values[-1]), xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=9, color=INK)
    res[name] = [round(float(v), 1) for v in rate.values]
ax.set_ylabel("Late deliveries (%)"); ax.set_xlabel("Quartile of index (low to high)")
ax.set_title("Late-delivery rate climbs with weather and traffic"); ax.set_ylim(0, 60)
save_fig(fig, "w3_weather_traffic.png")
insights["weather_traffic"] = res
rho_w = stats.spearmanr(df["weather_index"], df["delay_days"]); rho_t = stats.spearmanr(df["traffic_index"], df["delay_days"])
insights["weather_traffic_rho"] = {"weather": [round(float(rho_w[0]), 3), float(rho_w[1])], "traffic": [round(float(rho_t[0]), 3), float(rho_t[1])]}

# ------------------------------------------------------------------ 9. Rating vs lateness ---------
d = df.dropna(subset=["customer_rating"]).copy()
d["bucket"] = pd.cut(d["delay_days"], [-0.01, 0, 1, 2, 100], labels=["On time", "0-1 d late", "1-2 d late", "2+ d late"])
rb = d.groupby("bucket", observed=True)["customer_rating"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(6.6, 3.5))
bars = ax.bar(rb.index.astype(str), rb["mean"], color=PALETTE[0], width=0.55)
for b, (m, n) in zip(bars, rb.itertuples(index=False)):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.04, f"{m:.2f}\n(n={int(n)})", ha="center", fontsize=8.5)
ax.set_ylim(0, 5.6); ax.set_ylabel("Mean rating (1-5)"); ax.set_title("Late deliveries cost customer satisfaction"); ax.grid(axis="x", visible=False)
save_fig(fig, "w3_rating_vs_delay.png")
insights["rating_by_delay"] = {k: {"mean_rating": round(float(r["mean"]), 2), "n": int(r["count"])} for k, r in rb.iterrows()}

save_json({"overall": overall, "describe": desc.reset_index().rename(columns={"index": "variable"}).to_dict("records"),
           "top_corr_pairs": top_pairs, "delay_corr": delay_corr, "actual_corr": actual_corr, "insights": insights},
          "week3_results.json")

print(overall)
print(desc.to_string())
print("delay corr:", delay_corr); print("actual corr:", actual_corr)
print("peak:", insights["peak_season"]); print("NE:", insights["north_east"])
print("carrier:", insights["carrier"]); print("mode:", insights["mode"], insights["mode_kruskal_p"])
print("cost_distance:", insights["cost_distance"]); print("worst lanes:", insights["worst_lanes"])
print("weather/traffic:", res, insights["weather_traffic_rho"]); print("rating:", insights["rating_by_delay"])
print("warehouse:", insights["warehouse"])
