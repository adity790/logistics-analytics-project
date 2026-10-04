"""Week 2 - Data collection simulation, cleaning and preprocessing pipeline.

Input : data/raw/logistics_shipments_raw.csv          (dirty feed, 12,160 rows)
Output: data/processed/logistics_shipments_clean.csv  (analysis-ready)
        data/processed/logistics_shipments_scaled.csv (normalisation demo)
        figures/w2_*.png, results/week2_results.json
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from common import (CARRIER_CANON, CLEAN_FILE, DATA_PROC, INK, INK_2, MUTED, PALETTE, RAW_FILE, SEED,
                    ensure_dirs, parse_mixed_dates, save_fig, save_json, set_style)

ensure_dirs()
set_style()
rng = np.random.default_rng(SEED)
log: list[dict] = []   # cleaning log -> goes into the report


def note(step: str, issue: str, affected: int, action: str) -> None:
    log.append({"step": step, "issue": issue, "rows_affected": int(affected), "action": action})


# ------------------------------------------------------------------ 1. Load & profile ----------
raw = pd.read_csv(RAW_FILE)
df = raw.copy()
n_raw = len(raw)
NUM = ["distance_km", "weight_kg", "volume_m3", "order_value_inr", "weather_index", "traffic_index",
       "planned_days", "actual_days", "delivery_cost_inr", "customer_rating"]
missing_before = raw.isna().sum()
profile = {
    "rows": n_raw, "columns": raw.shape[1],
    "duplicate_ids": int(raw["shipment_id"].duplicated().sum()),
    "missing_cells": int(raw.isna().sum().sum()),
    "missing_pct_by_col": (100 * raw.isna().mean()).round(2).to_dict(),
    "negative_cost": int((raw["delivery_cost_inr"] < 0).sum()),
    "zero_distance": int((raw["distance_km"] == 0).sum()),
    "negative_days": int((raw["actual_days"] <= 0).sum()),
    "days_over_30": int((raw["actual_days"] > 30).sum()),
    "mode_label_variants": int(raw["shipping_mode"].nunique()),
    "carrier_label_variants": int(raw["carrier"].nunique()),
}

# Figure: missingness bar chart (before)
miss = (100 * raw.isna().mean()); miss = miss[miss > 0].sort_values()
fig, ax = plt.subplots(figsize=(7.2, 3.4))
bars = ax.barh(miss.index, miss.values, color=PALETTE[0], height=0.55)
for b, v in zip(bars, miss.values):
    ax.text(v + 0.08, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Missing values (% of rows)"); ax.grid(axis="y", visible=False)
ax.set_title("Missing data in the raw feed"); ax.set_xlim(0, miss.max() * 1.2)
save_fig(fig, "w2_missingness.png")

# ------------------------------------------------------------------ 2. Duplicates --------------
dups = int(df["shipment_id"].duplicated().sum())
df = df.drop_duplicates(subset="shipment_id", keep="first")
note("Duplicates", "Repeated shipment_id (system retries)", dups, "Dropped, keeping first occurrence")

# ------------------------------------------------------------------ 3. Text standardisation ----
mode_variants = df["shipping_mode"].nunique()
df["shipping_mode"] = df["shipping_mode"].astype("string").str.strip().str.title()
df["origin_warehouse"] = df["origin_warehouse"].astype("string").str.strip().str.title()
carrier_before = df["carrier"].astype("string")
df["carrier"] = carrier_before.str.strip().str.lower().map(CARRIER_CANON)
_m = raw["shipping_mode"].astype("string"); _w = raw["origin_warehouse"].astype("string"); _c = raw["carrier"].astype("string")
n_label_fix = int(((_m != _m.str.strip().str.title()) | (_w != _w.str.strip().str.title())
                   | (_c.notna() & (_c != _c.str.strip().str.lower().map(CARRIER_CANON)))).sum())
note("Categorical labels", f"{mode_variants} spellings of 3 shipping modes (case/whitespace); carrier and warehouse case drift",
     n_label_fix, "strip() + title-case; carrier mapped through a canonical dictionary")

# ------------------------------------------------------------------ 4. Dates -------------------
date_raw = df["order_date"].astype("string")
iso_share = date_raw.str.match(r"^\d{4}-\d{2}-\d{2}$").mean()
df["order_date"] = parse_mixed_dates(date_raw)
bad_dates = int(df["order_date"].isna().sum())
note("Dates", f"Three date layouts (ISO {iso_share:.0%}, DD/MM/YYYY, 'Mon DD, YYYY')",
     int((~date_raw.str.match(r"^\d{4}-\d{2}-\d{2}$").fillna(False)).sum()),
     "Parsed with explicit formats (never guessing day/month order); unparseable -> NaT")

# ------------------------------------------------------------------ 5. Invalid values ----------
neg_cost = df["delivery_cost_inr"] < 0
df.loc[neg_cost, "delivery_cost_inr"] = df.loc[neg_cost, "delivery_cost_inr"].abs()
note("Invalid values", "Negative delivery cost (sign-entry error)", neg_cost.sum(), "Absolute value taken (magnitude plausible)")

bad_dist = df["distance_km"] <= 0
df.loc[bad_dist, "distance_km"] = np.nan
note("Invalid values", "Zero distance between different cities", bad_dist.sum(), "Set to missing, re-derived from lane median")

bad_days = (df["actual_days"] <= 0) | (df["actual_days"] > 30)
n_bad_days = int(bad_days.sum())
df = df.loc[~bad_days].copy()
note("Invalid values", "actual_days <= 0 or > 30 (target variable)", n_bad_days,
     "Rows removed - the target cannot be imputed without leaking assumptions")

# distance first: cost-per-km outlier checks need a valid denominator for every row
# distance: lane median of valid values (road distance on a lane is essentially fixed)
lane_med = df.groupby(["origin_warehouse", "destination_hub"])["distance_km"].transform("median")
n_dist = int(df["distance_km"].isna().sum())
df["distance_km"] = df["distance_km"].fillna(lane_med)
note("Missing values", "distance_km missing / invalid", n_dist, "Imputed with median of the same origin-destination lane")

# ------------------------------------------------------------------ 6. Outlier detection --------
def robust_z(x: pd.Series) -> pd.Series:
    med = x.median(); mad = (x - med).abs().median()
    return 0.6745 * (x - med) / (mad if mad else np.nan)

def iqr_flags(x: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = x.quantile([0.25, 0.75]); i = q3 - q1
    return (x < q1 - k * i) | (x > q3 + k * i)

# weight: compared on log scale inside each product category (furniture is heavier by nature)
lw = np.log(df["weight_kg"])
z_w = lw.groupby(df["product_category"]).transform(robust_z)
method_counts = {
    "weight_kg": {
        "IQR (raw, global)": int(iqr_flags(df["weight_kg"].dropna()).sum()),
        "z-score |z|>3 (raw, global)": int((np.abs(stats.zscore(df["weight_kg"].dropna())) > 3).sum()),
        "Robust z>4 (log, per category)": int((z_w.abs() > 4).sum()),
    }
}
flag_w = (z_w.abs() > 4) | (df["weight_kg"] > 200)  # 200 kg = carrier parcel limit (business rule)
df["weight_flag"] = flag_w.fillna(False)

# cost: per-km cost within each shipping mode, log scale
cpk = df["delivery_cost_inr"] / df["distance_km"]
z_c = np.log(cpk).groupby(df["shipping_mode"]).transform(robust_z)
method_counts["delivery_cost_inr"] = {
    "IQR (raw, global)": int(iqr_flags(df["delivery_cost_inr"].dropna()).sum()),
    "z-score |z|>3 (raw, global)": int((np.abs(stats.zscore(df["delivery_cost_inr"].dropna())) > 3).sum()),
    "Robust z>4 (log cost/km, per mode)": int((z_c.abs() > 4).sum()),
}
flag_c = (z_c.abs() > 4).fillna(False)
df["cost_flag"] = flag_c

# figure: before/after for weight and cost (log axes keep the long tail readable)
before_w = df["weight_kg"].copy(); before_c = df["delivery_cost_inr"].copy()

n_w_out, n_c_out = int(flag_w.sum()), int(flag_c.sum())
df.loc[df["weight_flag"], "weight_kg"] = np.nan
df.loc[df["cost_flag"], "delivery_cost_inr"] = np.nan
note("Outliers", "Implausible parcel weights (robust z>4 in log space per category, or >200 kg limit)", n_w_out,
     "Treated as entry errors -> set to missing and imputed")
note("Outliers", "Implausible delivery costs (robust z>4 on log cost/km per shipping mode)", n_c_out,
     "Treated as billing errors -> set to missing and imputed")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, b, a, ttl in ((axes[0], before_w, df["weight_kg"], "Parcel weight (kg)"),
                      (axes[1], before_c, df["delivery_cost_inr"], "Delivery cost (INR)")):
    bins = np.logspace(np.log10(max(b.min(), 0.1)), np.log10(b.max()), 45)
    ax.hist(b.dropna(), bins=bins, color=PALETTE[1], alpha=0.85, label="before")
    ax.hist(a.dropna(), bins=bins, color=PALETTE[0], alpha=0.85, label="after")
    ax.set_xscale("log"); ax.set_title(ttl); ax.set_ylabel("Shipments"); ax.legend(fontsize=8)
save_fig(fig, "w2_outliers_before_after.png")

# ------------------------------------------------------------------ 7. Missing-value treatment --
# 7a. Missingness mechanism check: is weight missing more often in some groups? (chi-square)
miss_w = df["weight_kg"].isna().astype(int)
chi = stats.chi2_contingency(pd.crosstab(df["product_category"], miss_w))
mechanism = {"chi2_p_value_weight_missing_vs_category": float(chi[1]),
             "reading": "p > 0.05 suggests weight is missing (close to) completely at random w.r.t. product category"}

# 7b. Benchmark imputation methods for weight by masking known values
ok = df.dropna(subset=["weight_kg", "volume_m3"]).copy()
mask_idx = ok.sample(frac=0.10, random_state=SEED).index
truth = ok.loc[mask_idx, "weight_kg"]
train = ok.drop(index=mask_idx)
dens = (train["weight_kg"] / train["volume_m3"]).groupby(train["product_category"]).median()
preds = {
    "Global mean": pd.Series(train["weight_kg"].mean(), index=mask_idx),
    "Global median": pd.Series(train["weight_kg"].median(), index=mask_idx),
    "Category median": ok.loc[mask_idx, "product_category"].map(train["weight_kg"].groupby(train["product_category"]).median()),
    "Volume x category density": ok.loc[mask_idx, "volume_m3"] * ok.loc[mask_idx, "product_category"].map(dens),
}
knn_df = pd.get_dummies(ok[["product_category"]], dtype=float)
knn_df["lw"] = np.log(ok["weight_kg"]); knn_df["lv"] = np.log(ok["volume_m3"]); knn_df["lo"] = np.log(ok["order_value_inr"])
knn_in = knn_df.copy(); knn_in.loc[mask_idx, "lw"] = np.nan
knn_out = KNNImputer(n_neighbors=7).fit_transform((knn_in - knn_in.mean()) / knn_in.std())
lw_hat = pd.Series(knn_out[:, knn_df.columns.get_loc("lw")], index=ok.index) * knn_in["lw"].std() + knn_in["lw"].mean()
preds["KNN (k=7, log scale)"] = np.exp(lw_hat.loc[mask_idx])
bench = {k: float((v - truth).abs().mean()) for k, v in preds.items()}

fig, ax = plt.subplots(figsize=(7.2, 3.3))
order = sorted(bench, key=bench.get, reverse=True)
colors = [PALETTE[0] if k != min(bench, key=bench.get) else PALETTE[2] for k in order]
bars = ax.barh(order, [bench[k] for k in order], color=colors, height=0.55)
for b, k in zip(bars, order):
    ax.text(bench[k] + 0.05, b.get_y() + b.get_height() / 2, f"{bench[k]:.2f} kg", va="center", fontsize=9)
ax.set_xlabel("Mean absolute error on 10% masked weights (kg)"); ax.grid(axis="y", visible=False)
ax.set_title("Which imputation method recovers parcel weight best?")
ax.set_xlim(0, max(bench.values()) * 1.18)
save_fig(fig, "w2_imputation_benchmark.png")

# 7c. Apply the chosen strategy

# weight: volume x category density when volume exists, else category median
dens_all = (df["weight_kg"] / df["volume_m3"]).groupby(df["product_category"]).median()
cat_med = df.groupby("product_category")["weight_kg"].transform("median")
n_w = int(df["weight_kg"].isna().sum())
w_from_vol = df["volume_m3"] * df["product_category"].map(dens_all)
df["weight_kg"] = df["weight_kg"].fillna(w_from_vol).fillna(cat_med)
note("Missing values", "weight_kg missing (incl. removed outliers)", n_w,
     f"Volume x category median density (MAE {bench['Volume x category density']:.2f} kg vs {bench['KNN (k=7, log scale)']:.2f} kg for KNN, "
     "chosen for transparency and zero tuning); category median as fallback")

# volume: weight / category density
n_v = int(df["volume_m3"].isna().sum())
df["volume_m3"] = df["volume_m3"].fillna(df["weight_kg"] / df["product_category"].map(dens_all))
note("Missing values", "volume_m3 missing", n_v, "Weight / category median density")

# weather & traffic: median of the same calendar month (seasonal signal)
for col in ("weather_index", "traffic_index"):
    n = int(df[col].isna().sum())
    df[col] = df[col].fillna(df.groupby(df["order_date"].dt.month)[col].transform("median"))
    note("Missing values", f"{col} missing", n, "Median of same calendar month")

# carrier: keep the information that it is unknown
n_car = int(df["carrier"].isna().sum())
df["carrier"] = df["carrier"].fillna("Unknown")
note("Missing values", "carrier missing", n_car, "Kept as explicit category 'Unknown' (no honest way to infer it)")

# cost: median cost-per-km by shipping mode x distance, adjusted for weight via simple ratio
valid_cost = df["delivery_cost_inr"].notna()
cpk_mode = (df.loc[valid_cost, "delivery_cost_inr"] / df.loc[valid_cost, "distance_km"]).groupby(df.loc[valid_cost, "shipping_mode"]).median()
n_cost = int((~valid_cost).sum())
df["delivery_cost_inr"] = df["delivery_cost_inr"].fillna(df["distance_km"] * df["shipping_mode"].map(cpk_mode))
note("Missing values", "delivery_cost_inr removed as outlier", n_cost, "Median cost-per-km of the shipping mode x distance")

# rating: keep missing (outcome-like variable; imputing it would fabricate customer opinion)
n_rate = int(df["customer_rating"].isna().sum())
note("Missing values", "customer_rating missing", n_rate, "Left missing; excluded only from rating-specific statistics")

# ------------------------------------------------------------------ 8. Feature engineering -----
df["delay_days"] = (df["actual_days"] - df["planned_days"]).clip(lower=0).round(2)
df["delayed"] = (df["actual_days"] > df["planned_days"]).astype(int)
df["cost_per_km"] = (df["delivery_cost_inr"] / df["distance_km"]).round(4)
df["cost_per_kg"] = (df["delivery_cost_inr"] / df["weight_kg"]).round(3)
df["lane"] = df["origin_warehouse"] + " -> " + df["destination_hub"]
df["order_month"] = df["order_date"].dt.month
df["order_dow"] = df["order_date"].dt.dayofweek
df["is_weekend"] = (df["order_dow"] >= 5).astype(int)
df["is_peak_season"] = df["order_month"].isin([10, 11, 12]).astype(int)
df = df.drop(columns=["weight_flag", "cost_flag"]).sort_values("order_date").reset_index(drop=True)
df["weight_kg"] = df["weight_kg"].round(2); df["volume_m3"] = df["volume_m3"].round(4)
df["distance_km"] = df["distance_km"].round(1); df["delivery_cost_inr"] = df["delivery_cost_inr"].round(2)
df["weather_index"] = df["weather_index"].round(3); df["traffic_index"] = df["traffic_index"].round(3)

# ------------------------------------------------------------------ 9. Normalisation ------------
skew_table = {}
for col in ("weight_kg", "volume_m3", "order_value_inr", "delivery_cost_inr", "distance_km"):
    skew_table[col] = {"skew_raw": float(stats.skew(df[col])), "skew_log1p": float(stats.skew(np.log1p(df[col])))}

demo_col = "weight_kg"
x = df[[demo_col]]
scaled_variants = {
    "Raw": df[demo_col].values,
    "log1p": np.log1p(df[demo_col].values),
    "Min-max [0,1]": MinMaxScaler().fit_transform(x).ravel(),
    "Z-score": StandardScaler().fit_transform(x).ravel(),
    "Robust (median/IQR)": RobustScaler().fit_transform(x).ravel(),
    "log1p + Z-score": StandardScaler().fit_transform(np.log1p(x)).ravel(),
}
norm_table = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                  "skew": float(stats.skew(v))} for k, v in scaled_variants.items()}

fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
for ax, (k, v) in zip(axes.ravel(), scaled_variants.items()):
    ax.hist(v, bins=50, color=PALETTE[0], alpha=0.9)
    ax.set_title(f"{k}  (skew {stats.skew(v):.2f})", fontsize=10)
    ax.set_yticks([])
fig.suptitle("Effect of normalisation on parcel weight", fontweight="bold", y=1.02)
save_fig(fig, "w2_normalisation.png")

# scaled copy for modelling demos (note: in Week 4 scaling is fitted inside CV folds to avoid leakage)
num_model = ["distance_km", "weight_kg", "volume_m3", "order_value_inr", "weather_index", "traffic_index", "planned_days"]
sc = df.copy()
for c in ("weight_kg", "volume_m3", "order_value_inr"):
    sc[c] = np.log1p(sc[c])
sc[num_model] = StandardScaler().fit_transform(sc[num_model])

# ------------------------------------------------------------------ 10. Quality scorecard ------
def scorecard(d: pd.DataFrame, ref_cols) -> dict:
    cells = d[ref_cols].size
    complete = 100 * (1 - d[ref_cols].isna().sum().sum() / cells)
    return {"rows": int(len(d)), "completeness_pct": round(complete, 2)}

ref = [c for c in raw.columns]
before_card = scorecard(raw, ref)
after_card = scorecard(df, ref)
before_card["duplicate_rows"] = int(raw["shipment_id"].duplicated().sum()); after_card["duplicate_rows"] = int(df["shipment_id"].duplicated().sum())
before_card["invalid_values"] = int(((raw["delivery_cost_inr"] <= 0) | (raw["distance_km"] <= 0) | (raw["actual_days"] <= 0) | (raw["actual_days"] > 30)).sum())
after_card["invalid_values"] = int(((df["delivery_cost_inr"] <= 0) | (df["distance_km"] <= 0) | (df["actual_days"] <= 0) | (df["actual_days"] > 30)).sum())
before_card["label_variants_shipping_mode"] = int(raw["shipping_mode"].nunique()); after_card["label_variants_shipping_mode"] = int(df["shipping_mode"].nunique())

df.to_csv(CLEAN_FILE, index=False)
sc.to_csv(DATA_PROC / "logistics_shipments_scaled.csv", index=False)

save_json({
    "profile": profile, "cleaning_log": log, "outlier_method_counts": method_counts,
    "missingness_mechanism": mechanism, "imputation_benchmark_mae_kg": bench,
    "skew_table": skew_table, "normalisation_table": norm_table,
    "scorecard_before": before_card, "scorecard_after": after_card,
    "rows_raw": n_raw, "rows_clean": int(len(df)), "rows_removed": n_raw - int(len(df)),
    "date_range": [str(df["order_date"].min().date()), str(df["order_date"].max().date())],
    "cpk_by_mode": cpk_mode.round(3).to_dict(), "density_by_category": dens_all.round(1).to_dict(),
}, "week2_results.json")

print(f"raw {n_raw} -> clean {len(df)}")
print(pd.DataFrame(log).to_string(index=False))
print("outlier methods:", method_counts)
print("benchmark:", {k: round(v, 3) for k, v in bench.items()})
print("mechanism p:", mechanism["chi2_p_value_weight_missing_vs_category"])
print("before:", before_card); print("after:", after_card)
print("skew:", {k: (round(v['skew_raw'], 2), round(v['skew_log1p'], 2)) for k, v in skew_table.items()})
print("remaining NaN:", df.isna().sum()[df.isna().sum() > 0].to_dict())
