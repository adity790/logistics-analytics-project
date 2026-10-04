"""Week 4 - Predictive modelling and optimisation for the logistics network.

Part A  Predict delivery time (actual_days): baselines, linear, tree and ensemble models, CV, tuning.
Part B  Predict late delivery risk (classification) and quantify how much a risk-ranked review would catch.
Part C  Optimisation: (1) warehouse -> hub allocation LP, (2) carrier-mix LP per zone,
        (3) model-based delivery promises (quantile regression) vs rule-based promises.

Input : data/processed/logistics_shipments_clean.csv
Output: figures/w4_*.png, results/week4_results.json
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linprog
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, TimeSeriesSplit, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from common import (CLEAN_FILE, INK, INK_2, MUTED, PALETTE, SEED, SEQ_BLUES, ensure_dirs, save_fig, save_json, set_style)

warnings.filterwarnings("ignore")
ensure_dirs()
set_style()
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)

df = pd.read_csv(CLEAN_FILE, parse_dates=["order_date"]).sort_values("order_date").reset_index(drop=True)

NUM = ["distance_km", "weight_kg", "volume_m3", "order_value_inr", "weather_index", "traffic_index", "planned_days",
       "is_peak_season", "order_month", "order_dow"]
CAT = ["shipping_mode", "priority", "carrier", "origin_warehouse", "destination_zone", "product_category"]
FEATURES = NUM + CAT          # everything here is known (or forecast) at dispatch time - no leakage
TARGET = "actual_days"

X, y = df[FEATURES], df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)
idx_tr, idx_te = X_tr.index, X_te.index


def prep() -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])


def pipe(model) -> Pipeline:
    return Pipeline([("prep", prep()), ("model", model)])


# ------------------------------------------------------------------ Part A: regression ---------
models = {
    "Mean predictor": DummyRegressor(strategy="mean"),
    "Linear regression": LinearRegression(),
    "Ridge (alpha=1)": Ridge(alpha=1.0),
    "Decision tree (depth 8)": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=SEED),
    "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
    "Gradient boosting (default)": HistGradientBoostingRegressor(random_state=SEED),
}
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
rows = []
fitted = {}
for name, m in models.items():
    p = pipe(m)
    cvr = cross_validate(p, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
    p.fit(X_tr, y_tr); pred = p.predict(X_te); fitted[name] = p
    rows.append({"model": name, "cv_rmse": -cvr["test_rmse"].mean(), "cv_rmse_sd": cvr["test_rmse"].std(),
                 "cv_mae": -cvr["test_mae"].mean(), "cv_r2": cvr["test_r2"].mean(),
                 "test_rmse": root_mean_squared_error(y_te, pred), "test_mae": mean_absolute_error(y_te, pred),
                 "test_r2": r2_score(y_te, pred)})

# business baseline: simply trust the promised delivery days
base_pred = X_te["planned_days"]
baseline_row = {"model": "Promised days (current rule)", "cv_rmse": np.nan, "cv_rmse_sd": np.nan, "cv_mae": np.nan, "cv_r2": np.nan,
                "test_rmse": root_mean_squared_error(y_te, base_pred), "test_mae": mean_absolute_error(y_te, base_pred),
                "test_r2": r2_score(y_te, base_pred)}

# hyper-parameter tuning of the best ensemble family
best_family = min(("Random forest", "Gradient boosting (default)"), key=lambda n: next(r["cv_rmse"] for r in rows if r["model"] == n))
if best_family == "Random forest":
    est = RandomForestRegressor(n_estimators=250, n_jobs=-1, random_state=SEED)
    space = {"model__max_depth": [8, 12, 16, None], "model__min_samples_leaf": [1, 3, 5, 10, 20],
             "model__max_features": [0.3, 0.5, 0.8, 1.0]}
else:
    est = HistGradientBoostingRegressor(random_state=SEED)
    space = {"model__learning_rate": [0.03, 0.05, 0.08, 0.12], "model__max_leaf_nodes": [8, 15, 31, 63],
             "model__min_samples_leaf": [10, 20, 40, 80], "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
             "model__max_iter": [200, 400, 600]}
search = RandomizedSearchCV(pipe(est), space, n_iter=20, cv=cv, scoring="neg_root_mean_squared_error",
                            random_state=SEED, n_jobs=1, refit=True)
search.fit(X_tr, y_tr)
tuned = search.best_estimator_
pred_t = tuned.predict(X_te)
tuned_name = f"Tuned {best_family.split(' (')[0].lower()}"
rows.append({"model": tuned_name, "cv_rmse": -search.best_score_, "cv_rmse_sd": np.nan, "cv_mae": np.nan, "cv_r2": np.nan,
             "test_rmse": root_mean_squared_error(y_te, pred_t), "test_mae": mean_absolute_error(y_te, pred_t),
             "test_r2": r2_score(y_te, pred_t)})
rows.append(baseline_row)
reg_table = pd.DataFrame(rows)
best_params = {k.replace("model__", ""): (v if not isinstance(v, (np.integer, np.floating)) else v.item()) for k, v in search.best_params_.items()}

# time-aware robustness check: forward-chaining CV on date-ordered data
tscv = TimeSeriesSplit(n_splits=5)
ts_rmse = []
for tr, va in tscv.split(df):
    p = Pipeline(tuned.steps); p.fit(X.iloc[tr], y.iloc[tr])
    ts_rmse.append(root_mean_squared_error(y.iloc[va], p.predict(X.iloc[va])))

# figures -----------------------------------------------------------------------------------
tbl = reg_table.sort_values("test_rmse", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.2))
best_name = reg_table.loc[reg_table["test_rmse"].idxmin(), "model"]
cols = [PALETTE[2] if n == best_name else (PALETTE[1] if n.startswith("Promised") else PALETTE[0]) for n in tbl["model"]]
bars = ax.barh(tbl["model"], tbl["test_rmse"], color=cols, height=0.58)
for b, v, r2 in zip(bars, tbl["test_rmse"], tbl["test_r2"]):
    ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}   (R² {r2:.2f})", va="center", fontsize=8.8)
ax.set_xlim(0, tbl["test_rmse"].max() * 1.28); ax.set_xlabel("Hold-out RMSE (days) - lower is better"); ax.grid(axis="y", visible=False)
ax.set_title("Predicting delivery time: model comparison on the hold-out set")
save_fig(fig, "w4_model_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
ax = axes[0]
ss = np.random.default_rng(SEED).choice(len(y_te), size=min(2200, len(y_te)), replace=False)
ax.scatter(y_te.values[ss], pred_t[ss], s=9, alpha=0.35, color=PALETTE[0], edgecolor="none")
lim = [0, max(y_te.max(), pred_t.max())]
ax.plot(lim, lim, color=PALETTE[1], lw=2); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual delivery days"); ax.set_ylabel("Predicted delivery days"); ax.set_title("Predicted versus actual")
ax = axes[1]
res = y_te.values - pred_t
ax.hist(res, bins=50, color=PALETTE[0], alpha=0.9); ax.axvline(0, color=PALETTE[1], lw=2)
ax.set_xlabel("Residual (actual - predicted, days)"); ax.set_ylabel("Shipments"); ax.set_title(f"Residuals (mean {res.mean():.2f}, sd {res.std():.2f})")
save_fig(fig, "w4_pred_vs_actual.png")

pi = permutation_importance(tuned, X_te, y_te, n_repeats=8, random_state=SEED, scoring="neg_root_mean_squared_error", n_jobs=1)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
imp_sd = pd.Series(pi.importances_std, index=FEATURES)
top = imp.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
bars = ax.barh(top.index, top.values, xerr=imp_sd[top.index].values, color=PALETTE[0], height=0.58, error_kw=dict(ecolor=MUTED, lw=1))
for b, v in zip(bars, top.values):
    ax.text(v + imp.max() * 0.03, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5)
ax.set_xlabel("Increase in RMSE when the feature is shuffled (days)"); ax.grid(axis="y", visible=False)
ax.set_title("What drives delivery time? Permutation importance")
save_fig(fig, "w4_feature_importance.png")

# ------------------------------------------------------------------ Part B: late-risk classifier -
yc = df["delayed"]
yc_tr, yc_te = yc.loc[idx_tr], yc.loc[idx_te]
clfs = {"Logistic regression": LogisticRegression(max_iter=2000, C=1.0),
        "Gradient boosting classifier": HistGradientBoostingClassifier(learning_rate=0.06, max_iter=300, max_leaf_nodes=15,
                                                                        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)}
clf_rows, probs = [], {}
for name, m in clfs.items():
    p = pipe(m)
    cvr = cross_validate(p, X_tr, yc_tr, cv=cv, scoring={"auc": "roc_auc", "ap": "average_precision"}, n_jobs=1)
    p.fit(X_tr, yc_tr); pr = p.predict_proba(X_te)[:, 1]; probs[name] = pr
    thr = 0.5
    clf_rows.append({"model": name, "cv_auc": cvr["test_auc"].mean(), "cv_auc_sd": cvr["test_auc"].std(),
                     "test_auc": roc_auc_score(yc_te, pr), "test_avg_precision": average_precision_score(yc_te, pr),
                     "test_f1_at_0.5": f1_score(yc_te, pr >= thr), "base_rate": float(yc_te.mean())})
clf_table = pd.DataFrame(clf_rows)
best_clf = clf_table.loc[clf_table["test_auc"].idxmax(), "model"]
pr = probs[best_clf]

order_ = np.argsort(-pr)
late_sorted = yc_te.values[order_]
cum = np.cumsum(late_sorted) / late_sorted.sum()
share = np.arange(1, len(pr) + 1) / len(pr)
gain20 = float(cum[int(0.2 * len(pr)) - 1]); gain30 = float(cum[int(0.3 * len(pr)) - 1])
prec_top20 = float(late_sorted[: int(0.2 * len(pr))].mean())
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(share * 100, cum * 100, color=PALETTE[0], lw=2.6)
ax.plot([0, 100], [0, 100], color=MUTED, lw=1.2, ls="--")
ax.scatter([20], [gain20 * 100], s=70, color=PALETTE[1], zorder=5)
ax.annotate(f"Reviewing the riskiest 20% of shipments\ncatches {gain20*100:.0f}% of all late deliveries", (20, gain20 * 100),
            xytext=(30, gain20 * 100 - 22), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.text(62, 52, "random review", color=INK_2, fontsize=8.5, rotation=34)
ax.set_xlabel("Shipments reviewed (%, highest predicted risk first)"); ax.set_ylabel("Late deliveries caught (%)")
ax.set_title(f"Late-risk gain curve ({best_clf}, AUC {clf_table['test_auc'].max():.2f})")
ax.set_xlim(0, 100); ax.set_ylim(0, 102)
save_fig(fig, "w4_gain_curve.png")

# ------------------------------------------------------------------ Part C1: warehouse -> hub LP --
PENALTY = 150.0          # assumed INR cost of a late delivery (credit, re-attempt, support contact)
CAP_FACTOR = 1.20        # each warehouse can absorb +20% volume
W = sorted(df["origin_warehouse"].unique()); H = sorted(df["destination_hub"].unique())
lane = (df.groupby(["origin_warehouse", "destination_hub"])
        .agg(n=("shipment_id", "count"), cost=("delivery_cost_inr", "mean"), late=("delayed", "mean")))
cur = lane["n"].unstack().reindex(index=W, columns=H).fillna(0.0)
cost_m = lane["cost"].unstack().reindex(index=W, columns=H)
late_m = lane["late"].unstack().reindex(index=W, columns=H)
cost_m = cost_m.fillna(cost_m.max().max()); late_m = late_m.fillna(late_m.max().max())
N_TOT = float(cur.values.sum())
base_late_total = float((cur.values * late_m.values).sum())
base_cost_ps = float((cur.values * cost_m.values).sum() / N_TOT)
base_late_pct = 100 * base_late_total / N_TOT


def solve_alloc(penalty=PENALTY, cap_factor=CAP_FACTOR, floor=0.0, protect_service=False):
    """Transportation LP. floor = minimum share of today's volume each lane must keep (lane stickiness);
    protect_service = network late-delivery count may not exceed today's."""
    C = (cost_m + penalty * late_m).values
    nW, nH = C.shape
    demand = cur.sum(axis=0).values
    cap = cur.sum(axis=1).values * cap_factor
    A_eq = np.zeros((nH, nW * nH)); A_ub = np.zeros((nW, nW * nH))
    for h in range(nH):
        for w in range(nW):
            A_eq[h, w * nH + h] = 1
    for w in range(nW):
        A_ub[w, w * nH:(w + 1) * nH] = 1
    b_ub = list(cap)
    if protect_service:
        A_ub = np.vstack([A_ub, late_m.values.ravel()]); b_ub.append(base_late_total)
    lb = floor * cur.values.ravel()
    r = linprog(C.ravel(), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=demand, bounds=[(l, None) for l in lb], method="highs")
    assert r.success, r.message
    x = r.x.reshape(nW, nH)
    return x, {"delivery_cost_per_shipment": float((x * cost_m.values).sum() / N_TOT),
               "late_pct": float(100 * (x * late_m.values).sum() / N_TOT),
               "reassigned_pct": float(100 * np.abs(x - cur.values).sum() / 2 / N_TOT)}


SCEN = [("Cost-optimal,\nfree reassignment", dict(floor=0.0, protect_service=False)),
        ("Service-protected,\nfree reassignment", dict(floor=0.0, protect_service=True)),
        ("Practical:\n<=50% of a lane moves", dict(floor=0.5, protect_service=True)),
        ("Conservative:\n<=25% of a lane moves", dict(floor=0.75, protect_service=True))]
MAIN = 2
scen_rows, sols = [], []
for name, kw_ in SCEN:
    xs_, st = solve_alloc(**kw_)
    sols.append(xs_)
    scen_rows.append({"scenario": name.replace("\n", " "), **st,
                      "delivery_cost_saving_pct": 100 * (base_cost_ps - st["delivery_cost_per_shipment"]) / base_cost_ps})
scen = pd.DataFrame(scen_rows)
pen_sens = []
for pen in (0, 150, 400):
    _, st = solve_alloc(penalty=pen, floor=0.5, protect_service=True)
    pen_sens.append({"late_penalty_inr": pen, **st, "delivery_cost_saving_pct": 100 * (base_cost_ps - st["delivery_cost_per_shipment"]) / base_cost_ps})
x_opt = sols[MAIN]
opt = pd.DataFrame(x_opt, index=W, columns=H)
cur_share = cur / cur.sum(axis=0); opt_share = opt / opt.sum(axis=0)

fig = plt.figure(figsize=(11.6, 4.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.08)
axes = [fig.add_subplot(gs[0, 0]), None, fig.add_subplot(gs[0, 2])]
axes[1] = fig.add_subplot(gs[0, 1], sharey=axes[0])
for ax, mat, ttl in ((axes[0], cur_share, "Current allocation"), (axes[1], opt_share, "Practical optimised allocation")):
    im = ax.imshow(mat.T.values * 100, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(W))); ax.set_xticklabels(W, fontsize=8.5, rotation=20)
    ax.set_yticks(range(len(H))); ax.set_yticklabels(H if ax is axes[0] else [], fontsize=8.5); ax.grid(False); ax.set_title(ttl)
    for i in range(len(H)):
        for j in range(len(W)):
            v = mat.T.values[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="#ffffff" if v > 55 else INK)
cb = fig.colorbar(im, cax=axes[2]); cb.set_label("% of hub volume served by warehouse", fontsize=9)
axes[0].set_xlabel("Origin warehouse"); axes[1].set_xlabel("Origin warehouse"); axes[0].set_ylabel("Destination hub")
save_fig(fig, "w4_lp_allocation.png")

# ------------------------------------------------------------------ Part C2: carrier mix per zone -
known = df[df["carrier"] != "Unknown"]
CARR = sorted(known["carrier"].unique())
Z = sorted(known["destination_zone"].unique())
cz = known.groupby(["destination_zone", "carrier"]).agg(n=("shipment_id", "count"), cost=("delivery_cost_inr", "mean"),
                                                         late=("delayed", "mean"))
carrier_rows, carrier_mix = [], {}
for z in Z:
    t = cz.loc[z].reindex(CARR)
    unit = t["cost"].values                      # minimise delivery cost ...
    lt = t["late"].values
    N = t["n"].sum()
    cur_x = t["n"].values.astype(float)
    # ... without letting the zone's late-delivery count exceed today's, each carrier keeping 10-45% of the zone
    r = linprog(unit, A_ub=lt.reshape(1, -1), b_ub=[float((cur_x * lt).sum())], A_eq=np.ones((1, len(CARR))), b_eq=[N],
                bounds=[(0.10 * N, 0.45 * N)] * len(CARR), method="highs")
    assert r.success, r.message
    b, o = float((cur_x * unit).sum() / N), float((r.x * unit).sum() / N)
    carrier_rows.append({"zone": z, "baseline": b, "optimised": o, "saving_pct": 100 * (b - o) / b,
                         "late_pct_before": float(100 * (cur_x * lt).sum() / N), "late_pct_after": float(100 * (r.x * lt).sum() / N), "n": int(N)})
    carrier_mix[z] = {"current_share_pct": dict(zip(CARR, (100 * cur_x / N).round(1))),
                      "optimised_share_pct": dict(zip(CARR, (100 * r.x / N).round(1)))}
carrier_df = pd.DataFrame(carrier_rows)
tot_n = known.groupby("destination_zone").size().reindex(Z).values
carrier_overall_saving = float(100 * (1 - (carrier_df["optimised"] * tot_n).sum() / (carrier_df["baseline"] * tot_n).sum()))
carrier_late_before = float((carrier_df["late_pct_before"] * carrier_df["n"]).sum() / carrier_df["n"].sum())
carrier_late_after = float((carrier_df["late_pct_after"] * carrier_df["n"]).sum() / carrier_df["n"].sum())

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
ax = axes[0]
xpos = np.arange(len(scen))
bars = ax.bar(xpos, scen["delivery_cost_saving_pct"], color=[PALETTE[0]] * len(scen), width=0.6)
bars[MAIN].set_color(PALETTE[2])
for i, r in enumerate(scen.itertuples()):
    ax.text(i, r.delivery_cost_saving_pct + 0.8, f"{r.delivery_cost_saving_pct:.1f}%\nlate {r.late_pct:.1f}%", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(xpos); ax.set_xticklabels(["Cost-only,\nfree moves", "Service-\nprotected,\nfree moves", "Practical:\n<=50% of lane\nmoves", "Conservative:\n<=25% of lane\nmoves"], fontsize=8)
ax.set_ylabel("Delivery-cost saving (%)"); ax.set_ylim(0, scen["delivery_cost_saving_pct"].max() * 1.3)
ax.set_title(f"Warehouse LP scenarios (baseline late rate {base_late_pct:.1f}%)"); ax.grid(axis="x", visible=False)
ax = axes[1]
cd = carrier_df.sort_values("saving_pct")
bars = ax.barh(cd["zone"], cd["saving_pct"], color=PALETTE[0], height=0.55)
for b, v in zip(bars, cd["saving_pct"]):
    ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Expected cost saving (%)"); ax.set_title("Carrier-mix LP: saving by destination zone")
ax.set_xlim(0, cd["saving_pct"].max() * 1.25); ax.grid(axis="y", visible=False)
save_fig(fig, "w4_savings.png")

# ------------------------------------------------------------------ Part C3: model-based promises -
test = df.loc[idx_te].copy()
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
frontier = []
for a in alphas:
    q = pipe(HistGradientBoostingRegressor(loss="quantile", quantile=a, learning_rate=0.06, max_iter=300, max_leaf_nodes=15,
                                           min_samples_leaf=40, random_state=SEED))
    q.fit(X_tr, y_tr)
    promise = np.maximum(np.ceil(q.predict(X_te)), 1)
    frontier.append({"quantile": a, "avg_promised_days": float(promise.mean()), "on_time_pct": float(100 * (y_te.values <= promise).mean())})
frontier = pd.DataFrame(frontier)
cur_avg_promise = float(X_te["planned_days"].mean()); cur_otd = float(100 * (y_te.values <= X_te["planned_days"].values).mean())
# on-time % the model-based promise would achieve at the SAME average promise length (linear interpolation)
otd_same_len = float(np.interp(cur_avg_promise, frontier["avg_promised_days"], frontier["on_time_pct"]))
buf = pd.DataFrame([{"buffer_days": b_, "avg_promised_days": float((X_te["planned_days"].values + b_).mean()),
                     "on_time_pct": float(100 * (y_te.values <= X_te["planned_days"].values + b_).mean())} for b_ in (0, 1, 2)])
len_for_90_buffer = float(np.interp(90, buf["on_time_pct"], buf["avg_promised_days"]))
len_for_90 = float(np.interp(90, frontier["on_time_pct"], frontier["avg_promised_days"]))

fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.plot(frontier["avg_promised_days"], frontier["on_time_pct"], color=PALETTE[0], lw=2.5, marker="o", ms=6)
ax.plot(buf["avg_promised_days"], buf["on_time_pct"], color=PALETTE[1], lw=2.2, marker="s", ms=7, ls="--")
for r in frontier.itertuples():
    if int(round(r.quantile * 100)) in (50, 80, 95):
        ax.annotate(f"q{int(round(r.quantile*100))}", (r.avg_promised_days, r.on_time_pct), xytext=(6, -13), textcoords="offset points", fontsize=8, color=INK_2)
for r in buf.itertuples():
    ax.annotate(f"+{int(r.buffer_days)} d", (r.avg_promised_days, r.on_time_pct), xytext=(6, -13), textcoords="offset points", fontsize=8, color=INK_2)
ax.axhline(90, color=MUTED, lw=1, ls=":")
ax.text(3.72, 91, "90% target", fontsize=8.5, color=INK_2)
ax.text(4.08, 55, "Model-based promises\n(per-shipment quantile)", color=PALETTE[0], fontsize=9, fontweight="bold")
ax.text(4.55, 74, "Flat buffer on\ncurrent rule", color=PALETTE[1], fontsize=9, fontweight="bold")
ax.set_xlabel("Average promised delivery time (days)"); ax.set_ylabel("On-time deliveries (%)")
ax.set_title("Reaching 90% on-time: model-based promise vs flat buffer"); ax.set_ylim(50, 101); ax.set_xlim(3.7, 5.9)
save_fig(fig, "w4_promise_frontier.png")

# ------------------------------------------------------------------ Save --------------------------
top_moves = []
diff = (opt_share - cur_share) * 100
for h in H:
    w_up = diff[h].idxmax(); w_dn = diff[h].idxmin()
    top_moves.append({"hub": h, "gain_warehouse": w_up, "gain_pp": round(float(diff[h].max()), 1),
                      "lose_warehouse": w_dn, "lose_pp": round(float(diff[h].min()), 1)})
top_moves = sorted(top_moves, key=lambda d: -d["gain_pp"])[:5]

save_json({
    "features": {"numeric": NUM, "categorical": CAT, "target": TARGET},
    "split": {"train": int(len(X_tr)), "test": int(len(X_te))},
    "regression_table": reg_table.round(4).to_dict("records"),
    "best_model": best_name, "tuned_family": best_family, "best_params": best_params,
    "tuning": {"n_iter": 20, "cv_folds": 5, "scoring": "neg RMSE"},
    "time_series_cv_rmse": [round(v, 4) for v in ts_rmse], "time_series_cv_rmse_mean": float(np.mean(ts_rmse)),
    "residual": {"mean": float(res.mean()), "sd": float(res.std()), "share_within_0_5d": float((np.abs(res) <= 0.5).mean()),
                 "share_within_1d": float((np.abs(res) <= 1.0).mean())},
    "importance": imp.round(4).head(10).to_dict(),
    "classification_table": clf_table.round(4).to_dict("records"), "best_classifier": best_clf,
    "gain": {"top20_captures_pct": 100 * gain20, "top30_captures_pct": 100 * gain30, "top20_precision_pct": 100 * prec_top20,
             "base_rate_pct": 100 * float(yc_te.mean())},
    "lp_warehouse": {"penalty_inr": PENALTY, "capacity_factor": CAP_FACTOR, "baseline_delivery_cost": base_cost_ps,
                     "baseline_late_pct": base_late_pct, "scenarios": scen.round(3).to_dict("records"), "main_index": MAIN,
                     "main": scen.round(3).to_dict("records")[MAIN], "top_moves": top_moves,
                     "penalty_sensitivity": pd.DataFrame(pen_sens).round(3).to_dict("records"), "annual_shipments": int(N_TOT)},
    "lp_carrier": {"table": carrier_df.round(2).to_dict("records"), "mix": carrier_mix, "overall_saving_pct": carrier_overall_saving,
                   "late_pct_before": carrier_late_before, "late_pct_after": carrier_late_after,
                   "bounds": "each carrier 10%-45% of zone volume"},
    "promise": {"frontier": frontier.round(2).to_dict("records"), "current_avg_promise": cur_avg_promise, "current_on_time_pct": cur_otd,
                "on_time_at_same_length_pct": otd_same_len, "avg_promise_for_90pct": len_for_90,
                "buffer_frontier": buf.round(2).to_dict("records"), "avg_promise_for_90pct_flat_buffer": len_for_90_buffer},
}, "week4_results.json")

pd.set_option("display.width", 200)
print(reg_table.round(3).to_string(index=False))
print("best params:", best_params, "| family:", best_family)
print("time-series CV RMSE:", [round(v, 3) for v in ts_rmse])
print("residual:", round(res.mean(), 3), round(res.std(), 3), "within .5d:", round(float((np.abs(res) <= 0.5).mean()), 3))
print("importance:", imp.round(3).head(8).to_dict())
print(clf_table.round(3).to_string(index=False)); print("gain20:", round(gain20, 3), "prec20:", round(prec_top20, 3))
print("base cost/ship", round(base_cost_ps, 1), "late%", round(base_late_pct, 2)); print(scen.round(2).to_string(index=False)); print(pd.DataFrame(pen_sens).round(2).to_string(index=False))
print(carrier_df.round(2).to_string(index=False)); print("carrier overall saving%", round(carrier_overall_saving, 2), "late before/after", round(carrier_late_before, 2), round(carrier_late_after, 2))
print(frontier.round(2).to_string(index=False)); print("buffer", buf.round(2).to_dict("records"), "len90 buffer", round(len_for_90_buffer, 2), "cur", round(cur_avg_promise, 2), round(cur_otd, 1), "same-length OTD", round(otd_same_len, 1), "len for 90", round(len_for_90, 2))
