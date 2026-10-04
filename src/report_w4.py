"""Week 4 report: predictive modelling and optimisation."""
from __future__ import annotations

from docx_helpers import Report
from reports_common import AUTHOR, FIG, MONTH_YEAR, f, grab, load_json, refs

SCRIPT = "04_modeling_optimisation.py"
HDR = "# ------------------------------------------------------------------ "


def build() -> Report:
    r = load_json("week4_results.json")
    reg = {x["model"]: x for x in r["regression_table"]}
    best = reg[r["best_model"]]; base = reg["Promised days (current rule)"]; lin = reg["Linear regression"]
    clf = {x["model"]: x for x in r["classification_table"]}
    bc = clf[r["best_classifier"]]; g = r["gain"]; lp = r["lp_warehouse"]; main = lp["main"]
    cz, pr, res, imp = r["lp_carrier"], r["promise"], r["residual"], r["importance"]

    R = Report("Week 4 | Predictive Modeling and Optimization in Logistics Systems")
    R.title_page("Logistics Data Science Internship - Week 4", "Predictive Modeling and Optimization",
                 "Forecasting delivery time and late risk, then optimising warehouse allocation, carrier mix and delivery promises", AUTHOR,
                 [("Report date", MONTH_YEAR), ("Dataset", f"Cleaned shipments (n = {r['split']['train'] + r['split']['test']:,}); 80/20 split: {r['split']['train']:,} train, {r['split']['test']:,} test"),
                  ("Tools", "Python 3 (scikit-learn, SciPy HiGHS linear programming, pandas, Matplotlib)"), ("Script", "src/04_modeling_optimisation.py")])

    R.h1("1. Executive summary")
    R.p(f"This report builds a predictive model for delivery time (days from order to delivery), a classifier for late-delivery risk, and three optimisation models that use them. "
        f"The tuned gradient-boosting model reaches a hold-out RMSE of {best['test_rmse']:.2f} days (MAE {best['test_mae']:.2f}, R-squared {best['test_r2']:.2f}), against RMSE {base['test_rmse']:.2f} for simply trusting the promised days "
        f"and {lin['test_rmse']:.2f} for linear regression. Because most variation in delivery time is already explained by distance and the promised days, the gain over simple baselines is real but modest, and the report says so plainly.")
    R.table(["Result", "Value"], [
        ["Delivery-time model (hold-out)", f"RMSE {best['test_rmse']:.3f} d, MAE {best['test_mae']:.3f} d, R-squared {best['test_r2']:.3f}; {100*res['share_within_0_5d']:.0f}% of predictions within 0.5 day, {100*res['share_within_1d']:.0f}% within 1 day"],
        ["Late-risk classifier (hold-out)", f"ROC-AUC {bc['test_auc']:.2f}; reviewing the riskiest 20% of shipments catches {g['top20_captures_pct']:.0f}% of late deliveries (precision {g['top20_precision_pct']:.0f}% vs {g['base_rate_pct']:.0f}% base rate)"],
        ["Warehouse-to-hub LP (practical)", f"{main['delivery_cost_saving_pct']:.1f}% lower delivery cost, late rate unchanged at {main['late_pct']:.1f}%, {main['reassigned_pct']:.0f}% of volume re-routed"],
        ["Carrier-mix LP", f"{cz['overall_saving_pct']:.1f}% lower cost overall; late rate {cz['late_pct_before']:.1f}% to {cz['late_pct_after']:.1f}%"],
        ["Model-based delivery promise", f"90% on-time needs an average promise of {pr['avg_promise_for_90pct']:.2f} days vs {pr['avg_promise_for_90pct_flat_buffer']:.2f} days with a flat buffer"]],
        widths_cm=[5, 11.8], caption="Headline results.", font=8.5)

    R.h1("2. Problem definition and data")
    R.p("**Prediction target.** actual_days, the delivery time in days, is a continuous regression target. A second target, delayed (actual_days greater than planned_days), supports a classification model for proactive intervention. "
        "**Business use.** Better delivery-time predictions allow realistic promise dates; late-risk scores allow the operations team to review a small share of shipments instead of all of them.")
    R.p("**Features.** Only information available at dispatch time is used, so there is no target leakage: distance, weight, volume, order value, weather and traffic indices (as forecast), planned days, peak-season flag, order month and weekday (numeric); "
        "shipping mode, priority, carrier, origin warehouse, destination zone and product category (categorical, one-hot encoded). Customer rating, delay_days and delayed are excluded because they are only known after delivery.")
    R.p(f"**Split.** An 80/20 random split ({r['split']['train']:,} / {r['split']['test']:,}) with seed 42. All tuning uses 5-fold cross-validation on the training part only; the test set is touched once for final scoring. "
        "A forward-chaining time-series CV is added as a robustness check because deployed models predict the future from the past.")
    R.code(grab(SCRIPT, "NUM = [\"distance_km\"", HDR + "Part A", 20), "Code 1. Features, split and preprocessing pipeline")

    R.h1("3. Model selection and implementation")
    R.h2("3.1 Candidate models and rationale")
    R.table(["Model", "Why it was included"], [
        ["Mean predictor", "Floor: any useful model must beat predicting the average"],
        ["Promised days (current rule)", "Business baseline: how good is today's promise as a predictor?"],
        ["Linear / Ridge regression", "Interpretable, fast; delivery time is strongly linear in distance and promised days"],
        ["Decision tree (depth 8)", "Captures interactions and non-linearities; easy to explain, but high variance"],
        ["Random forest (Breiman, 2001)", "Averages many trees; reduces variance, robust to scaling and outliers"],
        ["Gradient boosting (Friedman, 2001)", "Builds trees sequentially on residuals; usually the strongest tabular learner"]],
        widths_cm=[5, 11.8], caption="Candidate models.")
    R.code(grab(SCRIPT, "models = {", "# business baseline", 22), "Code 2. Training and cross-validating every candidate inside one Pipeline")
    R.p("Scaling and encoding live inside a scikit-learn Pipeline, so they are re-fitted on each training fold. This prevents information from the validation fold leaking into preprocessing.")
    R.h2("3.2 Evaluation metrics")
    R.bullets(["**RMSE** (days): square root of mean squared error; penalises large misses, so it is the main metric because very late parcels are the costly ones.",
               "**MAE** (days): average absolute miss; easy to explain to operations ('wrong by about 0.4 day').",
               "**R-squared:** share of variance explained, for comparing with the mean predictor. For classification: ROC-AUC, average precision and a gain curve, because the late class is a 30% minority and accuracy would be misleading."])
    R.h2("3.3 Results")
    rows = []
    for x in r["regression_table"]:
        cv = f"{x['cv_rmse']:.3f}" if x["cv_rmse"] == x["cv_rmse"] else "-"
        rows.append([x["model"], cv, f"{x['test_rmse']:.3f}", f"{x['test_mae']:.3f}", f"{x['test_r2']:.3f}"])
    R.table(["Model", "CV RMSE", "Test RMSE", "Test MAE", "Test R-squared"], rows, widths_cm=[6, 2.6, 2.6, 2.6, 3], caption="Delivery-time models: 5-fold CV on training data and hold-out test scores.", align_right_from=1)
    R.fig(FIG / "w4_model_comparison.png", "Hold-out RMSE by model (lower is better).", 5.8)
    R.p(f"Every learned model is far better than the mean predictor (R-squared about 0.89 vs 0.00). Linear regression (RMSE {lin['test_rmse']:.3f}) is already strong because distance and promised days carry most of the signal. "
        f"Tree ensembles add a small, consistent gain (random forest {reg['Random forest']['test_rmse']:.3f}, boosting {reg['Gradient boosting (default)']['test_rmse']:.3f}). Tuning brings RMSE to {best['test_rmse']:.3f}. "
        f"Compared with today's promised days (RMSE {base['test_rmse']:.3f}) the final model reduces error by {100*(1-best['test_rmse']/base['test_rmse']):.0f}%. The differences among the ensemble models are small relative to the cross-validation spread "
        f"(about {reg['Random forest']['cv_rmse_sd']:.3f}), so the choice among them should also weigh simplicity: linear regression is a defensible production fallback.")
    R.h2("3.4 Cross-validation and hyper-parameter tuning")
    bp = r["best_params"]
    R.p(f"The best ensemble family in CV was tuned with randomised search ({r['tuning']['n_iter']} candidate settings x {r['tuning']['cv_folds']}-fold CV, scored on RMSE). Selected parameters: "
        + ", ".join(f"{k} = {v}" for k, v in bp.items()) + ". The chosen settings are conservative (shallow trees, strong regularisation), which suits a dataset where the signal is mostly smooth.")
    R.code(grab(SCRIPT, "if best_family == ", "search.fit(X_tr, y_tr)", 20) + "\nsearch.fit(X_tr, y_tr)", "Code 3. Randomised hyper-parameter search")
    ts = r["time_series_cv_rmse"]
    R.p(f"Forward-chaining CV (train on earlier months, validate on the next block) gives RMSE of {', '.join(f'{v:.2f}' for v in ts)} over five folds (mean {r['time_series_cv_rmse_mean']:.3f}). "
        "Error rises in the later folds, which cover the festive peak: the model has seen fewer festive-season examples when it predicts them, which is a warning that it should be retrained before each peak and monitored for drift.")
    R.fig(FIG / "w4_pred_vs_actual.png", "Predicted vs actual delivery days (left) and residual distribution (right) on the hold-out set.", 6.2)
    R.p(f"Residuals are centred near zero (mean {res['mean']:.3f} day, standard deviation {res['sd']:.2f}); {100*res['share_within_0_5d']:.0f}% of predictions are within half a day and {100*res['share_within_1d']:.0f}% within one day.")
    R.h2("3.5 What drives delivery time?")
    R.fig(FIG / "w4_feature_importance.png", "Permutation importance (increase in RMSE when a feature is shuffled).", 5.4)
    top = list(imp.items())[:5]
    R.p("Permutation importance on the test set shows " + ", ".join(f"{k} ({v:.2f})" for k, v in top) + ". Planned days and distance dominate; service level and destination zone come next. Weather and traffic add small but positive information. "
        "Carrier, weight and order value matter little for transit time, although carrier matters for lateness and cost.")

    R.h1("4. Late-delivery risk model")
    R.table(["Model", "CV AUC", "Test AUC", "Test avg. precision", "F1 at 0.5"],
            [[x["model"], f"{x['cv_auc']:.3f}", f"{x['test_auc']:.3f}", f"{x['test_avg_precision']:.3f}", f"{x['test_f1_at_0.5']:.3f}"] for x in r["classification_table"]],
            widths_cm=[5.4, 2.4, 2.4, 3.8, 2.8], caption=f"Late-risk classifiers (base rate {100*bc['base_rate']:.0f}%).", align_right_from=1)
    R.fig(FIG / "w4_gain_curve.png", "Gain curve: share of late deliveries caught versus share of shipments reviewed.", 5.0)
    R.p(f"An AUC of {bc['test_auc']:.2f} is moderate: lateness depends on many small and partly random causes, so no model can be near-perfect. It is nevertheless operationally useful. Reviewing the riskiest 20% of shipments "
        f"catches {g['top20_captures_pct']:.0f}% of the late ones, with {g['top20_precision_pct']:.0f}% of reviewed shipments actually late against {g['base_rate_pct']:.0f}% at random, about twice as efficient. "
        "Reviewing the top 30% catches 54%. The default 0.5 threshold gives a low F1 because late shipments are a minority; in practice the review threshold should be set from the capacity of the control tower rather than at 0.5.")

    R.h1("5. Optimisation strategies")
    R.h2("5.1 Warehouse-to-hub allocation (transportation linear programme)")
    R.p("Decision: what share of each hub's demand should each warehouse serve? Variables x[w,h] are shipments sent from warehouse w to hub h. Objective: minimise the sum of x[w,h] x (mean delivery cost + penalty x late rate) "
        f"with an assumed late-delivery penalty of INR {lp['penalty_inr']:.0f}. Constraints: every hub's demand is met exactly; each warehouse can absorb at most {100*(lp['capacity_factor']-1):.0f}% more than its current volume; "
        "optionally, total expected late deliveries may not exceed today's, and each lane must keep a minimum share of today's volume (so the network does not change overnight). The model is solved with SciPy's HiGHS solver (Huangfu & Hall, 2018).")
    R.code(grab(SCRIPT, "def solve_alloc", "SCEN = [", 28), "Code 4. Transportation LP with service-protection and lane-stickiness constraints")
    rows = [[s["scenario"], f"{s['delivery_cost_per_shipment']:.0f}", f"{s['late_pct']:.1f}%", f"{s['reassigned_pct']:.0f}%", f"{s['delivery_cost_saving_pct']:.1f}%"] for s in lp["scenarios"]]
    R.table(["Scenario", "Cost / shipment (INR)", "Late rate", "Volume moved", "Cost saving"], rows, widths_cm=[6.4, 3.2, 2.2, 2.6, 2.4],
            caption=f"Warehouse-allocation scenarios (baseline: INR {lp['baseline_delivery_cost']:.0f} per shipment, {lp['baseline_late_pct']:.1f}% late).", align_right_from=1)
    R.fig(FIG / "w4_lp_allocation.png", "Share of each hub's volume served by each warehouse: today versus the practical optimised plan.", 6.4)
    R.p(f"The cost-only solution saves {lp['scenarios'][0]['delivery_cost_saving_pct']:.0f}% but raises the late rate to {lp['scenarios'][0]['late_pct']:.1f}%, which shows why a service constraint is essential. "
        f"The practical plan, which protects the late rate and moves at most half of any lane, saves **{main['delivery_cost_saving_pct']:.1f}%** while re-routing {main['reassigned_pct']:.0f}% of volume; the conservative plan saves {lp['scenarios'][3]['delivery_cost_saving_pct']:.1f}% with {lp['scenarios'][3]['reassigned_pct']:.0f}% moved. "
        "Biggest moves: " + "; ".join(f"{m['hub']} from {m['gain_warehouse']} (+{m['gain_pp']:.0f} pp, taken from {m['lose_warehouse']})" for m in lp["top_moves"][:3]) + ". "
        "The late penalty did not change the practical result for penalties of INR 0, 150 and 400, because the service-protection constraint, not the penalty, is what binds.")
    R.callout("Caveat.", "Lane costs and late rates are historical averages and are assumed to stay constant when volume moves. Real tariffs, congestion and warehouse stock may react to a 30% shift in flows, "
              "so these savings are an upper-bound planning estimate to be validated in a pilot on a few hubs.")
    R.h2("5.2 Carrier mix per destination zone")
    R.p("For each zone, a small LP chooses carrier shares to minimise mean delivery cost, subject to the zone's expected late count not exceeding today's and each carrier keeping between 10% and 45% of the zone's volume (to avoid dependence on one carrier).")
    rows = [[t["zone"], f"{t['n']:,}", f"{t['baseline']:.0f}", f"{t['optimised']:.0f}", f"{t['saving_pct']:.1f}%", f"{t['late_pct_before']:.1f}% to {t['late_pct_after']:.1f}%"] for t in cz["table"]]
    R.table(["Zone", "Shipments", "Cost now (INR)", "Cost optimised (INR)", "Saving", "Late rate"], rows, widths_cm=[2.8, 2.2, 2.8, 3.4, 2.0, 3.6], caption="Carrier-mix optimisation by zone.", align_right_from=1)
    R.p("Savings are modest (about 1% to 4%) because carriers are already fairly balanced and the bounds limit change; the North-East gains most because its cost is highest. Carrier mix is therefore a secondary lever; the allocation of warehouses to hubs is the primary one.")
    R.h2("5.3 Data-driven delivery promises")
    R.p("Today the promised date is a distance-and-mode rule. Instead, quantile gradient-boosting models predict the q-th percentile of delivery time for each shipment and the promise is that value rounded up. Raising q buys reliability with longer promises; the question is how efficiently.")
    R.fig(FIG / "w4_promise_frontier.png", "On-time rate versus average promised days: per-shipment quantile promise versus a flat buffer on the current rule.", 5.4)
    R.table(["Policy", "Avg promise (days)", "On-time"], [["Current rule", f"{pr['current_avg_promise']:.2f}", f"{pr['current_on_time_pct']:.1f}%"]] +
            [[f"Flat +{b['buffer_days']} day", f"{b['avg_promised_days']:.2f}", f"{b['on_time_pct']:.1f}%"] for b in pr["buffer_frontier"][1:]] +
            [[f"Quantile q{int(round(q['quantile']*100))}", f"{q['avg_promised_days']:.2f}", f"{q['on_time_pct']:.1f}%"] for q in pr["frontier"] if int(round(q["quantile"] * 100)) in (50, 70, 80, 90)],
            widths_cm=[5, 4, 3], caption="Promise policies on the hold-out set.", align_right_from=1)
    R.p(f"At the same average promise length as today ({pr['current_avg_promise']:.2f} days) the model gives {pr['on_time_at_same_length_pct']:.1f}% on time, essentially the same as the current rule ({pr['current_on_time_pct']:.1f}%): "
        f"better predictions do not create reliability for free. The gain appears when targeting a service level. To reach 90% on time, the per-shipment promise needs {pr['avg_promise_for_90pct']:.2f} days on average, whereas a flat buffer on the current rule needs {pr['avg_promise_for_90pct_flat_buffer']:.2f} days, "
        f"about {pr['avg_promise_for_90pct_flat_buffer']-pr['avg_promise_for_90pct']:.2f} day shorter for the same reliability, because the model gives long, risky lanes more time and short, safe lanes less.")

    R.h1("6. Final recommendations")
    R.table(["#", "Recommendation", "Evidence", "Expected effect"], [
        ["1", "Replace the rule-based promise with a quantile model targeting 85-90% on-time", f"Frontier in Section 5.3", f"90% on-time with about {pr['avg_promise_for_90pct_flat_buffer']-pr['avg_promise_for_90pct']:.1f} day shorter average promise than a flat buffer"],
        ["2", "Pilot the practical warehouse-allocation plan on 2-3 hubs (Guwahati, Chennai, Lucknow first)", "Section 5.1", f"Up to {main['delivery_cost_saving_pct']:.0f}% lower delivery cost with no rise in late rate (to be validated)"],
        ["3", "Score every shipment for late risk at dispatch and review the top 20%", "Section 4", f"Catch {g['top20_captures_pct']:.0f}% of late shipments with 20% of the effort"],
        ["4", "Re-balance carrier mix by zone within 10-45% bounds", "Section 5.2", "1-4% cost saving at constant service"],
        ["5", "Retrain monthly and before each peak; monitor RMSE and AUC", "Time-series CV, Section 3.4", "Prevents drift-related degradation"]],
        widths_cm=[0.8, 6.2, 3.4, 6.4], caption="Prioritised actions.", font=8.5)
    R.h2("Limitations")
    R.bullets(["Simulated data: effect sizes illustrate the method; they are not forecasts for a real company.",
               "Delivery time is dominated by distance and promised days, so absolute R-squared is high and the improvement over simple baselines is small.",
               "LP savings assume stable lane costs and late rates; the late penalty is an assumption.",
               "A random split was used for the main test; the forward-chaining CV shows error rises in peak months."])
    R.references(refs("breiman", "friedman", "pedregosa", "koenker", "toth", "dantzig", "huangfu", "virtanen", "christopher", "chopra", "hunter"))
    return R
