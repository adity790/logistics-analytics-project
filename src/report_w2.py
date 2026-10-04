"""Week 2 report: data collection, cleaning and preprocessing."""
from __future__ import annotations

from docx_helpers import Report
from reports_common import AUTHOR, FIG, MONTH_YEAR, f, grab, load_json, refs

SCRIPT = "02_cleaning_pipeline.py"
HDR = "# ------------------------------------------------------------------ "


def build() -> Report:
    r = load_json("week2_results.json")
    pf, log, mc, bench = r["profile"], r["cleaning_log"], r["outlier_method_counts"], r["imputation_benchmark_mae_kg"]
    sb, sa, norm, skew = r["scorecard_before"], r["scorecard_after"], r["normalisation_table"], r["skew_table"]
    mech = r["missingness_mechanism"]
    best_imp = min(bench, key=bench.get)

    R = Report("Week 2 | Data Collection, Cleaning and Preprocessing for Logistics Analysis")
    R.title_page("Logistics Data Science Internship - Week 2", "Data Collection, Cleaning and Preprocessing",
                 "A reproducible pandas pipeline for a pan-India shipment feed", AUTHOR,
                 [("Report date", MONTH_YEAR), ("Reference datasets", "DataCo Smart Supply Chain, Olist Brazilian E-Commerce (field-aligned simulation used here)"),
                  ("Tools", "Python 3 (pandas, NumPy, SciPy, scikit-learn, Matplotlib)"), ("Script", "src/02_cleaning_pipeline.py")])

    R.h1("1. Executive summary")
    R.p(f"The raw shipment feed contains {pf['rows']:,} rows and {pf['columns']} columns, and it is deliberately messy in the ways real logistics extracts are: "
        f"{pf['duplicate_ids']} repeated shipment IDs, {pf['missing_cells']:,} missing cells, {pf['mode_label_variants']} spellings of 3 shipping modes, three date layouts, "
        f"negative costs, zero distances and extreme weights and costs. A ten-step pipeline cleans it into {r['rows_clean']:,} analysis-ready rows ({r['rows_removed']} removed, "
        f"{100*r['rows_removed']/r['rows_raw']:.1f}%). Data completeness rises from {sb['completeness_pct']}% to {sa['completeness_pct']}%, invalid values fall from {sb['invalid_values']} to {sa['invalid_values']}, "
        f"and duplicates from {sb['duplicate_rows']} to {sa['duplicate_rows']}.")
    R.callout("Key decisions.", f"(1) Outliers are detected with a robust z-score on the log scale within product category or shipping mode, because global IQR and z-score rules flag "
              f"{mc['weight_kg']['IQR (raw, global)']:,} and {mc['weight_kg']['z-score |z|>3 (raw, global)']} weights respectively while the robust rule flags {mc['weight_kg']['Robust z>4 (log, per category)']}. "
              f"(2) Missing weights are imputed from volume and category density after a masked-value benchmark of five methods. (3) The target variable (actual_days) is never imputed. "
              f"(4) Scaling is applied after a log transform for skewed fields and is always fitted inside cross-validation folds in Week 4 to avoid leakage.")

    R.h1("2. Data collection simulation")
    R.h2("2.1 Reference datasets")
    R.p("The workflow mirrors two public logistics datasets. DataCo Smart Supply Chain (Constante et al., 2019) has about 180,000 orders with scheduled and real shipping days, shipping mode, "
        "region and a late-delivery flag. Olist's Brazilian E-Commerce data (Olist, 2018) has about 100,000 orders with estimated and actual delivery dates, freight value and product weight. "
        "Because those files are large and licensed, this project simulates a feed with matching fields and realistic defects, so the full pipeline can be run offline and reproduced exactly (seed 42).")
    R.h2("2.2 Characteristics of the raw feed")
    R.table(["Property", "Value"], [
        ["Rows / columns", f"{pf['rows']:,} / {pf['columns']}"],
        ["Period", f"{r['date_range'][0]} to {r['date_range'][1]}"],
        ["Grain", "One row per shipment (shipment_id), 5 warehouses, 8 destination hubs, 6 zones, 4 carriers"],
        ["Numeric fields", "distance_km, weight_kg, volume_m3, order_value_inr, weather_index, traffic_index, planned_days, actual_days, delivery_cost_inr, customer_rating"],
        ["Categorical fields", "origin_warehouse, destination_hub, destination_zone, product_category, shipping_mode, priority, carrier"],
        ["Delivery target", "actual_days (continuous) and delayed = actual_days > planned_days (binary, derived)"],
        ["Missing cells", f"{pf['missing_cells']:,} ({100*pf['missing_cells']/(pf['rows']*pf['columns']):.2f}% of all cells)"],
    ], widths_cm=[3.6, 13.2], caption="Raw dataset profile.")
    R.h2("2.3 Data quality issues found")
    miss = pf["missing_pct_by_col"]
    rows = [[k, f"{v:.2f}%"] for k, v in sorted(miss.items(), key=lambda kv: -kv[1]) if v > 0]
    R.table(["Column with missing values", "Share missing"], rows, widths_cm=[7, 4], caption="Missing values by column.", align_right_from=1)
    R.fig(FIG / "w2_missingness.png", "Missingness profile of the raw feed.", 5.6)
    R.table(["Issue", "Count"], [
        ["Duplicate shipment IDs (system retries)", pf["duplicate_ids"]], ["Negative delivery cost", pf["negative_cost"]], ["Zero distance between different cities", pf["zero_distance"]],
        ["Negative actual delivery days", pf["negative_days"]], ["Actual days above 30", pf["days_over_30"]], ["Spellings of 3 shipping modes", pf["mode_label_variants"]],
        ["Spellings of 4 carriers", pf["carrier_label_variants"]], ["Date layouts", "3 (ISO, DD/MM/YYYY, 'Mon DD, YYYY')"]],
        widths_cm=[10, 6.8], caption="Other quality problems detected during profiling.", align_right_from=1)

    R.h1("3. Cleaning methodology and rationale")
    R.p("Each step below states what was done, why that technique was chosen, and how it affects later analysis. The pipeline is ordered deliberately: structural fixes first "
        "(duplicates, labels, dates), then impossible values, then statistical outliers, then imputation, then transformation.")
    R.h2("3.1 Duplicates, labels and dates")
    R.p("Duplicate IDs come from system retries; keeping the first occurrence avoids double counting volume and cost. Labels such as 'express', 'EXPRESS ' and 'Express' are one category; "
        "left alone they would split into nine groups and bias every group-by KPI. Dates are parsed with explicit formats rather than guessing, because 03/04/2025 is ambiguous between 3 April and 4 March "
        "and a silent wrong guess would corrupt the seasonality analysis.")
    R.code(grab(SCRIPT, "dups = int", HDR + "3. Text") + "\n" + grab(SCRIPT, "df[\"shipping_mode\"] = df", "carrier_before"), "Code 1. Duplicates and label standardisation")
    R.h2("3.2 Impossible values")
    R.p("Values that break physical or business rules are treated differently from statistical outliers. A negative cost is a sign-entry error with a plausible magnitude, so the absolute value is kept. "
        "A zero distance between two different cities is missing information and is re-derived from the lane median. Non-positive or over-30-day actual_days is removed: the target variable must not be imputed, "
        "because any imputed target would simply teach a model the imputation rule.")
    R.h2("3.3 Outlier detection: comparing three methods")
    R.table(["Method", "Weight flags", "Cost flags"], [
        ["IQR rule (1.5 x IQR), global, raw scale", f"{mc['weight_kg']['IQR (raw, global)']:,}", f"{mc['delivery_cost_inr']['IQR (raw, global)']:,}"],
        ["Z-score |z| > 3, global, raw scale", f"{mc['weight_kg']['z-score |z|>3 (raw, global)']:,}", f"{mc['delivery_cost_inr']['z-score |z|>3 (raw, global)']:,}"],
        ["Robust z > 4 (median/MAD), log scale, within group", f"{mc['weight_kg']['Robust z>4 (log, per category)']:,}", f"{mc['delivery_cost_inr']['Robust z>4 (log cost/km, per mode)']:,}"],
    ], widths_cm=[9.5, 3.5, 3.5], caption="Number of rows flagged by each outlier rule.", align_right_from=1)
    R.p("The global IQR rule flags over a thousand weights because parcel weight is strongly right-skewed (skewness 4.4): heavy furniture is normal, not an error. The z-score rule is distorted "
        "because the mean and standard deviation are themselves pulled by the extremes (Leys et al., 2013). The chosen rule works on the log scale, uses the median and MAD, and compares each parcel with others of "
        "the same category (or each cost per km with others of the same shipping mode). It flags only values that are extreme relative to a sensible peer group. A 200 kg carrier limit is added as a business rule.")
    R.fig(FIG / "w2_outliers_before_after.png", "Weight and cost distributions before and after outlier treatment (log x-axis).", 6.0)
    R.code(grab(SCRIPT, "def robust_z", "# figure: before/after", 36), "Code 2. Outlier rules")
    R.p("The effect on analysis is large. In Week 1 a regression of cost on distance had R-squared 0.07 before trimming the extreme costs and 0.67 after: a handful of billing errors was hiding a strong relationship.")
    R.h2("3.4 Missing values")
    R.p(f"Before choosing a method, the missingness mechanism was checked. A chi-square test of 'weight is missing' against product category gives p = {mech['chi2_p_value_weight_missing_vs_category']:.3f}, so there is no evidence "
        "that missingness depends on category; it is plausibly missing completely at random (Little & Rubin, 2019), which means simple imputation will not introduce systematic bias. "
        "Dropping every row with any missing cell would have discarded about 20% of the data (2,445 of 12,160 rows), so imputation is preferable.")
    R.p("To pick a method for weight objectively, 10% of known weights were masked and five imputers were scored by mean absolute error:")
    rows = [[k, f"{v:.2f}", "chosen" if k == "Volume x category density" else ("lowest error" if k == best_imp else "")] for k, v in sorted(bench.items(), key=lambda kv: kv[1])]
    R.table(["Method", "MAE (kg)", "Note"], rows, widths_cm=[7, 3, 4], caption="Imputation benchmark for weight_kg (lower is better).", align_right_from=1)
    R.fig(FIG / "w2_imputation_benchmark.png", "Imputation error on masked weights.", 5.4)
    R.p(f"KNN is slightly more accurate (MAE {bench[best_imp]:.2f} kg) but the volume x category density rule ({bench['Volume x category density']:.2f} kg) is nearly as good, needs no tuning and can be explained to an operations manager in one sentence "
        "(weight = volume x typical density of that product type). It was therefore chosen. Mean and median imputation are clearly worse and also shrink the variance of the variable. "
        "Other fields: distance from the lane median, weather and traffic from the same calendar month median, carrier as an explicit 'Unknown' category (it cannot be inferred honestly), customer rating left missing and excluded only from rating statistics.")
    R.code(grab(SCRIPT, "# 7b. Benchmark", "fig, ax = plt.subplots(figsize=(7.2, 3.3))", 40), "Code 3. Benchmarking imputation methods on masked values")
    R.h2("3.5 Skewness and normalisation")
    rows = [[k, f"{v['skew_raw']:.2f}", f"{v['skew_log1p']:.2f}"] for k, v in skew.items()]
    R.table(["Variable", "Skewness (raw)", "Skewness (log1p)"], rows, widths_cm=[6, 4, 4], caption="Skewness before and after a log transform.", align_right_from=1)
    R.p("Right-skewed money and weight variables are log-transformed first (skewness of order value falls from 2.80 to 0.32, of weight from 4.44 to 0.75). "
        "Scaling alone does not change the shape of a distribution, as the table below shows: min-max, z-score and robust scaling all keep the skewness of 4.44. Min-max scaling is also very sensitive to the single largest value.")
    rows = [[k, f"{v['mean']:.2f}", f"{v['std']:.2f}", f"{v['min']:.2f}", f"{v['max']:.2f}", f"{v['skew']:.2f}"] for k, v in norm.items()]
    R.table(["Transformation", "Mean", "Std", "Min", "Max", "Skew"], rows, widths_cm=[5, 2.4, 2.4, 2.4, 2.4, 2.2], caption="Effect of each normalisation on weight_kg.", align_right_from=1)
    R.fig(FIG / "w2_normalisation.png", "Histograms of parcel weight under six transformations.", 6.0)
    R.p("Guidance used: log1p + z-score for linear and distance-based methods (regression, K-Means, KNN); no scaling for tree ensembles, which are invariant to monotone transformations; "
        "and always fit the scaler on training folds only. The scaled file in data/processed is for demonstration; Week 4 refits scaling inside a scikit-learn Pipeline.")
    R.code(grab(SCRIPT, "scaled_variants = {", "norm_table = {", 12), "Code 4. Normalisation variants compared")

    R.h1("4. Cleaning log and before/after scorecard")
    rows = [[x["step"], x["issue"], f"{x['rows_affected']:,}", x["action"]] for x in log]
    R.table(["Step", "Issue", "Rows", "Action"], rows, widths_cm=[2.6, 5.8, 1.4, 7.0], caption="Complete cleaning log (also saved to results/week2_results.json).", font=8)
    R.table(["Metric", "Before", "After"], [
        ["Rows", f"{sb['rows']:,}", f"{sa['rows']:,}"], ["Completeness of core columns", f"{sb['completeness_pct']}%", f"{sa['completeness_pct']}%"],
        ["Duplicate IDs", sb["duplicate_rows"], sa["duplicate_rows"]], ["Invalid values", sb["invalid_values"], sa["invalid_values"]],
        ["Shipping-mode spellings", sb["label_variants_shipping_mode"], sa["label_variants_shipping_mode"]]],
        widths_cm=[7, 4, 4], caption="Data-quality scorecard.", align_right_from=1)
    R.p("Completeness after cleaning is below 100% only because customer ratings were intentionally left missing.")

    R.h1("5. Reflection: why data quality matters for logistics decisions")
    R.bullets([
        "**KPIs.** Duplicate IDs inflate volume; mixed labels split carriers and modes; negative costs and zero distances corrupt cost per km. Every KPI in the Week 1 scorecard depends on these fixes.",
        "**Models.** A few extreme costs reduced a cost-distance R-squared from 0.67 to 0.07. A model trained on such data would mis-price lanes and mis-allocate volume.",
        "**Decisions.** Misread carrier performance leads to wrong contract decisions; a mis-dated peak season leads to wrong staffing and inventory buffers.",
        "**Trust.** A documented cleaning log lets managers audit every change; the pipeline is reproducible, so the same data gives the same result on any machine.",
        "**Limits.** Simulated defects are cleaner than real ones. In production the pipeline should add schema checks, range alerts and drift monitoring at ingestion."])
    R.references(refs("dataco", "olist", "mckinney", "pedregosa", "virtanen", "tukey", "little", "leys", "christopher"))
    return R
