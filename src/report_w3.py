"""Week 3 report: advanced data analysis and visualisation."""
from __future__ import annotations

from docx_helpers import Report
from reports_common import AUTHOR, FIG, MONTH_YEAR, f, grab, load_json, refs

SCRIPT = "03_eda_visualisation.py"


def build() -> Report:
    r = load_json("week3_results.json")
    ov, ins = r["overall"], r["insights"]
    ps, ne, cd, rd = ins["peak_season"], ins["north_east"], ins["cost_distance"], ins["rating_by_delay"]
    wt = ins["weather_traffic"]

    R = Report("Week 3 | Advanced Data Analysis and Visualization in Logistics")
    R.title_page("Logistics Data Science Internship - Week 3", "Advanced Data Analysis and Visualization",
                 "Exploratory analysis of delivery performance, cost drivers and bottlenecks", AUTHOR,
                 [("Report date", MONTH_YEAR), ("Dataset", f"Cleaned shipment feed, {ov['shipments']:,} shipments, Jan-Dec 2025 (output of Week 2)"),
                  ("Tools", "Python 3 (pandas, SciPy, Matplotlib)"), ("Script", "src/03_eda_visualisation.py")])

    R.h1("1. Executive summary")
    R.p(f"Across {ov['shipments']:,} cleaned shipments, {ov['on_time_pct']}% arrive on time, the mean lead time is {ov['mean_actual_days']} days (median {ov['median_actual_days']}) and total freight spend is INR {ov['total_cost_inr']/1e6:.2f} million. "
        "The exploration answers three management questions: where is efficiency lost, what drives cost, and where are the bottlenecks.")
    R.bullets([
        f"**Bottleneck 1 - North-East.** Only {ne['ne_share_of_shipments_pct']}% of shipments but {ne['ne_share_of_late_pct']}% of all late deliveries; late rate {ne['ne_late_pct']}% against {ne['rest_late_pct']}% elsewhere. All five Guwahati lanes are the worst in the network.",
        f"**Bottleneck 2 - festive peak.** October to December volume is {ps['volume_uplift_pct']}% higher per month and the late rate rises from {ps['offpeak_delay_pct']}% to {ps['peak_delay_pct']}% (two-proportion z = {ps['z']}, p < 0.001).",
        f"**Cost driver.** Distance dominates cost (Spearman rho = {cd['spearman_rho']}); service level is the second driver, with Express costing about twice as much per km as Economy. Short hauls cost more per km (INR {cd['short_haul_cpk']}) than long hauls (INR {cd['long_haul_cpk']}) because fixed charges are spread over fewer km.",
        f"**Carrier.** RoadRunner is cheapest per km but the least reliable ({ins['carrier']['table']['RoadRunner']['late_pct']}% late); BlueDart Lite is the most reliable ({ins['carrier']['table']['BlueDart Lite']['late_pct']}% late) but the most expensive per km.",
        f"**Customer impact.** Mean rating falls from {rd['On time']['mean_rating']} for on-time parcels to {rd['2+ d late']['mean_rating']} for parcels two or more days late."])

    R.h1("2. Dataset and variables")
    R.p("The input is data/processed/logistics_shipments_clean.csv, produced by the Week 2 pipeline. Each row is one shipment. Derived fields used in this week are delay_days = max(actual - planned, 0), "
        "delayed (1 if delay_days > 0), cost_per_km, is_peak_season (October to December) and order_month.")
    R.table(["Group", "Variables"], [
        ["Delivery time", "planned_days, actual_days, delay_days, delayed"],
        ["Shipment volume", "weight_kg, volume_m3, order_value_inr, monthly shipment count"],
        ["Transport cost", "delivery_cost_inr, cost_per_km, distance_km, shipping_mode, carrier"],
        ["Context", "weather_index, traffic_index (0 to 1), is_peak_season, destination_zone, origin_warehouse"],
        ["Outcome", "customer_rating (1 to 5)"]], widths_cm=[3.6, 13.2], caption="Variable groups.")

    R.h1("3. Exploratory data analysis")
    R.h2("3.1 Central tendency and spread")
    rows = [[d["variable"], f"{d['mean']:,.2f}", f"{d['median']:,.2f}", f"{d['std']:,.2f}", f"{d['min']:,.2f}", f"{d['max']:,.2f}", f"{d['skew']:.2f}", f"{d['IQR']:,.2f}"] for d in r["describe"]]
    R.table(["Variable", "Mean", "Median", "Std", "Min", "Max", "Skew", "IQR"], rows, widths_cm=[3.6, 2.1, 2.1, 2.1, 1.9, 2.2, 1.4, 1.9],
            caption="Descriptive statistics of the cleaned data.", font=8, align_right_from=1)
    R.p("Three patterns matter. Weight (skew 4.4), order value (2.8) and delay_days (4.3) are strongly right-skewed: the median is far below the mean, so medians are the right summary and log scales the right axes. "
        "Delay is zero for the median shipment, so the mean delay of 0.17 days hides a minority of costly failures. Customer rating is left-skewed (median 5.0, mean 4.6): most customers are satisfied and the interesting signal is in the tail.")
    R.code(grab(SCRIPT, "cols = [\"distance_km\"", "overall = {", 8), "Code 1. Descriptive statistics")
    R.h2("3.2 Distributions")
    R.fig(FIG / "w3_distributions.png", "Distributions of key variables.", 6.2)
    R.p("Histograms were chosen because they show shape, skew and outliers at a glance, which is what decides the choice of summary statistic and transformation. Distance is broad because lanes range from 270 to 3,000 km; "
        "weight and order value need a long right tail; actual days are right-skewed with a floor near one day.")
    R.h2("3.3 Correlation")
    R.fig(FIG / "w3_correlation.png", "Spearman correlation matrix.", 5.6)
    R.p("A heatmap of Spearman rank correlations was used instead of Pearson because most variables are skewed or bounded; ranks are robust to that. The strongest pairs:")
    R.table(["Variable A", "Variable B", "Spearman rho"], [[p["a"], p["b"], f"{p['rho']:.2f}"] for p in r["top_corr_pairs"]], widths_cm=[5, 5, 3], caption="Strongest correlations.", align_right_from=2)
    R.p("Interpretation: distance explains most of delivery time (0.83), planned days (0.87) and cost (0.87), which is expected. Delay is only weakly related to any single factor (actual days 0.28, traffic 0.14, peak season 0.13, weather 0.11), "
        "so lateness is a many-small-causes phenomenon and a multivariate model (Week 4) is needed. Delay is negatively related to customer rating (-0.24). Correlation is not causation: traffic and peak season are themselves correlated.")
    R.code(grab(SCRIPT, "corr = df[corr_cols]", "fig, ax = plt.subplots(figsize=(7.6, 6.2))", 3) + "\n...\n" + grab(SCRIPT, "tri = corr.where", "delay_corr = ", 5), "Code 2. Correlation matrix and ranking of pairs")

    R.h1("4. Visual analysis and insights")
    R.h2("4.1 Seasonality: volume and lateness by month")
    R.fig(FIG / "w3_monthly_trend.png", "Monthly shipments (top) and late-delivery rate (bottom).", 5.8)
    R.p(f"Why a bar chart plus a line: volume is a count per category (bars), while the late rate is a rate that is read as a trend (line). Stacking them on a shared month axis lets the reader see that lateness follows volume. "
        f"Peak months carry {ps['volume_uplift_pct']}% more volume and {ps['peak_delay_pct']}% late against {ps['offpeak_delay_pct']}% off-peak. **Recommendation:** open temporary capacity and add a 1-day buffer to promises from October.")
    R.h2("4.2 Where lateness concentrates: carrier x zone")
    R.fig(FIG / "w3_delay_heatmap.png", "Late-delivery rate by carrier and destination zone.", 5.8)
    z = ins["zone"]
    R.p(f"A heatmap suits a two-way categorical comparison with one numeric outcome. The North-East column is dark for every carrier ({z['North-East']['late_pct']}% late on average, {z['North-East']['mean_days']} days), which shows that the problem is the zone, not a single carrier. "
        f"The West is the second weakest zone ({z['West']['late_pct']}%). Carrier differences are also real (chi-square p < 0.001): RoadRunner is {ins['carrier']['table']['RoadRunner']['late_pct']}% late against {ins['carrier']['table']['BlueDart Lite']['late_pct']}% for BlueDart Lite.")
    rows = [[k, f"{v['shipments']:,}", f"{v['late_pct']}%", f"{v['median_cost_per_km']}", f"{v['mean_days']}", f"{v['mean_rating']}"] for k, v in sorted(ins["carrier"]["table"].items(), key=lambda kv: kv[1]["late_pct"])]
    R.table(["Carrier", "Shipments", "Late", "Median cost/km", "Mean days", "Rating"], rows, widths_cm=[3.4, 2.4, 2, 3.2, 2.4, 2], caption="Carrier scorecard on cleaned data.", align_right_from=1)
    R.code(grab(SCRIPT, "hm = df.pivot_table", "cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03); cb.set_label(\"Late deliveries (%)\"", 10), "Code 3. Carrier x zone heatmap")
    R.h2("4.3 Cost drivers")
    R.fig(FIG / "w3_cost_drivers.png", "Cost against distance and cost per km by service level.", 6.2)
    md = ins["mode"]
    R.p(f"A scatter plot shows the dominant, near-linear distance effect and the spread around it; a box plot of cost per km by mode isolates the service effect. Median cost per km is INR {md['Economy']['median_cost_per_km']} for Economy, "
        f"{md['Standard']['median_cost_per_km']} for Standard and {md['Express']['median_cost_per_km']} for Express (Kruskal-Wallis p < 0.001). Express saves {md['Economy']['mean_days']-md['Express']['mean_days']:.1f} days on average over Economy "
        f"but costs INR {md['Express']['mean_cost_inr']:.0f} per parcel against INR {md['Economy']['mean_cost_inr']:.0f}, and its late rate ({md['Express']['late_pct']}%) is only marginally better than Economy ({md['Economy']['late_pct']}%). "
        "**Recommendation:** reserve Express for high-value or priority orders and stop using it as a default for lanes where it does not improve reliability.")
    R.h2("4.4 Bottleneck lanes")
    R.fig(FIG / "w3_bottleneck_lanes.png", "Lanes with the highest late-delivery rates (minimum volume applied).", 5.6)
    rows = [[w["lane"], f"{w['shipments']:,}", f"{w['late_pct']}%"] for w in ins["worst_lanes"]] + [[b["lane"] + " (best)", f"{b['shipments']:,}", f"{b['late_pct']}%"] for b in ins["best_lanes"]]
    R.table(["Lane", "Shipments", "Late rate"], rows, widths_cm=[7, 3, 3], caption="Five worst and three best lanes.", align_right_from=1)
    R.p("A sorted horizontal bar chart is the clearest way to rank categories with long names. All five worst lanes end in Guwahati (41% to 54% late); the best lanes are about 14% to 16%. "
        "Delhi to Guwahati alone has 344 shipments, so a fix on that lane has real effect. **Recommendation:** serve Guwahati from Kolkata (the nearest origin) and pre-position stock there.")
    R.h2("4.5 Weather and traffic")
    R.fig(FIG / "w3_weather_traffic.png", "Late rate by weather severity and traffic congestion quartile.", 5.6)
    R.p(f"Quartile bars turn two continuous indices into an easy comparison. The late rate climbs from {wt['Weather severity'][0]}% in the calmest weather quartile to {wt['Weather severity'][3]}% in the worst, and from {wt['Traffic congestion'][0]}% to {wt['Traffic congestion'][3]}% for traffic. "
        f"Both effects are statistically significant but modest in correlation terms (Spearman rho {ins['weather_traffic_rho']['weather'][0]:.2f} for weather, {ins['weather_traffic_rho']['traffic'][0]:.2f} for traffic), so they should be inputs to the delivery-promise model rather than the sole explanation.")
    R.h2("4.6 Lateness and customer satisfaction")
    R.fig(FIG / "w3_rating_vs_delay.png", "Mean customer rating by lateness band.", 5.2)
    R.table(["Lateness band", "Shipments", "Mean rating"], [[k, f"{v['n']:,}", f"{v['mean_rating']:.2f}"] for k, v in rd.items()], widths_cm=[5, 3, 3], caption="Customer rating by lateness.", align_right_from=1)
    R.p("Ratings fall slowly for small delays and sharply for long ones (4.04 at one to two days late, 2.70 beyond two days). The practical lesson is that severity matters more than frequency: "
        "preventing the rare multi-day failures protects satisfaction far better than shaving small delays.")

    R.h1("5. Operational insights and recommendations")
    R.table(["#", "Finding", "Recommendation", "KPI affected"], [
        ["1", f"North-East: {ne['ne_late_pct']}% late, {ne['ne_share_of_late_pct']}% of all failures", "Dedicated lane carrier, longer promise, source from Kolkata", "On-time rate"],
        ["2", f"Peak season +{ps['volume_uplift_pct']}% volume, +{ps['peak_delay_pct']-ps['offpeak_delay_pct']:.1f} pp late", "Temporary capacity, +1 day promise from October", "On-time rate"],
        ["3", "Express costs 2x per km with little reliability gain", "Restrict Express to priority orders", "Cost per km"],
        ["4", "RoadRunner cheapest but 35.5% late", "Cap its share on high-risk lanes", "On-time rate, cost"],
        ["5", "Short hauls cost more per km", "Consolidate pickups, flat-rate tariffs", "Cost per km"],
        ["6", "Rating falls sharply beyond 2 days late", "Alert and compensate the long-delay tail first", "Customer rating"]],
        widths_cm=[0.8, 5.6, 6.6, 3.8], caption="Findings and actions.", font=8.5)
    R.p("Limitations: the data is simulated and relationships reflect the simulator; the findings are descriptive associations, not causal estimates. Week 4 turns the most important ones into a predictive and optimisation model.")
    R.references(refs("tukey", "hunter", "mckinney", "virtanen", "christopher", "chopra", "dataco", "olist"))
    return R
