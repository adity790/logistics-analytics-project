"""Week 1 report: Strategic planning and data exploration in logistics."""
from __future__ import annotations

from docx_helpers import Report
from reports_common import AUTHOR, FIG, MONTH_YEAR, RES, f, grab, load_json, refs

SCRIPT = "01_strategy_kpis_clustering.py"


def build() -> Report:
    r1 = load_json("week1_results.json")
    ov, zone, car, wh, mode = r1["overall"], r1["by_zone"], r1["by_carrier"], r1["by_warehouse"], r1["by_mode"]
    clus, inv, reg, sil, bk = r1["cluster_table"], r1["inventory"], r1["regression"], r1["silhouette"], r1["best_k"]
    ep = r1["eoq_params"]
    late_pct = round(100 - ov["on_time_rate_pct"], 1)
    worst_zone = min(zone, key=lambda k: zone[k]["on_time_rate_pct"])
    best_zone = max(zone, key=lambda k: zone[k]["on_time_rate_pct"])
    worst_car = min(car, key=lambda k: car[k]["on_time_rate_pct"])
    best_car = max(car, key=lambda k: car[k]["on_time_rate_pct"])

    R = Report("Week 1 | Strategic Planning and Data Exploration in Logistics")
    R.title_page("Logistics Data Science Internship - Week 1", "Strategic Planning and Data Exploration in Logistics",
                 "Scenario definition, KPI framework, literature and data research, analytical roadmap",
                 AUTHOR, [("Report date", MONTH_YEAR), ("Scenario", "Pan-India e-commerce fulfilment network (5 warehouses, 8 destination hubs)"),
                          ("Tools", "Python 3 (pandas, NumPy, SciPy, scikit-learn, Matplotlib)"),
                          ("Code & data", "Repository folder: src/, data/, figures/, results/")])

    # ------------------------------------------------------------------ 1
    R.h1("1. Executive summary")
    R.p(f"This report plans a data-science programme for a mid-sized Indian e-commerce fulfilment network. Five warehouses (Pune, Mumbai, Delhi, "
        f"Bengaluru, Kolkata) ship to eight destination hubs through four carriers and three service levels. A pilot extract of "
        f"**{ov['shipments']:,} shipments from 2025** already shows the problem the programme must solve: only **{ov['on_time_rate_pct']}%** of parcels arrive within the "
        f"promised window ({late_pct}% are late), delivery cost varies sharply by lane and service level, and the weakest destination zone "
        f"({worst_zone}) has an on-time rate of {zone[worst_zone]['on_time_rate_pct']}% against {zone[best_zone]['on_time_rate_pct']}% in {best_zone}.")
    R.p("The report (i) defines the scenario and six KPIs, (ii) grounds the plan in named public datasets and published methods, (iii) explains where "
        "regression, clustering, classification and optimisation fit, (iv) lays out an end-to-end Python roadmap, and (v) demonstrates the approach on the pilot data "
        "with working code: KPI computation, K-Means segmentation of delivery lanes, EOQ and safety-stock calculation, and a first cost-versus-distance regression.")
    R.callout("Headline findings from the pilot.",
              f"(1) Network on-time rate {ov['on_time_rate_pct']}% versus a proposed 90% target. (2) Lanes split into {bk} behaviour-based segments; the "
              f"highest-risk segment has a {max(c['delay_rate_pct'] for c in clus)}% late rate. (3) Cost rises almost linearly with distance, but the simple fit explains only part of the variation "
              f"(R-squared {reg['r2_simple']:.2f}) and billing outliers push the untrimmed fit down to {reg['r2_untrimmed']:.2f}, which is why Week 2 is devoted to data quality.")

    R.h2("Contents")
    R.bullets(["2. Project definition: the logistics scenario", "3. Key performance indicators", "4. Background research and public data sources",
               "5. How data-science concepts apply to this scenario", "6. Pilot exploration of the data", "7. Strategic roadmap",
               "8. Python code illustrations", "9. Assumptions, risks and limitations", "10. Conclusion: expected outcomes and decision impact", "References"])

    # ------------------------------------------------------------------ 2
    R.h1("2. Project definition: the logistics scenario")
    R.h2("2.1 Business context")
    R.p("The organisation is a simulated online retailer that fulfils orders for six product families (electronics, apparel, grocery, furniture, pharma, books) from five "
        "regional warehouses. Customers are served through eight destination hubs that stand in for the main demand centres: Ahmedabad, Jaipur, Lucknow, Hyderabad, Chennai, "
        "Nagpur, Bhubaneswar and Guwahati. Parcels travel by road with one of four carriers on Economy, Standard or Express service. Peak season runs October to December.")
    R.p("Three operational pain points motivate the work, and each maps to one of the three problem types named in the assignment:")
    R.bullets([
        "**Supply-chain integration / service reliability.** Promised delivery dates are set by a simple distance-and-mode rule and are missed on roughly three shipments in ten. "
        "Carrier, weather, traffic and season are not used in the promise.",
        "**Route and lane optimisation.** Dispatchers choose the origin warehouse for each order with little regard to total cost; some hubs are served from "
        "four or five warehouses at very different cost per kilometre.",
        "**Inventory management.** Each warehouse needs a reorder policy that reflects its demand variability and supplier lead time, otherwise stock-outs force "
        "costly transfers between warehouses."])
    R.h2("2.2 Business questions")
    R.bullets(["Which lanes, carriers, zones and months drive late deliveries, and by how much?",
               "How well can delivery time be predicted at the moment of dispatch, and can the promise be made smarter?",
               "Which warehouse should serve which hub, and which carrier mix should each zone use, to cut cost without hurting service?",
               "What reorder quantity and safety stock should each warehouse hold?"])
    R.h2("2.3 Scope and data used")
    R.p(f"A 12,000-shipment simulated feed (January to December 2025) was generated with a fixed random seed. It deliberately reproduces the structure and the quality problems of real "
        f"logistics extracts (Section 4.3). The Week 1 pilot applies only light cleaning (duplicate IDs removed, impossible values dropped, labels standardised), leaving "
        f"{r1['n_pilot_rows']:,} usable shipments across {r1['n_lanes']} origin-destination lanes. Full cleaning follows in Week 2.")

    # ------------------------------------------------------------------ 3
    R.h1("3. Key performance indicators")
    R.p("Six KPIs were selected to cover the service-cost-satisfaction triangle described in the logistics literature (Christopher, 2016; Chopra & Meindl, 2016). "
        "Baselines are computed on the pilot data; targets are proposals for the programme to test.")
    R.table(["KPI", "Definition", "Why it matters", "Pilot baseline", "Proposed target"], [
        ["On-time delivery rate (OTD)", "Shipments with actual_days <= planned_days / all shipments", "Primary service measure; drives rating and repeat orders", f"{ov['on_time_rate_pct']}%", ">= 90%"],
        ["Average lead time", "Mean of actual_days from order to delivery", "Speed of the network", f"{ov['avg_lead_time_days']} days", "<= 3.5 days"],
        ["Average delay when late", "Mean of (actual - planned) for late shipments", "Severity of failures, not just frequency", f"{ov['avg_delay_days_when_late']} days", "<= 0.4 days"],
        ["Cost per km (median)", "delivery_cost_inr / distance_km", "Transport efficiency; comparable across lanes", f"INR {ov['median_cost_per_km_inr']}", "-10%"],
        ["Cost per kg (median)", "delivery_cost_inr / weight_kg", "Pricing and load-density efficiency", f"INR {ov['median_cost_per_kg_inr']}", "-10%"],
        ["Customer rating", "Mean post-delivery rating, 1 to 5", "Outcome KPI linked to lateness", f"{ov['avg_customer_rating']}", ">= 4.7"],
    ], widths_cm=[3.4, 4.6, 4.2, 2.4, 2.2], caption="KPI framework with pilot baselines and proposed targets.", font=8.5)
    R.p("Inventory performance is tracked separately through fill rate (target 97%) and the reorder-point policy derived in Section 6.4.")

    # ------------------------------------------------------------------ 4
    R.h1("4. Background research and public data sources")
    R.h2("4.1 What the literature says")
    R.bullets([
        "**Service and cost trade-offs.** Christopher (2016) frames logistics as the management of the trade-off between service level and total cost, which is why this "
        "project always reports a service KPI (OTD) next to a cost KPI. Chopra and Meindl (2016) add that responsiveness and efficiency pull in opposite directions and must be balanced deliberately.",
        "**Network and lane decisions are optimisation problems.** The transportation and vehicle-routing family started with Dantzig and Ramser's truck-dispatching formulation (1959) "
        "and is surveyed in Toth and Vigo (2014). Allocating shipments from warehouses to hubs under capacity is the classic linear-programming case, solved in Week 4.",
        "**Inventory policies have closed forms.** Harris's economic order quantity (1913) and the safety-stock formulas in Silver, Pyke and Thomas (2017) give reorder policies "
        "from demand, variability and lead time; Section 6.4 applies them.",
        "**Tree ensembles are strong tabular baselines.** Random forests (Breiman, 2001) and gradient boosting (Friedman, 2001) are standard for predicting delivery time and "
        "late-delivery risk from mixed numeric and categorical features, and are implemented in scikit-learn (Pedregosa et al., 2011).",
        f"**National context.** India ranked 38th of 139 economies in the World Bank Logistics Performance Index 2023 (World Bank, 2023), and the National Logistics Policy "
        "(Government of India, 2022) sets out the aim of lowering logistics cost and improving reliability. Network analytics of the kind planned here is a direct "
        "contribution to those goals at company level."])
    R.h2("4.2 Public datasets that support the scenario")
    R.p("The following open datasets contain the same kinds of fields used in this project. They are the recommended real-world sources if the pipeline is moved from simulation to production research.")
    R.table(["Dataset", "Publisher / location", "Content", "Relevance to this project"], [
        ["DataCo Smart Supply Chain", "Constante, Silva & Pereira (2019), Mendeley Data; mirrored on Kaggle", "About 180,000 order records with shipping mode, scheduled vs real shipping days, late-delivery flag, region, sales and profit",
         "Closest analogue: supports OTD, lead-time prediction and late-risk classification"],
        ["Olist Brazilian E-Commerce", "Olist, Kaggle", "About 100,000 marketplace orders (2016-2018) with purchase, estimated and actual delivery dates, freight value, product weight, review score",
         "Delivery-time regression, freight cost per kg, link between lateness and review score"],
        ["Amazon Last Mile Routing Research Challenge", "Merchan et al. (2024); AWS open data", "Real delivery routes, stops and travel times for last-mile operations in US metros",
         "Route-level optimisation and travel-time modelling (future extension)"],
        ["M5 Forecasting (Walmart)", "Makridakis et al. (2022); Kaggle", "Daily unit sales for thousands of items across stores",
         "Demand forecasting and inventory policy (safety stock, reorder point)"],
        ["World Bank Logistics Performance Index", "World Bank (2023)", "Country scores for customs, infrastructure, timeliness, tracking",
         "Benchmarking context for reliability targets"],
    ], widths_cm=[3.2, 3.8, 5.4, 4.4], caption="Public datasets relevant to the logistics scenario.", font=8.5)
    R.h2("4.3 How the simulated feed mirrors the real datasets")
    R.p("Because the real files are large and carry their own licences, the project generates a seeded simulation whose columns are deliberately aligned with them. "
        "The mapping below shows which real fields each simulated field represents, so every method in this report transfers directly to DataCo or Olist.")
    R.table(["Simulated field", "DataCo analogue", "Olist analogue", "Used for"], [
        ["actual_days", "Days for shipping (real)", "delivered_customer_date minus purchase_timestamp", "Lead-time KPI, regression target"],
        ["planned_days", "Days for shipment (scheduled)", "estimated_delivery_date minus purchase_timestamp", "Promise window; OTD definition"],
        ["delayed (derived)", "Late_delivery_risk", "delivered > estimated date", "OTD KPI, classification target"],
        ["shipping_mode", "Shipping Mode", "-", "Service-level cost and speed comparison"],
        ["delivery_cost_inr", "-", "freight_value", "Cost per km / per kg"],
        ["weight_kg, volume_m3", "-", "product_weight_g, product dimensions", "Cost drivers, load density"],
        ["destination_zone / hub", "Order Region", "customer_state, customer_city", "Zone and lane analysis"],
        ["customer_rating", "-", "review_score", "Satisfaction impact of lateness"],
    ], widths_cm=[3.4, 4.3, 5.0, 4.1], caption="Field mapping between the simulation and public datasets.", font=8.5)
    R.p("The simulation also reproduces typical data-quality problems (missing values, duplicate IDs, inconsistent labels, mixed date formats and extreme values) so that Week 2 has a realistic cleaning task.")

    # ------------------------------------------------------------------ 5
    R.h1("5. How data-science concepts apply to this scenario")
    R.table(["Technique", "Logistics question", "Python tools", "Planned output"], [
        ["Descriptive statistics and correlation", "Which factors move with delay and cost?", "pandas, SciPy", "KPI tables, correlation matrix (Week 3)"],
        ["Linear and multiple regression", "How does cost scale with distance, weight and service level?", "scikit-learn, SciPy", "Cost model and cost benchmarks (this week, Week 4)"],
        ["Clustering (K-Means)", "Which lanes behave alike and need the same policy?", "scikit-learn", "Lane segments with tailored actions (this week)"],
        ["Tree ensembles (regression)", "How many days will this shipment take?", "scikit-learn", "Delivery-time predictor, quantile promise dates (Week 4)"],
        ["Classification", "Which shipments are likely to be late?", "scikit-learn", "Risk ranking for proactive intervention (Week 4)"],
        ["Linear programming", "Which warehouse and which carrier should serve each flow?", "SciPy (HiGHS)", "Cost-minimising allocation under service constraints (Week 4)"],
        ["Inventory models (EOQ, safety stock)", "How much to order and when?", "NumPy", "Reorder policy per warehouse (this week)"],
    ], widths_cm=[3.7, 5.0, 2.8, 5.3], caption="Mapping of data-science methods to logistics decisions.", font=8.5)

    # ------------------------------------------------------------------ 6
    R.h1("6. Pilot exploration of the data")
    R.h2("6.1 KPI baselines by zone, carrier and mode")
    R.fig(FIG / "w1_on_time_by_zone.png", f"On-time delivery rate by destination zone. {worst_zone} is the weakest zone ({zone[worst_zone]['on_time_rate_pct']}%).", 5.6)
    rows = []
    for k, v in sorted(car.items(), key=lambda kv: kv[1]["on_time_rate_pct"]):
        rows.append([k, f"{v['shipments']:,}", f"{v['on_time_rate_pct']}%", f"{v['avg_lead_time_days']}", f"{v['median_cost_per_km_inr']}", f"{v['avg_customer_rating']}"])
    R.table(["Carrier", "Shipments", "On-time", "Lead time (d)", "Median cost/km", "Rating"], rows, widths_cm=[3.6, 2.4, 2.4, 2.8, 3.0, 2.0],
            caption="Carrier scorecard (pilot data), sorted from weakest to strongest on-time rate.", align_right_from=1)
    R.p(f"{worst_car} has the lowest on-time rate ({car[worst_car]['on_time_rate_pct']}%) and {best_car} the highest ({car[best_car]['on_time_rate_pct']}%). "
        f"Mode matters too: Express shipments average {mode['Express']['avg_lead_time_days']} days against {mode['Economy']['avg_lead_time_days']} days for Economy, "
        f"but their median cost per km is roughly twice as high (INR {mode['Express']['median_cost_per_km_inr']} versus INR {mode['Economy']['median_cost_per_km_inr']}).")

    R.h2("6.2 Clustering: segmenting delivery lanes")
    R.p(f"The {r1['n_lanes']} origin-destination lanes were described by five standardised features: average distance, late-delivery rate, median cost per km, average lead time and "
        f"shipment count. K-Means was run for k = 2 to 7; the silhouette score (Rousseeuw, 1987) and the elbow curve guided the choice. Silhouette values for k = 2, 3, 4 were "
        f"{sil['2']:.3f}, {sil['3']:.3f} and {sil['4']:.3f}. k = 2 is too coarse to act on, so the best score among k = 3 to 6 was selected: **k = {bk}**.")
    R.fig(FIG / "w1_cluster_selection.png", "Elbow and silhouette diagnostics for K-Means on lane features.", 5.8)
    R.fig(FIG / "w1_lane_clusters.png", "Lanes projected onto the first two principal components and coloured by cluster.", 5.6)
    rows = [[f"C{c['cluster']}", c["segment"], c["lanes"], f"{c['shipments']:,}", f"{c['avg_distance_km']:.0f}", f"{c['delay_rate_pct']}%", c["cost_per_km_inr"], c["lead_time_days"]] for c in clus]
    R.table(["Cluster", "Segment (auto-labelled)", "Lanes", "Shipments", "Avg km", "Late rate", "Cost/km", "Lead (d)"], rows,
            widths_cm=[1.4, 5.0, 1.3, 2.0, 1.6, 1.7, 1.6, 1.6], caption="Cluster profiles (means of lane features).", font=8.5)
    R.h3("Interpretation of the segments")
    for c in clus:
        seg = c["segment"]
        if "high-delay" in seg:
            act = ("These are the lanes that the network most needs to fix. They run over long distances where transit variability accumulates and the promise window is the same "
                   "kind of rule used elsewhere. Recommended actions: promise longer and lane-specific windows, assign the most reliable carriers, and pre-position stock in the nearest warehouse.")
        elif "short-haul" in seg:
            act = ("Cost per km is highest here because fixed handling and pickup charges are spread over few kilometres. These lanes carry the most shipments, so small savings per parcel "
                   "matter. Recommended actions: consolidate pickups, negotiate flat-rate short-haul tariffs and keep Economy as the default service.")
        else:
            act = ("This is the core of the network. Performance is close to the average, so the opportunity is incremental: carrier mix tuning and better promise dates rather than structural change.")
        R.bullets([f"**C{c['cluster']} - {seg}** ({c['lanes']} lanes, {c['shipments']:,} shipments). Average distance {c['avg_distance_km']:.0f} km, late rate {c['delay_rate_pct']}%, "
                   f"lead time {c['lead_time_days']} days, median cost per km INR {c['cost_per_km_inr']}. Largest lanes: {c['example_lanes']}. {act}"])
    R.p("Two cautions apply. First, K-Means is distance based, so features were standardised; otherwise distance in km would dominate. Second, with only 40 lanes the clusters are "
        "descriptive rather than statistically strong (the silhouette is moderate), so they are used to organise policy discussion, not as a final classification.")

    R.h2("6.3 Regression preview: cost versus distance")
    R.fig(FIG / "w1_cost_regression_preview.png", "Simple linear regression of delivery cost on distance (top 1% of costs trimmed for this preview).", 5.4)
    R.p(f"A first fit gives cost = INR {reg['intercept']:.0f} + INR {reg['slope_per_km']:.3f} per km (R-squared {reg['r2_simple']:.2f}); adding weight and a weight-distance interaction "
        f"lifts R-squared only to {reg['r2_multi']:.2f}, so the remaining variation comes from service level, carrier and peak-season surcharges, which Week 4 includes. "
        f"Importantly, **before trimming the top 1% of costs the same fit has an R-squared of only {reg['r2_untrimmed']:.2f}**: a few extreme billing values destroy the apparent relationship. "
        "This is the strongest argument for the data-cleaning work planned in Week 2.")

    R.h2("6.4 Inventory policy: EOQ and safety stock")
    R.p(f"For each warehouse, daily shipment counts were treated as demand. Using an ordering cost of INR {ep['order_cost_inr']:,.0f} per order, a holding cost of {100*ep['holding_rate']:.0f}% of unit value "
        f"per year, a supplier lead time of {ep['lead_time_days']:.0f} days (standard deviation {ep['lead_time_sd']} days) and a 95% service level (z = {ep['z']}), the formulas are: "
        "EOQ = sqrt(2DS / H); safety stock = z x sqrt(L x sigma_d^2 + d^2 x sigma_L^2); reorder point = d x L + safety stock. These are illustrative parameters to be replaced with real supplier terms.")
    R.table(["Warehouse", "Mean daily units", "Std dev", "Annual units", "EOQ (units)", "Safety stock", "Reorder point"],
            [[i["warehouse"], i["mean_daily_units"], i["sd_daily_units"], f"{i['annual_units']:,}", i["EOQ_units"], i["safety_stock_units"], i["reorder_point_units"]] for i in inv],
            widths_cm=[2.8, 2.5, 1.8, 2.4, 2.4, 2.4, 2.5], caption="Illustrative inventory policy per warehouse.", align_right_from=1)

    # ------------------------------------------------------------------ 7
    R.h1("7. Strategic roadmap")
    R.p("The analysis follows six linked stages, each feeding the next. Weeks 2 to 4 of the internship implement stages 2 to 5; stage 6 describes how the work would be operationalised.")
    R.fig(FIG / "w1_roadmap.png", "End-to-end analytical process flow.", 6.3)
    R.table(["Stage", "Key activities", "Python tools", "Output"], [
        ["1 Collect", "Export shipment, carrier, warehouse and weather data; define one row per shipment; version the raw file", "pandas, requests, SQL connectors", "Raw feed + data dictionary"],
        ["2 Clean", "Remove duplicates, standardise labels, parse dates, treat invalid values and outliers, impute missing values, engineer features, scale", "pandas, NumPy, scikit-learn", "Analysis-ready table + cleaning log (Week 2)"],
        ["3 Explore", "KPIs, distributions, correlations, seasonality, lane and carrier ranking", "pandas, Matplotlib, SciPy", "Insight report and charts (Week 3)"],
        ["4 Model", "Delivery-time regression, late-risk classification, cross-validation, tuning", "scikit-learn", "Validated models (Week 4)"],
        ["5 Optimise", "Warehouse-hub allocation LP, carrier-mix LP, data-driven promise dates", "SciPy (HiGHS), scikit-learn", "Recommended policies and savings (Week 4)"],
        ["6 Decide and monitor", "Dashboard, KPI targets, pilot rollouts, drift monitoring and retraining schedule", "Streamlit or Power BI, cron/Airflow", "Operating cadence"],
    ], widths_cm=[2.4, 7.0, 3.6, 3.8], caption="Roadmap stages, tools and outputs.", font=8.5)
    R.h3("Pseudocode of the end-to-end pipeline")
    R.code('''def run_pipeline():
    raw      = collect(sources=["WMS", "TMS", "carrier_api", "weather_api"])
    clean    = preprocess(raw)                      # dedupe, parse dates, outliers, impute, scale
    insights = explore(clean)                       # KPIs, correlations, seasonality, lane ranking
    model    = train(clean, target="actual_days")   # gradient boosting + 5-fold CV + tuning
    plan     = optimise(model, clean)               # LP: lane allocation, carrier mix, promise dates
    publish(plan, dashboard="ops_control_tower")
    monitor(kpis=["OTD", "cost_per_km"], drift=True, retrain_every="monthly")''')

    # ------------------------------------------------------------------ 8
    R.h1("8. Python code illustrations")
    R.p("The snippets below are taken directly from the project script src/01_strategy_kpis_clustering.py and were executed to produce the numbers in this report.")
    R.code(grab(SCRIPT, "def kpi_block", "overall = kpi_block(df)"), "8.1 KPI computation")
    R.code(grab(SCRIPT, 'feat = ["distance_km"', "prof = lanes.groupby"), "8.2 Lane clustering with model selection by silhouette score")
    R.code(grab(SCRIPT, "daily = df.groupby", "# ------------------------------------------------------------------ 4. Regression preview"), "8.3 EOQ, safety stock and reorder point")

    # ------------------------------------------------------------------ 9
    R.h1("9. Assumptions, risks and limitations")
    R.table(["Item", "Description", "Mitigation"], [
        ["Simulated data", "Patterns reflect the simulator's design, not a real firm", "Pipeline is dataset-agnostic; field mapping to DataCo and Olist given in Table 3"],
        ["Pilot cleaning is light", "Remaining outliers and missing values bias the pilot statistics", "Full treatment in Week 2; Week 1 numbers labelled as pilot"],
        ["Assumed economics", "EOQ ordering/holding costs and the late-delivery penalty are assumptions", "Sensitivity analysis in Week 4; replace with finance data"],
        ["Few lanes (40)", "Clusters are descriptive, silhouette moderate", "Re-cluster at shipment level or with real lane counts"],
        ["Seasonality and drift", "Peak-season behaviour differs from the rest of the year", "Time-aware validation and monthly retraining"],
    ], widths_cm=[3.0, 7.0, 6.8], caption="Main assumptions and mitigations.", font=8.5)

    # ------------------------------------------------------------------ 10
    R.h1("10. Conclusion: expected outcomes and decision impact")
    R.p(f"The pilot confirms that the scenario is analytically rich: service reliability ({ov['on_time_rate_pct']}% on time), lane heterogeneity and cost structure all leave room for improvement, "
        "and each can be tackled with a standard data-science technique implemented in Python. The roadmap sequences the work so that data quality is solved before modelling and "
        "modelling is solved before optimisation.")
    R.p("Expected outcomes, to be tested in Weeks 2 to 4, and the decisions they would inform:")
    R.bullets([
        "**Reliability:** a data-driven promise date and risk flag should lift OTD towards the 90% target by setting realistic windows and routing attention to high-risk shipments.",
        "**Cost:** lane and carrier re-allocation under service constraints should reduce delivery cost per shipment without raising the late rate.",
        "**Inventory:** warehouse-specific reorder points replace one-size-fits-all rules and reduce emergency transfers.",
        "**Governance:** the six KPIs give managers a common scorecard; clusters turn a long list of lanes into a handful of policies."])
    R.references(refs("christopher", "chopra", "dantzig", "toth", "harris", "silver", "breiman", "friedman", "rousseeuw", "pedregosa", "dataco", "olist", "amazon", "m5", "lpi", "nlp"))
    return R
