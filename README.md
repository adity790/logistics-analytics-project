# Logistics Data Science Project: Weeks 1 to 4

An end-to-end, reproducible logistics analytics project in Python: strategic planning and clustering (Week 1), data cleaning and preprocessing (Week 2), exploratory analysis and visualisation (Week 3), and predictive modelling with optimisation (Week 4). Each week has a runnable script and a Word report in `reports/`.

**Scenario.** A pan-India e-commerce fulfilment network: 5 warehouses (Pune, Mumbai, Delhi, Bengaluru, Kolkata), 8 destination hubs, 6 zones, 4 carriers and 3 service levels, about 12,000 shipments in 2025.

> **About the data.** The dataset is *simulated* (seeded, `src/00_generate_dataset.py`). Its fields are aligned with public datasets (DataCo Smart Supply Chain, Olist Brazilian E-Commerce) and it contains realistic defects (duplicates, missing values, mixed date formats, outliers). Numbers describe the simulator, not a real company. The same code runs on real files after mapping column names (see the field-mapping table in the Week 1 report).

## Results at a glance

| Week | Topic | Headline result |
|---|---|---|
| 1 | Strategy, KPIs, clustering | 6 KPIs; 40 lanes clustered into 3 segments (silhouette-selected); pilot on-time rate 70.4% |
| 2 | Cleaning and preprocessing | 12,160 raw rows to 11,880 clean; completeness 98.8% to 99.8%; robust outlier rule vs IQR and z-score compared; imputation methods benchmarked |
| 3 | EDA and visualisation | North-East = 12% of shipments but 19% of late deliveries; peak season lifts late rate from 25.2% to 37.6%; Express about 2x cost per km |
| 4 | Modelling and optimisation | Delivery-time RMSE 0.55 days (R-squared 0.90) vs 0.67 for the current promise; late-risk AUC 0.76; warehouse LP saves up to 19% cost at constant late rate (planning estimate) |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── run_all.py                  # runs everything end to end
├── data/
│   ├── raw/                    # simulated raw feed (with defects)
│   └── processed/              # cleaned and scaled files
├── src/
│   ├── common.py               # paths, plot style, helpers
│   ├── 00_generate_dataset.py  # seeded simulator
│   ├── 01_strategy_kpis_clustering.py   # Week 1
│   ├── 02_cleaning_pipeline.py          # Week 2
│   ├── 03_eda_visualisation.py          # Week 3
│   ├── 04_modeling_optimisation.py      # Week 4
│   ├── 05_build_reports.py     # builds the Word reports
│   └── report_w1..w4.py, docx_helpers.py, reports_common.py
├── figures/                    # all charts (PNG)
├── results/                    # week1..4_results.json (every number quoted in the reports)
├── reports/                    # Week 1-4 Word reports (.docx)
└── docs/submission_descriptions.md     # 200+ word descriptions for the portal
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                                       # about 4 minutes
```

Everything is seeded (`SEED = 42`), so results are identical on every run. To run a single week: `cd src && python 02_cleaning_pipeline.py` (run `00_generate_dataset.py` and Week 2 first, because Weeks 3 and 4 read the cleaned file). The optional optimisation uses SciPy's HiGHS solver, so no extra solver install is needed.

## What each week contains

- **Week 1:** KPI framework with baselines, literature and public-dataset review, K-Means lane clustering with silhouette selection and interpreted segments, EOQ and safety-stock policy, cost-distance regression preview, roadmap diagram, pseudocode.
- **Week 2:** profiling, duplicates, label standardisation, explicit date parsing, impossible-value rules, three outlier methods compared, missingness test, five imputers benchmarked on masked data, skew and six normalisations compared, full cleaning log and before/after scorecard.
- **Week 3:** descriptive statistics, distributions, Spearman correlations, seasonality, carrier x zone heatmap, cost drivers, bottleneck lanes, weather and traffic effects, rating vs lateness, significance tests, recommendations.
- **Week 4:** six regression models with 5-fold CV, randomised hyper-parameter search, time-series CV, permutation importance, late-risk classifier and gain curve, warehouse-to-hub LP (four scenarios), carrier-mix LP, quantile-based delivery promises.

## Limitations

Simulated data; LP savings assume stable lane costs and an assumed late penalty (a planning estimate to be validated in a pilot); delivery time is dominated by distance and the promised days, so the model's improvement over simple baselines is modest; error rises in peak-season folds, so regular retraining is advised.

## Author

Kaju, Logistics Data Science internship. Libraries: pandas, NumPy, SciPy, scikit-learn, Matplotlib, python-docx.
