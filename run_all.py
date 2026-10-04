"""Run the whole project end to end:  python run_all.py

Order: generate data -> Week 1 -> Week 2 -> Week 3 -> Week 4 -> build the four Word reports.
Everything is seeded, so results are identical on every run.
"""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = [
    "00_generate_dataset.py",
    "02_cleaning_pipeline.py",       # Week 2 first: it produces the clean file used by Weeks 3-4
    "01_strategy_kpis_clustering.py",
    "03_eda_visualisation.py",
    "04_modeling_optimisation.py",
    "05_build_reports.py",
]

if __name__ == "__main__":
    for step in STEPS:
        t0 = time.time()
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, step], cwd=SRC, check=True)
        print(f"    done in {time.time() - t0:.1f}s", flush=True)
    print("\nAll steps finished. Reports are in ./reports, figures in ./figures, results in ./results.")
