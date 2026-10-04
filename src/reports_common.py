"""Shared bits for the four report builders: references, code-snippet extraction, number formatting."""
from __future__ import annotations

import os
from pathlib import Path

from common import CLEAN_FILE, FIG, RAW_FILE, REPORTS, RES, ROOT, load_json  # noqa: F401  (re-exported)

AUTHOR = os.environ.get("REPORT_AUTHOR", "Kaju")
SRC = ROOT / "src"
MONTH_YEAR = "October 2026"


def grab(fname: str, start: str, end: str | None = None, max_lines: int = 60) -> str:
    """Return the exact source lines from the first line containing `start` up to (not including) the line containing `end`.
    The reports embed real code from the repo, so the text in the Word files can never drift from the scripts."""
    lines = (SRC / fname).read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if start in ln)
    j = len(lines) if end is None else next(k for k in range(i + 1, len(lines)) if end in lines[k])
    block = lines[i:j]
    while block and not block[-1].strip():
        block.pop()
    return "\n".join(block[:max_lines])


def f(x, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


REFS = {
    "christopher": "Christopher, M. (2016). Logistics & Supply Chain Management (5th ed.). Pearson.",
    "chopra": "Chopra, S., & Meindl, P. (2016). Supply Chain Management: Strategy, Planning, and Operation (6th ed.). Pearson.",
    "toth": "Toth, P., & Vigo, D. (Eds.) (2014). Vehicle Routing: Problems, Methods, and Applications (2nd ed.). SIAM.",
    "dantzig": "Dantzig, G. B., & Ramser, J. H. (1959). The truck dispatching problem. Management Science, 6(1), 80-91.",
    "harris": "Harris, F. W. (1913). How many parts to make at once. Factory, The Magazine of Management, 10(2), 135-136, 152.",
    "silver": "Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017). Inventory and Production Management in Supply Chains (4th ed.). CRC Press.",
    "breiman": "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
    "friedman": "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232.",
    "rousseeuw": "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53-65.",
    "pedregosa": "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "dataco": "Constante, F., Silva, F., & Pereira, A. (2019). DataCo Smart Supply Chain for Big Data Analysis [Data set]. Mendeley Data (also mirrored on Kaggle).",
    "olist": "Olist (2018). Brazilian E-Commerce Public Dataset by Olist [Data set]. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
    "m5": "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting, 38(4), 1346-1364.",
    "amazon": "Merchan, D., et al. (2024). 2021 Amazon Last Mile Routing Research Challenge: Data set. Transportation Science, 58(1), 8-11.",
    "lpi": "World Bank (2023). Logistics Performance Index 2023: Connecting to Compete. Washington, DC: The World Bank. (India: rank 38 of 139 economies; see PIB press release, Ministry of Commerce and Industry, https://www.pib.gov.in/PressReleasePage.aspx?PRID=2003541)",
    "nlp": "Government of India, Ministry of Commerce and Industry (2022). National Logistics Policy.",
    "tukey": "Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.",
    "little": "Little, R. J. A., & Rubin, D. B. (2019). Statistical Analysis with Missing Data (3rd ed.). Wiley.",
    "leys": "Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. Journal of Experimental Social Psychology, 49(4), 764-766.",
    "mckinney": "McKinney, W. (2010). Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference, 56-61.",
    "hunter": "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95.",
    "virtanen": "Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261-272.",
    "koenker": "Koenker, R., & Bassett, G. (1978). Regression quantiles. Econometrica, 46(1), 33-50.",
    "huangfu": "Huangfu, Q., & Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. Mathematical Programming Computation, 10, 119-142.",
}


def refs(*keys: str) -> list[str]:
    return [REFS[k] for k in keys]
