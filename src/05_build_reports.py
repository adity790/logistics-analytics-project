"""Build the four Word reports into ./reports (run after scripts 00-04)."""
from common import REPORTS, ensure_dirs
import report_w1, report_w2, report_w3, report_w4

if __name__ == "__main__":
    ensure_dirs()
    for n, mod in enumerate((report_w1, report_w2, report_w3, report_w4), start=1):
        out = REPORTS / f"Week{n}_{['Strategic_Planning','Data_Cleaning_Preprocessing','Analysis_and_Visualization','Predictive_Modeling_Optimization'][n-1]}_Report.docx"
        mod.build().save(out)
        print("saved", out.name)
