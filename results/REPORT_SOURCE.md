# Two-page report source and provenance

The generated `Two_Page_Research_Report.pdf` is based on `tables/dataset_condition_summary.csv` and `tables/primary_js_results.csv` from the uploaded `04_Results` package. It focuses on the controlled TF-IDF study across DynaSent Round 1, DynaSent Round 2, and SentiMP-En.

The supplied `source_report/main.tex` describes a separate Tiny-BERT pilot (100 labels, seeds 0–4). It is preserved in `source_report/` as source evidence but is not used to populate the TF-IDF result tables. Do not mix the two experiments' metrics.

Regenerate the plots with Python/matplotlib from the canonical CSVs. This PDF is a consolidated reporting draft; review it with the supervisor and confirm final model/configuration details before submission.
