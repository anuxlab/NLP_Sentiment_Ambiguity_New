# Results package

This directory consolidates the essential reporting artifacts for *Same Words, Different Bets*.

## Start here
- `Two_Page_Research_Report.pdf`: consolidated two-page report draft using canonical controlled TF-IDF results.
- `ANALYSIS.md`: interpretation and caveats.
- `presentation/soft_target_research_presentation_updated.pptx`: original editable deck with three results/interpretation slides appended.
- `figures/`: regenerated JS learning curves, primary paired effects, and macro-F1 curves in PNG and PDF.
- `tables/`: canonical CSVs plus an executive comparison table.
- `source_report/`: supplied Tiny-BERT pilot report source and source figures, preserved for traceability.
- `MANIFEST.json`: summary of provenance and key counts.

## Evidence policy
The canonical primary test table reports 15 comparisons. Human soft targets have a negative mean JS contrast in all 15; only DynaSent Round 2 at budget 300 survives the paired seed-level Holm correction (adjusted p=0.029296875). Bootstrap intervals are unadjusted. Results are exploratory and do not establish universal superiority.

The original `source_report/main.tex` describes a separate Tiny-BERT pilot. Its metrics are not mixed with the controlled TF-IDF results used in the new report.
