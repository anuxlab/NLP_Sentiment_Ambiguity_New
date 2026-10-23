# Results and evidence policy

`primary_js_results.csv` merges the canonical seed-level Holm tests and two-level bootstrap summary. `paired_two_level_bootstrap.csv` preserves the original bootstrap output. `holm_corrected_cross_dataset_tests.csv` preserves the corrected seed-level tests. `dataset_condition_summary.csv` contains the three matched conditions; it does not include the separate null-augmented or tie-inclusive composition ablations. `paired_uncertainty_summary.csv` preserves seed-paired secondary metric summaries.

Negative `mean_js_delta_bits` means human-soft-target JS minus majority-CE JS and therefore favors soft targets. Bootstrap intervals are unadjusted. Holm p-values come from separate paired seed-level exact sign-flip tests. Do not conflate them.
