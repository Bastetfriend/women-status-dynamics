# Analysis code — *A Dynamical Regularity for the Evolution of Women's Status*

Preprint: https://doi.org/10.5281/zenodo.22787591

Python 3.11. Dependencies: numpy, scipy, pandas, scikit-learn, matplotlib, python-docx.

## Layout

```
code/
  01_fit/              Richards fitting pipeline (191 countries + three V-Dem sub-dimensions)
  02_model_selection/  Five-model AIC comparison; non-sigmoid five-parameter controls
  03_events/           Suppression–rebound detection (328 events); detection-bias supplement; Chile case
  04_piecewise/        Two-segment piecewise-linear winners; breakpoint event classification; k_A/k_B
  05_external/         M(E(t)) tests: GDP modulation, EDI, channel decomposition, class proxies, IDV×EDI
  06_validation/       Scaling collapse; isotonic sham collapse; pre-1900 sovereign subset; IPU validation; curve clustering
  07_conditions/       Reduction conditions: marriage-age (SMAM) tests
  08_second_stage/     Symmetry extrapolation and acceleration model (Supplemental Material)
  09_figures/          Figure generation
data/                  Intermediate outputs referenced by the manuscript
```

## Manuscript → script

| Manuscript | Script |
|---|---|
| §1, Table I, Figs. 1–2 (fitting) | `01_fit/_vdem_fit_all.py`, `_richards_diagnostics.py` |
| §1 (sub-dimensions) | `01_fit/_r3_item7_subdimensions.py` |
| §1 (model comparison) | `02_model_selection/_vdem_model_comparison.py`, `_vdem_daic_analysis.py`, `_sigmoid_vs_unstructured.py` |
| §2 (reduction conditions i–iii) | `07_conditions/*`, `05_external/*` |
| §3.1 (328 events, irreversibility) | `03_events/_vdem_ratchet.py`, `_p0-4_detection_bias.py`, `_chile_case.py` |
| §3.2 (k_A ≫ k_B) | `04_piecewise/_pw_winners.py`, `_compute_ka_kb_ratio.py` |
| §3.2 (breakpoint event classification) | `04_piecewise/_pw_historical_events.py` |
| §4 (parametric autonomy, channel decomposition) | `05_external/_mediation_panel.py`, `_mediation_bootstrap.py`, `_edi_direct.py`, `_edi_transmission.py` |
| §4 (class proxies) | `05_external/_08_class_laborshare_rank.py`, `_07_class_gini_rank.py`, `_10_class_mobility_rank.py` |
| §4 (IDV × EDI) | `05_external/_2x2_idv_edi.py`, `_idv_vs_edi.py`, `_full_recompute_idv_edi.py` |
| §4 (reverse prediction) | `05_external/_test_S_predicts_GDP.py` |
| §5(6) (scaling collapse) | `06_validation/_p1-3_scaling_collapse.py` |
| §5(6) (isotonic sham collapse) | `06_validation/_r3_sham_collapse_v2.py`, `_r3_items_6_8_analysis.py` |
| §5(7) (IPU validation) | `06_validation/_validate_ipu_wb.py`, `_validate_ipu_v2.py`, `_validate_ipu_compare.py` |
| SM (sovereign-state antiquity) | `06_validation/_r3_items_6_8_analysis.py` |
| SM (second-stage acceleration) | `08_second_stage/*` |
| Fig. S1 (parameter diagnostics) | `01_fit/_richards_diagnostics.py` |

## Data

Raw inputs are public and are not bundled here:

- V-Dem v16 — WPEI and its sub-indices (`v4x_gencl`, `v4x_gencs`, `v4x_genpp`), v-dem.net
- Our World in Data — processed WPEI series, ourworldindata.org
- World Bank WDI — parliamentary seats (`SG.GEN.PARL.ZS`), female age at first marriage (`SP.DYN.SMAM.FE`), GDP, education
- Penn World Table 11.0 — labour share (`labsh`)
- SWIID 9.9.2 — income Gini
- GDIM 2023 v3 — intergenerational educational persistence (BETA)
- Hofstede — IDV; IPU — seat shares

`data/` holds the intermediate outputs produced by these scripts and cited in the manuscript.

## Known issues

1. **Run from the repository root.** Scripts read and write data via the relative path `data/`, so they must be run from the repository root. The intermediate outputs cited by the manuscript are bundled in `data/`; the raw public inputs (V-Dem, World Bank, PWT, SWIID, GDIM, Hofstede, IPU — listed above) are *not* bundled, so download them and place them in `data/` before running the scripts that read them.
2. **One analysis has no script.** The Supplemental Material section "GDP Threshold Robustness" (171 countries split into GDP-per-capita quartiles) is not reproducible from this repository — its analysis code was not retained.
3. Scripts that perform one-off text edits on the manuscript `.docx` are not analysis code and are excluded here.
