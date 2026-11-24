# Changelog

## `v2026.11.24` — 24 November 2026

Synced the replication package with the revised 20-page working paper (*Can a Language Model Write Like an Economist?*, SHA-256 `971b3d82c7c80ac2e8ff5c88136cb0723e4f2275e5c5a7a4fb81bfb796094907`, live at [fatih.ai/ai_economist.pdf](https://fatih.ai/ai_economist.pdf)). No empirical data, estimates, confidence intervals, or figures changed.

- **Full raw-verdict verification:** `code/01_main_results.py` now also recomputes and exports raw-draft (`D`) and guided-Opus (`R`) baselines directly from `data/baseline_verdicts.jsonl` (`output/tables/baselines.csv` and `main_results.json`), and `run_all.py` verifies every headline estimate across baselines, Run 1, Run 2, equal-length subsets, Table A2 fidelity levels and joint gains, and the human reading test.
- **Table A2 alignment:** Updated version `B` label in `code/02_content_fidelity.py` and `DATA_DICTIONARY.md` to `Supervised writer, run 1` to match Table A2 of the revised paper, and added `output/tables/tableA2_joint.csv` for Panel A differences and Panel B joint outcomes.
- **Documentation & metadata:** Updated `CITATION.cff` (`version: 2026.11.24`, `date-released: 2026-11-24`, keywords), clarified the Step-0 supervised length comparator (`|log(final / step_0)| <= log(1.05)`: 124 and 135 paragraphs) in `DATA_DICTIONARY.md`, and enriched `README.md` with the revised abstract, Figure 2 preview, headline results table, training pipeline guide, and BibTeX citation.
- **Continuous replication CI & Release:** Added `.github/workflows/replicate.yml` to test `python3 run_all.py` automatically on Linux and published GitHub Release `v2026.11.24` with the 20-page working paper PDF attached.

## `v2026.11.19` — 19 November 2026

- Initial public release of replication scripts, training and evaluation code, and anonymised evaluation data for *Can a Language Model Write Like an Economist?* (`cb33642`).
