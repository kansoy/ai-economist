# Can a Language Model Write Like an Economist? — Replication Package

This repository contains the replication code, training and evaluation scripts, and anonymised evaluation data for *Can a Language Model Write Like an Economist?* (Fatih Kansoy, University of Oxford, November 2026; revised 20-page working paper, 24 November 2026).

- Paper PDF (20 pages): https://fatih.ai/ai_economist.pdf
- Research summary: https://fatih.ai/research/ai_economist/
- Project page: https://fatih.ai/projects/ai-economist/
- Release note: https://fatih.ai/news/2026-11-19-ai-economist/
- Repository: https://github.com/kansoy/ai-economist

## Abstract

In most comparisons against economists' prose, outside judges told them apart even with detailed rules and examples. On one rented GPU we fine-tuned an open model without explicit rules on 5,267 pairs of drafts and pre-2023 authors' paragraphs, then trained it against a co-trained judge providing a length-adjusted authorship reward. This adversarial training improved performance by 21.1 and 20.8 percentage points on 191 records in two runs, with similar gains among paragraphs within 5 per cent of supervised revision length. For 2026 working papers, GPT-6.1 Sol chose the final revisions 70.8 and 68.2 per cent of the time (both higher than chance) while Claude Opus 5.5 chose them 55.8 and 54.4 per cent of the time (well ahead of raw drafts and the guided model). Humans facing trained writers picked out authors' prose 41.0 per cent of the time (pooled) versus 61.8 per cent for raw drafts. Blind annotators judged the trained writers to alter claims more often than the guided model, so authors have to check the meaning.

## Requirements

- Python 3.10 or newer
- For reproducing all paper tables and figures from the provided evaluation data:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Optional: LuaLaTeX with Source Serif Pro embeds exact manuscript typography in vector PDF figures; otherwise Matplotlib falls back automatically to embedded TrueType PDF and 300-DPI PNG output.
- For re-running GPU training (`training/`): PyTorch with CUDA, `transformers`, `peft`, `bitsandbytes`, and `safetensors` on a single 80 GB GPU (`google/gemma-4-31B-it`). API evaluation scripts read `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` from standard environment variables.

## One-Command Reproduction

From the repository root, run:

```bash
python3 run_all.py
```

`run_all.py` executes `code/01_main_results.py` through `code/04_figures.py` sequentially, writes all tables to `output/tables/` and figures to `output/figures/`, and verifies every headline estimate against the paper.

## Repository Structure

```text
.
├── run_all.py
├── requirements.txt
├── DATA_DICTIONARY.md
├── CITATION.cff
├── LICENSE
├── code/
│   ├── 01_main_results.py
│   ├── 02_content_fidelity.py
│   ├── 03_reading_test.py
│   └── 04_figures.py
├── data/
│   ├── test_metadata.jsonl
│   ├── run1_verdicts.jsonl
│   ├── run2_verdicts.jsonl
│   ├── baseline_verdicts.jsonl
│   ├── run1_summary.jsonl
│   ├── run2_summary.jsonl
│   ├── baseline_summary.jsonl
│   ├── fidelity_annotations.jsonl
│   ├── reading_test.jsonl
│   ├── training_log_run1.jsonl
│   └── training_log_run2.jsonl
├── training/
│   ├── prompts.py
│   ├── train_supervised.py
│   ├── adversarial_trainer.py
│   ├── evaluate_judges.py
│   └── evaluate_fidelity.py
└── output/
    ├── tables/
    └── figures/
```

## Exhibit Map

| Paper Exhibit | Script | Output File |
|---|---|---|
| Table 2 (Outside judges before and after adversarial training, including Panel C equal-length subset within 5% of Step 0 supervised revision length) | `code/01_main_results.py` | `output/tables/table2_main.csv`, `output/tables/main_results.json` |
| Table A2 (Content fidelity of each version against its draft, using Run 1 supervised writer as baseline) | `code/02_content_fidelity.py` | `output/tables/tableA2_fidelity.csv`, `output/tables/content_fidelity.json` |
| Reading test statistics (Section 4 & Figure A2; 23 readers, 95 answers as of 17 November 2026) | `code/03_reading_test.py` | `output/tables/reading_test.json` |
| Figure 2 (How often each version is taken for the authors' paragraph) | `code/04_figures.py` | `output/figures/fig2_attribution.pdf`, `.png` |
| Figure A1 (Training dynamics in the two runs) | `code/04_figures.py` | `output/figures/figA1_training.pdf`, `.png` |
| Figure A2 (Readers choosing the human paragraph) | `code/04_figures.py` | `output/figures/figA2_readers.pdf`, `.png` |

## Data Availability Note

The original corpus paragraphs come from other researchers' papers and cannot be redistributed in full. The main drafting, training, and judging prompts appear in the Online Appendix (`training/prompts.py`). Every anonymised verdict, content-fidelity rating, training log, word count, number-retention indicator, and reading-test answer required to reproduce every table, figure, confidence interval, and decision rule in the paper is provided directly in `data/`. See `DATA_DICTIONARY.md` for schema details.

## Citation

```bibtex
@unpublished{kansoy2026aieconomist,
  title  = {Can a Language Model Write Like an Economist?},
  author = {Fatih Kansoy},
  note   = {University of Oxford Working Paper},
  month  = nov,
  year   = {2026},
  url    = {https://fatih.ai/ai_economist.pdf}
}
```

## Licence

Code is released under the MIT Licence (`LICENSE`).
