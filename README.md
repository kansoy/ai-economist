# Can a Language Model Write Like an Economist? — Replication Package

This repository contains the replication code, training and evaluation scripts, and anonymised evaluation data for *Can a Language Model Write Like an Economist?* (Fatih Kansoy, University of Oxford, November 2026).

Paper PDF: https://fatih.ai/ai_economist.pdf  
Paper page: https://fatih.ai/research/ai_economist/  
Repository: https://github.com/kansoy/ai-economist

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
| Table 2 (Outside judges before and after adversarial training) | `code/01_main_results.py` | `output/tables/table2_main.csv`, `output/tables/main_results.json` |
| Table A2 (Content fidelity of each version against its draft) | `code/02_content_fidelity.py` | `output/tables/tableA2_fidelity.csv`, `output/tables/content_fidelity.json` |
| Reading test statistics (Section 4 & Figure A2) | `code/03_reading_test.py` | `output/tables/reading_test.json` |
| Figure 2 (Attribution across four versions) | `code/04_figures.py` | `output/figures/fig2_attribution.pdf`, `.png` |
| Figure A1 (Training dynamics in both runs) | `code/04_figures.py` | `output/figures/figA1_training.pdf`, `.png` |
| Figure A2 (Human readers choosing the human paragraph) | `code/04_figures.py` | `output/figures/figA2_readers.pdf`, `.png` |

## Data Availability Note

Third-party authors' paragraphs come from published articles and working papers by other researchers and cannot be redistributed in full. Every anonymised verdict, content-fidelity rating, training log, word count, number-retention indicator, and reading-test answer required to reproduce every table, figure, confidence interval, and decision rule in the paper is provided directly in `data/`. See `DATA_DICTIONARY.md` for schema details.

## Licence

Code is released under the MIT Licence (`LICENSE`).
