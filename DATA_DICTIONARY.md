# Data Dictionary

All files in `data/` are UTF-8 JSON Lines (`.jsonl`) records containing no third-party manuscript text, no personal identifiers, and no IP addresses.

## `test_metadata.jsonl` (191 rows)
Structural metadata for the 191 test records (137 paragraphs from 59 working papers dated 2026; 54 records representing 53 distinct paragraphs from 53 papers dated 1999–2022).
- `id` (string): Paragraph record identifier.
- `paper` (string): Paper cluster identifier used for paper-clustered bootstrap resampling.
- `set` (string): Benchmark cohort (`clean2026` or `pre2023`).
- `draft_words` (int): Word count of the GPT-6.1 Sol input draft.
- `run1_words`, `run2_words` (object): Word counts of revised paragraphs keyed by training step (`0`, `50`, `72` for Run 1; `0`, `50`, `69` for Run 2).
- `run1_numbers_kept`, `run2_numbers_kept` (object): Boolean indicator of whether every number in the draft is preserved in the revision, keyed by training step.

## `run1_verdicts.jsonl`, `run2_verdicts.jsonl` (2,292 rows each) & `baseline_verdicts.jsonl` (1,528 rows)
Pairwise outside-judge verdicts across both display orders (191 paragraphs × 2 orders × 2 judges per condition/step).
- `step` (int, runs only): Training step (`0`, `50`, `72` in Run 1; `0`, `50`, `69` in Run 2).
- `cond` (string): Condition code (`T` trained revision; `D` raw GPT draft; `R` prompted Claude Opus 5.5 with writing guide).
- `judge` (string): Outside judge model (`gpt-6.1-sol` or `claude-opus-5-5`).
- `id` (string): Paragraph record identifier matching `test_metadata.jsonl`.
- `shown` (string): Presentation order (`HT` = authors first, model second; `TH` = model first, authors second).
- `pick` (string): Judge selection mapped back to source (`H` = authors, `T` = machine version).
- `raw` (string): Single-letter model response (`A` or `B`).

## `run1_summary.jsonl`, `run2_summary.jsonl` (12 rows each) & `baseline_summary.jsonl` (8 rows)
Cohort-level summary statistics per judge and step/condition used directly by Figure 2.
- `step` / `cond`, `judge`, `set`: Group identifiers.
- `fooled`, `n`, `rate`, `ci95`: Count taken for authors, total showings, percentage rate, and 95% paper-clustered bootstrap interval.
- `numbers_kept`, `paragraphs`, `papers`, `len_ratio`, `unanswered`: Number retention count, paragraph count, unique paper count, mean revision-to-draft word ratio, and unanswered count.

## `fidelity_annotations.jsonl` (1,910 rows)
Blind content-fidelity ratings across 191 paragraphs × 5 versions × 2 annotators (`gpt-6.1-sol`, `claude-opus-5-5`).
- `id` (string): Paragraph record identifier.
- `version` (string): Version code (`A` authors' paragraph, `B` supervised writer, `C` adversarial Run 1, `D` adversarial Run 2, `E` prompted Opus).
- `annotator` (string): Blind annotator model.
- `verdict` (string): Overall fidelity call (`same`, `minor`, or `major`).
- `dropped`, `added`, `hedge`, `entity`, `meaning` (int): Count of flagged differences in each category.

## `reading_test.jsonl` (95 rows)
Anonymised answers from 23 human readers evaluating pairs drawn from 24 of the 2026 paragraphs.
- `reader_id` (string): Anonymised reader identifier (`r01` through `r23`) used for reader-clustered bootstrap intervals.
- `degree` (string): Self-reported master's or doctoral degree status (`yes` or `no`).
- `paragraph_id` (string): Anonymised test paragraph identifier.
- `field_group` (string): Journal group of the paragraph.
- `condition` (string): Machine version paired with the authors' paragraph (`draft`, `v1` supervised writer, `ral` Run 1 adversarial writer).
- `correct` (int): `1` if the reader identified the authors' paragraph, `0` otherwise.
- `position` (int): Pair display position (`1` through `5`).
- `completed` (int): `1` if the reader completed all five pairs, `0` otherwise.

## `training_log_run1.jsonl` (72 rows) & `training_log_run2.jsonl` (69 rows)
Per-step training diagnostics from both adversarial runs used in Figure A1.
- `step` (int): Adversarial training step.
- `warmup`, `paused`, `judge_trained`, `writer_updated` (bool): Step control flags.
- `judge_acc`, `judge_ema`, `judge_loss` (float): Step accuracy, exponential moving mean accuracy (weight 0.9), and judge training loss.
- `reward_p`, `reward_logodds`, `gate_pass`, `kl`, `words`, `len_ratio`, `len_slope`, `len_slope_step`, `groups_used`, `skip`, `sec`: Step reward, length adjustment, KL divergence, and runtime diagnostics.
