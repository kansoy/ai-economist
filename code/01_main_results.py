import collections
import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output" / "tables"


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def evaluate_run(metadata, verdicts, word_key, kept_key, final_step):
    judges = sorted({row["judge"] for row in verdicts if row["pick"] != "?"})
    votes = collections.defaultdict(int)
    totals = collections.defaultdict(int)
    for row in verdicts:
        if row["pick"] == "?":
            continue
        key = (row["step"], row["judge"], row["id"])
        votes[key] += int(row["pick"] == "T")
        totals[key] += 1

    def paper_of(item_id):
        return metadata[item_id]["paper"]

    def rate(ids, selected_judges, step):
        num = sum(votes[(step, judge, item_id)] for judge in selected_judges for item_id in ids)
        den = sum(totals[(step, judge, item_id)] for judge in selected_judges for item_id in ids)
        return 100.0 * num / den if den else float("nan")

    def change(ids, selected_judges, step=final_step):
        num = sum(
            votes[(step, judge, item_id)] - votes[(0, judge, item_id)]
            for judge in selected_judges
            for item_id in ids
        )
        den = sum(totals[(0, judge, item_id)] for judge in selected_judges for item_id in ids)
        return 100.0 * num / den if den else float("nan")

    def bootstrap_ci(ids, fn, reps=4000, seed=1):
        by_paper = collections.defaultdict(list)
        for item_id in ids:
            by_paper[paper_of(item_id)].append(item_id)
        papers = sorted(by_paper)
        rng = random.Random(seed)
        draws = []
        for _ in range(reps):
            sample = [item_id for paper in (rng.choice(papers) for _ in papers) for item_id in by_paper[paper]]
            value = fn(sample)
            if not math.isnan(value):
                draws.append(value)
        draws.sort()
        return round(draws[int(0.025 * len(draws))], 1), round(draws[int(0.975 * len(draws)) - 1], 1)

    results = {"final_step": final_step, "cohorts": {}, "rules": {}}
    for cohort in ("all", "clean2026", "pre2023"):
        ids = [item_id for item_id, row in metadata.items() if cohort in ("all", row["set"])]
        cohort_res = {
            "paragraphs": len(ids),
            "papers": len({paper_of(item_id) for item_id in ids}),
            "judges": {},
        }
        for selected in [[judge] for judge in judges] + [judges]:
            label = selected[0] if len(selected) == 1 else "both"
            diff = change(ids, selected)
            lo, hi = bootstrap_ci(ids, lambda sample: change(sample, selected))
            level_lo, level_hi = bootstrap_ci(ids, lambda sample: rate(sample, selected, final_step))
            cohort_res["judges"][label] = {
                "step_0": round(rate(ids, selected, 0), 1),
                "step_50": round(rate(ids, selected, 50), 1),
                "final": round(rate(ids, selected, final_step), 1),
                "final_ci95": [level_lo, level_hi],
                "change": round(diff, 1),
                "change_ci95": [lo, hi],
            }

        kept_0 = 100.0 * sum(metadata[item_id][kept_key]["0"] for item_id in ids) / len(ids)
        kept_f = 100.0 * sum(metadata[item_id][kept_key][str(final_step)] for item_id in ids) / len(ids)
        lr_0 = statistics.mean(metadata[item_id][word_key]["0"] / metadata[item_id]["draft_words"] for item_id in ids)
        lr_f = statistics.mean(
            metadata[item_id][word_key][str(final_step)] / metadata[item_id]["draft_words"] for item_id in ids
        )
        near = [
            item_id
            for item_id in ids
            if abs(math.log(metadata[item_id][word_key][str(final_step)] / metadata[item_id][word_key]["0"]))
            <= math.log(1.05)
        ]
        near_diff = change(near, judges)
        near_lo, near_hi = bootstrap_ci(near, lambda sample: change(sample, judges))

        cohort_res["numbers_kept_pct"] = {"step_0": round(kept_0, 1), "final": round(kept_f, 1)}
        cohort_res["length_ratio"] = {
            "step_0": round(lr_0, 3),
            "final": round(lr_f, 3),
            "pct_change": round(100.0 * (lr_f / lr_0 - 1.0), 1),
        }
        cohort_res["equal_length"] = {
            "paragraphs": len(near),
            "change": round(near_diff, 1),
            "change_ci95": [near_lo, near_hi],
        }
        results["cohorts"][cohort] = cohort_res

    all_both = results["cohorts"]["all"]["judges"]["both"]
    c26 = results["cohorts"]["clean2026"]
    pre = results["cohorts"]["pre2023"]
    grew = (
        results["cohorts"]["all"]["length_ratio"]["final"] / results["cohorts"]["all"]["length_ratio"]["step_0"] - 1.0
    )
    results["rules"] = {
        "rule_1_overall_positive": all_both["change_ci95"][0] > 0.0,
        "rule_2_2026_each_judge_ge_5": all(c26["judges"][judge]["change"] >= 5.0 for judge in judges),
        "rule_3_2026_numbers_kept": (
            c26["numbers_kept_pct"]["final"] >= 90.0
            and c26["numbers_kept_pct"]["final"] >= c26["numbers_kept_pct"]["step_0"] - 3.0
        ),
        "rule_4_pre2023_no_fall": pre["judges"]["both"]["change_ci95"][1] >= 0.0,
        "rule_5_not_length_driven": grew <= 0.05 or results["cohorts"]["all"]["equal_length"]["change_ci95"][0] > 0.0,
    }
    return results


def evaluate_baselines(metadata, verdicts):
    judges = sorted({row["judge"] for row in verdicts if row["pick"] != "?"})
    conds = {"D": "GPT draft", "R": "Guided Opus"}
    votes = collections.defaultdict(int)
    totals = collections.defaultdict(int)
    for row in verdicts:
        if row["pick"] == "?":
            continue
        key = (row["cond"], row["judge"], row["id"])
        votes[key] += int(row["pick"] == "T")
        totals[key] += 1

    def paper_of(item_id):
        return metadata[item_id]["paper"]

    def rate(ids, cond, selected_judges):
        num = sum(votes[(cond, judge, item_id)] for judge in selected_judges for item_id in ids)
        den = sum(totals[(cond, judge, item_id)] for judge in selected_judges for item_id in ids)
        return 100.0 * num / den if den else float("nan")

    def bootstrap_ci(ids, cond, selected_judges, reps=4000, seed=1):
        by_paper = collections.defaultdict(list)
        for item_id in ids:
            by_paper[paper_of(item_id)].append(item_id)
        papers = sorted(by_paper)
        rng = random.Random(seed)
        draws = []
        for _ in range(reps):
            sample = [item_id for paper in (rng.choice(papers) for _ in papers) for item_id in by_paper[paper]]
            value = rate(sample, cond, selected_judges)
            if not math.isnan(value):
                draws.append(value)
        draws.sort()
        return round(draws[int(0.025 * len(draws))], 1), round(draws[int(0.975 * len(draws)) - 1], 1)

    out = {}
    for cond, label in conds.items():
        out[cond] = {"label": label, "cohorts": {}}
        for cohort in ("clean2026", "pre2023", "all"):
            ids = [item_id for item_id, row in metadata.items() if cohort in ("all", row["set"])]
            cohort_res = {"paragraphs": len(ids), "judges": {}}
            for selected in [[judge] for judge in judges] + [judges]:
                key = selected[0] if len(selected) == 1 else "both"
                lo, hi = bootstrap_ci(ids, cond, selected)
                cohort_res["judges"][key] = {
                    "rate": round(rate(ids, cond, selected), 1),
                    "ci95": [lo, hi],
                }
            out[cond]["cohorts"][cohort] = cohort_res
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {row["id"]: row for row in read_jsonl(DATA / "test_metadata.jsonl")}
    baselines = evaluate_baselines(metadata, read_jsonl(DATA / "baseline_verdicts.jsonl"))
    run1 = evaluate_run(metadata, read_jsonl(DATA / "run1_verdicts.jsonl"), "run1_words", "run1_numbers_kept", 72)
    run2 = evaluate_run(metadata, read_jsonl(DATA / "run2_verdicts.jsonl"), "run2_words", "run2_numbers_kept", 69)
    summary = {"baselines": baselines, "run_1": run1, "run_2": run2}
    (OUT / "main_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    baseline_csv = []
    for cond in ("D", "R"):
        for cohort in ("clean2026", "pre2023", "all"):
            cohort_data = baselines[cond]["cohorts"][cohort]
            for judge in ("gpt-6.1-sol", "claude-opus-5-5", "both"):
                cell = cohort_data["judges"][judge]
                baseline_csv.append(
                    {
                        "cond": cond,
                        "label": baselines[cond]["label"],
                        "cohort": cohort,
                        "paragraphs": cohort_data["paragraphs"],
                        "judge": judge,
                        "rate": cell["rate"],
                        "ci_lo": cell["ci95"][0],
                        "ci_hi": cell["ci95"][1],
                    }
                )
    with open(OUT / "baselines.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_csv[0].keys()))
        writer.writeheader()
        writer.writerows(baseline_csv)

    csv_rows = []
    for run_name, run_data in (("run_1", run1), ("run_2", run2)):
        for cohort in ("clean2026", "pre2023", "all"):
            cohort_data = run_data["cohorts"][cohort]
            judge_keys = ("gpt-6.1-sol", "claude-opus-5-5") if cohort != "all" else ("both",)
            for judge in judge_keys:
                cell = cohort_data["judges"][judge]
                csv_rows.append(
                    {
                        "run": run_name,
                        "cohort": cohort,
                        "paragraphs": cohort_data["paragraphs"],
                        "judge": judge,
                        "step_0": cell["step_0"],
                        "step_50": cell["step_50"] if cohort != "all" else "",
                        "final": cell["final"],
                        "change": cell["change"],
                        "ci_lo": cell["change_ci95"][0],
                        "ci_hi": cell["change_ci95"][1],
                    }
                )
        eq = run_data["cohorts"]["all"]["equal_length"]
        csv_rows.append(
            {
                "run": run_name,
                "cohort": "equal_length_5pct",
                "paragraphs": eq["paragraphs"],
                "judge": "both",
                "step_0": "",
                "step_50": "",
                "final": "",
                "change": eq["change"],
                "ci_lo": eq["change_ci95"][0],
                "ci_hi": eq["change_ci95"][1],
            }
        )

    with open(OUT / "table2_main.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    d26 = baselines["D"]["cohorts"]["clean2026"]["judges"]
    r26 = baselines["R"]["cohorts"]["clean2026"]["judges"]
    print(
        f"Baselines (2026): GPT draft = {d26['gpt-6.1-sol']['rate']}% GPT / {d26['claude-opus-5-5']['rate']}% Opus; "
        f"Guided Opus = {r26['gpt-6.1-sol']['rate']}% GPT / {r26['claude-opus-5-5']['rate']}% Opus"
    )
    for run_name, run_data in (("Run 1", run1), ("Run 2", run2)):
        overall = run_data["cohorts"]["all"]["judges"]["both"]
        eq = run_data["cohorts"]["all"]["equal_length"]
        print(
            f"{run_name}: step 0 = {overall['step_0']}%, final = {overall['final']}%, "
            f"change = {overall['change']:+.1f} ({overall['change_ci95'][0]:+.1f}, {overall['change_ci95'][1]:+.1f}); "
            f"equal-length ({eq['paragraphs']}) = {eq['change']:+.1f} ({eq['change_ci95'][0]:+.1f}, {eq['change_ci95'][1]:+.1f}); "
            f"all rules met = {all(run_data['rules'].values())}"
        )


if __name__ == "__main__":
    main()
