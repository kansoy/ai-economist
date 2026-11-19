import collections
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output" / "tables"

ANNOTATORS = ["gpt-6.1-sol", "claude-opus-5-5"]
VERSIONS = {
    "A": "Authors' paragraph",
    "B": "Supervised writer",
    "C": "Adversarial, run 1",
    "D": "Adversarial, run 2",
    "E": "Prompted Opus",
}
CATEGORIES = ("dropped", "added", "hedge", "entity", "meaning")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {row["id"]: row for row in read_jsonl(DATA / "test_metadata.jsonl")}
    annotations = {
        (row["annotator"], row["version"], row["id"]): row
        for row in read_jsonl(DATA / "fidelity_annotations.jsonl")
        if row.get("verdict")
    }

    showings = collections.defaultdict(list)
    for row in read_jsonl(DATA / "run1_verdicts.jsonl"):
        if row["pick"] in ("H", "T") and row["step"] in (0, 72):
            showings["B" if row["step"] == 0 else "C"].append((row["id"], row["pick"] == "T"))
    for row in read_jsonl(DATA / "run2_verdicts.jsonl"):
        if row["pick"] in ("H", "T") and row["step"] == 69:
            showings["D"].append((row["id"], row["pick"] == "T"))
    for row in read_jsonl(DATA / "baseline_verdicts.jsonl"):
        if row["pick"] in ("H", "T") and row["cond"] == "R":
            showings["E"].append((row["id"], row["pick"] == "T"))

    papers = sorted({row["paper"] for row in metadata.values()})
    by_paper = collections.defaultdict(list)
    for item_id, row in metadata.items():
        by_paper[row["paper"]].append(item_id)
    all_ids = list(metadata)

    def is_major(annotator, version, item_id):
        return annotations.get((annotator, version, item_id), {}).get("verdict") == "major"

    def major_share(annotator, version, ids):
        return 100.0 * sum(is_major(annotator, version, item_id) for item_id in ids) / len(ids)

    def taken_share(version, ids, annotator=None):
        weights = collections.Counter(ids)
        taken = 0
        total = 0
        for item_id, picked in showings[version]:
            if item_id in weights:
                taken += weights[item_id] * int(picked and not (annotator and is_major(annotator, version, item_id)))
                total += weights[item_id]
        return 100.0 * taken / total

    def bootstrap_ci(statistic, reps=2000, seed=20261117):
        rng = random.Random(seed)
        draws = sorted(
            statistic([item_id for paper in (rng.choice(papers) for _ in papers) for item_id in by_paper[paper]])
            for _ in range(reps)
        )
        return round(draws[int(0.025 * reps)], 1), round(draws[int(0.975 * reps)], 1)

    missing = sum(
        (annotator, version, item_id) not in annotations
        for annotator in ANNOTATORS
        for version in VERSIONS
        for item_id in metadata
    )
    results = {
        "missing_annotations": missing,
        "major": {},
        "categories": {},
        "agreement": {},
        "joint": {},
        "decision_rule": {},
    }

    csv_rows = []
    for version, label in VERSIONS.items():
        row_out = {"version": version, "label": label}
        for annotator in ANNOTATORS:
            share = major_share(annotator, version, all_ids)
            lo, hi = bootstrap_ci(lambda ids: major_share(annotator, version, ids))
            results["major"][f"{version}|{annotator}"] = {"share": round(share, 1), "lo": lo, "hi": hi}
            row_out[f"{annotator}_major"] = round(share, 1)
            row_out[f"{annotator}_lo"] = lo
            row_out[f"{annotator}_hi"] = hi
            results["categories"][f"{version}|{annotator}"] = {
                category: round(
                    100.0
                    * sum(bool(annotations.get((annotator, version, item_id), {}).get(category)) for item_id in all_ids)
                    / len(all_ids),
                    1,
                )
                for category in CATEGORIES
            }
        agree = (
            100.0
            * sum(is_major(ANNOTATORS[0], version, item_id) == is_major(ANNOTATORS[1], version, item_id) for item_id in all_ids)
            / len(all_ids)
        )
        both = (
            100.0
            * sum(is_major(ANNOTATORS[0], version, item_id) and is_major(ANNOTATORS[1], version, item_id) for item_id in all_ids)
            / len(all_ids)
        )
        results["agreement"][version] = {"agree": round(agree, 1), "both_major": round(both, 1)}
        row_out["agree"] = round(agree, 1)
        row_out["both_major"] = round(both, 1)
        csv_rows.append(row_out)

    for version in ("C", "D", "E"):
        for annotator in ANNOTATORS:
            diff = major_share(annotator, version, all_ids) - major_share(annotator, "B", all_ids)
            lo, hi = bootstrap_ci(lambda ids: major_share(annotator, version, ids) - major_share(annotator, "B", ids))
            results["major"][f"{version}-B|{annotator}"] = {"diff": round(diff, 1), "lo": lo, "hi": hi}

    for version in ("B", "C", "D", "E"):
        results["joint"][f"{version}|plain"] = round(taken_share(version, all_ids), 1)
        for annotator in ANNOTATORS:
            results["joint"][f"{version}|{annotator}"] = round(taken_share(version, all_ids, annotator), 1)

    gain_plain = taken_share("C", all_ids) - taken_share("B", all_ids)
    lo_plain, hi_plain = bootstrap_ci(lambda ids: taken_share("C", ids) - taken_share("B", ids))
    results["joint"]["gain_plain"] = {"diff": round(gain_plain, 1), "lo": lo_plain, "hi": hi_plain}
    for annotator in ANNOTATORS:
        gain = taken_share("C", all_ids, annotator) - taken_share("B", all_ids, annotator)
        lo, hi = bootstrap_ci(lambda ids: taken_share("C", ids, annotator) - taken_share("B", ids, annotator))
        results["joint"][f"gain_joint|{annotator}"] = {"diff": round(gain, 1), "lo": lo, "hi": hi}

    no_cost = all(
        results["major"][f"{version}-B|{annotator}"]["diff"] <= 5.0
        and results["major"][f"{version}-B|{annotator}"]["hi"] < 10.0
        for version in ("C", "D")
        for annotator in ANNOTATORS
    )
    joint_positive = all(results["joint"][f"gain_joint|{annotator}"]["lo"] > 0.0 for annotator in ANNOTATORS)
    results["decision_rule"] = {
        "condition_1_no_fidelity_cost": no_cost,
        "condition_2_joint_gain_above_zero": joint_positive,
    }

    (OUT / "content_fidelity.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with open(OUT / "tableA2_fidelity.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    g_gain = results["joint"]["gain_joint|gpt-6.1-sol"]
    o_gain = results["joint"]["gain_joint|claude-opus-5-5"]
    print(
        f"Content fidelity: missing = {missing}; "
        f"run 1 joint gain = {g_gain['diff']:+.1f} ({g_gain['lo']:+.1f}, {g_gain['hi']:+.1f}) GPT, "
        f"{o_gain['diff']:+.1f} ({o_gain['lo']:+.1f}, {o_gain['hi']:+.1f}) Opus; "
        f"condition 1 met = {no_cost}, condition 2 met = {joint_positive}"
    )


if __name__ == "__main__":
    main()
