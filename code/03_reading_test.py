import collections
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output" / "tables"
VERSIONS = ["draft", "v1", "ral"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def share(answers):
    return sum(item["correct"] for item in answers) / len(answers) if answers else float("nan")


def group_by_reader(answers):
    groups = collections.defaultdict(list)
    for item in answers:
        groups[item["reader_id"]].append(item)
    return groups


def resample(groups, rng):
    readers = list(groups)
    return [item for reader in (rng.choice(readers) for _ in readers) for item in groups[reader]]


def bootstrap_ci(answers, reps=4000, seed=1):
    groups = group_by_reader(answers)
    rng = random.Random(seed)
    draws = sorted(share(resample(groups, rng)) for _ in range(reps))
    return draws[int(0.025 * reps)], draws[int(0.975 * reps)]


def draft_minus_writers(answers):
    draft = [item for item in answers if item["condition"] == "draft"]
    writers = [item for item in answers if item["condition"] != "draft"]
    return share(draft) - share(writers) if draft and writers else None


def pooled_difference(answers, reps=4000, seed=7):
    groups = group_by_reader(answers)
    rng = random.Random(seed)
    gaps = (draft_minus_writers(resample(groups, rng)) for _ in range(reps))
    draws = sorted(gap for gap in gaps if gap is not None)
    return {
        "point": draft_minus_writers(answers),
        "lo": draws[int(0.025 * len(draws))],
        "hi": draws[int(0.975 * len(draws))],
    }


def fisher_one_sided(a, b, c, d):
    n1 = a + b
    n2 = c + d
    m = a + c
    total = math.comb(n1 + n2, m)
    return sum(math.comb(n1, x) * math.comb(n2, m - x) / total for x in range(a, min(n1, m) + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    answers = read_jsonl(DATA / "reading_test.jsonl")
    stats = {}
    for group_name, subset in [("all", answers), ("degree", [item for item in answers if item["degree"] == "yes"])]:
        for version in VERSIONS:
            rows = [item for item in subset if item["condition"] == version]
            found = sum(item["correct"] for item in rows)
            lo, hi = bootstrap_ci(rows)
            stats[f"{group_name}_{version}"] = {"n": len(rows), "k": found, "lo": lo, "hi": hi}
        draft = [item for item in subset if item["condition"] == "draft"]
        writers = [item for item in subset if item["condition"] != "draft"]
        found_draft = sum(item["correct"] for item in draft)
        found_writers = sum(item["correct"] for item in writers)
        stats[f"{group_name}_fisher_one_sided"] = round(
            fisher_one_sided(found_draft, len(draft) - found_draft, found_writers, len(writers) - found_writers),
            4,
        )
        stats[f"{group_name}_by_position"] = {
            str(pos): round(100.0 * share([item for item in subset if item["position"] == pos]), 1)
            for pos in sorted({item["position"] for item in subset})
        }
    stats["pooled_diff"] = pooled_difference(answers)
    stats["summary"] = {
        "readers": len({item["reader_id"] for item in answers}),
        "answers": len(answers),
        "completed_all_five": len({item["reader_id"] for item in answers if item["completed"] == 1}),
        "degree_holders": len({item["reader_id"] for item in answers if item["degree"] == "yes"}),
    }
    (OUT / "reading_test.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    diff = stats["pooled_diff"]
    print(
        f"Reading test: {stats['summary']['readers']} readers, {stats['summary']['answers']} answers; "
        f"draft = {100 * stats['all_draft']['k'] / stats['all_draft']['n']:.1f}%, "
        f"supervised = {100 * stats['all_v1']['k'] / stats['all_v1']['n']:.1f}%, "
        f"adversarial = {100 * stats['all_ral']['k'] / stats['all_ral']['n']:.1f}%; "
        f"pooled diff = {100 * diff['point']:.1f} pp ({100 * diff['lo']:.1f}, {100 * diff['hi']:.1f})"
    )


if __name__ == "__main__":
    main()
