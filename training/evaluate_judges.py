import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import random
import re

from adversarial_trainer import extract_numbers, normalise_punctuation
from prompts import OUTSIDE_JUDGE_SYSTEM, OUTSIDE_JUDGE_USER


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def call_judge_api(model, system_prompt, user_prompt):
    if model.startswith("gpt-"):
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=64,
        )
        return response.output_text.strip()
    if model.startswith("claude-"):
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=64,
        )
        return response.content[0].text.strip()
    raise ValueError(f"Unsupported judge model: {model}")


def score_pair(job, inputs):
    model, step, item_id, revision, first = job
    entry = inputs[item_id]
    texts = {"H": normalise_punctuation(entry["human"]), "T": normalise_punctuation(revision)}
    pos_a, pos_b = ("H", "T") if first == "H" else ("T", "H")
    prompt = OUTSIDE_JUDGE_USER.format(title=entry["title"], a=texts[pos_a], b=texts[pos_b])
    try:
        raw = call_judge_api(model, OUTSIDE_JUDGE_SYSTEM, prompt)
        match = re.search(r"\b([AB])\b", raw.strip().upper())
        pick = (pos_a if match.group(1) == "A" else pos_b) if match else "?"
    except Exception as exc:
        raw = f"ERROR {exc}"
        pick = "?"
    return {
        "step": step,
        "judge": model,
        "cond": "T",
        "id": item_id,
        "shown": pos_a + pos_b,
        "pick": pick,
        "raw": raw[:30],
    }


def paper_bootstrap_ci(per_paper, reps=2000, seed=1):
    papers = list(per_paper)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sample = [rng.choice(papers) for _ in papers]
        denom = sum(per_paper[p][1] for p in sample)
        draws.append(sum(per_paper[p][0] for p in sample) / denom if denom else 0.0)
    draws.sort()
    return round(100.0 * draws[int(0.025 * reps)], 1), round(100.0 * draws[int(0.975 * reps)], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--revisions", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--judges", nargs="+", default=["gpt-6.1-sol", "claude-opus-5-5"])
    parser.add_argument("--out-verdicts", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    inputs = {row["id"]: row for row in read_jsonl(args.inputs)}
    revisions = {row["id"]: row["T"] for row in read_jsonl(args.revisions) if row["id"] in inputs}
    jobs = [
        (model, args.step, item_id, text, first)
        for model in args.judges
        for item_id, text in revisions.items()
        for first in ("H", "T")
    ]
    with cf.ThreadPoolExecutor(args.workers) as executor:
        results = list(executor.map(lambda job: score_pair(job, inputs), jobs))

    with open(args.out_verdicts, "a", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.out_summary, "a", encoding="utf-8") as handle:
        for model in args.judges:
            for cohort in ("clean2026", "pre2023"):
                subset = {k: v for k, v in revisions.items() if inputs[k]["set"] == cohort}
                rows = [r for r in results if r["judge"] == model and r["id"] in subset and r["pick"] != "?"]
                per_paper = {}
                for row in rows:
                    paper = inputs[row["id"]]["paper"]
                    wins, count = per_paper.get(paper, (0, 0))
                    per_paper[paper] = (wins + int(row["pick"] == "T"), count + 1)
                fooled = sum(row["pick"] == "T" for row in rows)
                summary = {
                    "step": args.step,
                    "judge": model,
                    "set": cohort,
                    "fooled": fooled,
                    "n": len(rows),
                    "rate": round(100.0 * fooled / max(1, len(rows)), 1),
                    "ci95": list(paper_bootstrap_ci(per_paper)) if per_paper else None,
                    "numbers_kept": sum(
                        extract_numbers(subset[k]) == extract_numbers(inputs[k]["draft"]) for k in subset
                    ),
                    "paragraphs": len(subset),
                    "papers": len(per_paper),
                    "len_ratio": round(
                        sum(len(subset[k].split()) / len(inputs[k]["draft"].split()) for k in subset) / max(1, len(subset)),
                        3,
                    ),
                    "unanswered": sum(
                        r["judge"] == model and r["id"] in subset and r["pick"] == "?" for r in results
                    ),
                }
                handle.write(json.dumps(summary) + "\n")
                print(json.dumps(summary))


if __name__ == "__main__":
    main()
