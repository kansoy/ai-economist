import argparse
import concurrent.futures as cf
import json
import os
import re

from adversarial_trainer import normalise_punctuation
from prompts import FIDELITY_SYSTEM, FIDELITY_USER


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def call_annotator_api(model, system_prompt, user_prompt):
    if model.startswith("gpt-"):
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=2048,
        )
        return response.output_text.strip()
    if model.startswith("claude-"):
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=2048,
        )
        return response.content[0].text.strip()
    raise ValueError(f"Unsupported annotator model: {model}")


def annotate_pair(job, drafts, versions):
    model, version, item_id = job
    prompt = FIDELITY_USER.replace("{src}", normalise_punctuation(drafts[item_id])).replace(
        "{ed}", normalise_punctuation(versions[version][item_id])
    )
    last_error = ""
    for _ in range(3):
        try:
            raw = call_annotator_api(model, FIDELITY_SYSTEM, prompt)
            match = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(match.group(0))
            verdict = str(parsed.get("verdict", "")).strip().lower()
            if verdict not in ("same", "minor", "major"):
                raise ValueError(f"Invalid verdict: {verdict}")
            return {
                "id": item_id,
                "version": version,
                "annotator": model,
                "verdict": verdict,
                "dropped": len(parsed.get("dropped") or []),
                "added": len(parsed.get("added") or []),
                "hedge": len(parsed.get("hedge") or []),
                "entity": len(parsed.get("entity") or []),
                "meaning": len(parsed.get("meaning") or []),
            }
        except Exception as exc:
            last_error = str(exc)[:200]
    return {"id": item_id, "version": version, "annotator": model, "verdict": None, "error": last_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--version-map", required=True)
    parser.add_argument("--annotators", nargs="+", default=["gpt-6.1-sol", "claude-opus-5-5"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    inputs = {row["id"]: row for row in read_jsonl(args.inputs)}
    drafts = {k: row["draft"] for k, row in inputs.items()}
    version_paths = json.loads(open(args.version_map, encoding="utf-8").read())
    versions = {"A": {k: row["human"] for k, row in inputs.items()}}
    for label, path in version_paths.items():
        rows = read_jsonl(path)
        field = "T" if "T" in rows[0] else "R"
        versions[label] = {row["id"]: row[field] for row in rows}

    jobs = [
        (model, version, item_id)
        for item_id in sorted(inputs)
        for version in versions
        for model in args.annotators
        if item_id in versions[version]
    ]
    with cf.ThreadPoolExecutor(args.workers) as executor, open(args.out, "w", encoding="utf-8") as handle:
        for record in executor.map(lambda job: annotate_pair(job, drafts, versions), jobs):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
