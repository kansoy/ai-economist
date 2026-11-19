from collections import Counter, deque
import gc
import json
import math
from pathlib import Path
import random
import re
import time

import torch
import torch.nn.functional as F

from prompts import INTERNAL_JUDGE_USER

NUMBER_RE = re.compile(r"(?<![A-Za-z\\])[-+\u2212]?\d[\d,]*(?:\.\d+)?\s?%?")

DEFAULTS = {
    "P": 8,
    "K": 4,
    "gen_bs": 16,
    "gen_temp": 0.7,
    "max_new": 400,
    "eval_temp": 0.7,
    "eval_top_p": 0.95,
    "eval_max_new": 450,
    "lr_writer": 5e-6,
    "lr_judge": 2e-5,
    "beta": 0.02,
    "mb": 4,
    "judge_mb": 4,
    "ema": 0.9,
    "gate_acc": 0.9,
    "max_skip": 20,
    "warmup_steps": 0,
    "warmup_acc": 0.75,
    "replay": 32,
    "buffer": 2000,
    "len_lo": 0.7,
    "len_hi": 1.4,
    "floor": -10.0,
    "clip_logodds": 8.0,
    "adv_clip": 3.0,
    "steps": 100,
    "max_hours": 22.5,
    "eval_every": 50,
    "ckpt_every": 10,
    "keep_every": 25,
    "seed": 20261115,
    "len_neutral": True,
    "len_decay": 0.9,
    "len_prior": 0.05,
    "len_clip": 40.0,
    "pause_acc": 0.60,
    "resume_acc": 0.65,
    "max_pause": 15,
}


def extract_numbers(text):
    return Counter(
        token.replace(",", "").replace("\u2212", "-").replace(" ", "").lstrip("+")
        for token in NUMBER_RE.findall(text)
    )


def normalise_punctuation(text):
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def within_group_length_sums(rewards, log_lengths, gate, group_size):
    sxy = 0.0
    sxx = 0.0
    for group in range(len(rewards) // group_size):
        indices = [i for i in range(group * group_size, (group + 1) * group_size) if gate[i]]
        if len(indices) < 2:
            continue
        mean_l = sum(log_lengths[i] for i in indices) / len(indices)
        mean_r = sum(rewards[i] for i in indices) / len(indices)
        sxy += sum((log_lengths[i] - mean_l) * (rewards[i] - mean_r) for i in indices)
        sxx += sum((log_lengths[i] - mean_l) ** 2 for i in indices)
    return sxy, sxx


def adjust_for_length(rewards, log_lengths, gate, group_size, slope):
    adjusted = list(rewards)
    for group in range(len(rewards) // group_size):
        indices = [i for i in range(group * group_size, (group + 1) * group_size) if gate[i]]
        if not indices:
            continue
        mean_l = sum(log_lengths[i] for i in indices) / len(indices)
        for i in indices:
            adjusted[i] = rewards[i] - slope * (log_lengths[i] - mean_l)
    return adjusted


def draw_from_pool(items, count, step, seed, pool_index):
    size = len(items)
    selected = []
    for idx in range((step - 1) * count, step * count):
        order = list(range(size))
        random.Random(f"{seed}-{pool_index}-{idx // size}").shuffle(order)
        selected.append(items[order[idx % size]])
    return selected


class AdversarialTrainer:
    def __init__(self, model, tokenizer, out_dir, cfg=None, log=print):
        self.model = model
        self.tok = tokenizer
        self.log = log
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.out = Path(out_dir)
        (self.out / "eval").mkdir(parents=True, exist_ok=True)
        self.dev = next(model.parameters()).device
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        gen_cfg = getattr(model, "generation_config", None)
        eos_val = getattr(gen_cfg, "eos_token_id", None) or tokenizer.eos_token_id
        self.eos = set(eos_val if isinstance(eos_val, (list, tuple)) else [eos_val])
        for token in ("<turn|>", "<end_of_turn>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
                self.eos.add(token_id)
        self.tok_a = self._single_token_id("A")
        self.tok_b = self._single_token_id("B")
        self.ema = None
        self.skip = 0
        self.buffer = deque(maxlen=self.cfg["buffer"])
        self.len_sxy = 0.0
        self.len_sxx = 0.0
        self.paused = False
        self.pause_run = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.opt_w = torch.optim.AdamW(self._adapter_params("writer"), lr=self.cfg["lr_writer"], weight_decay=0.0)
        self.opt_j = torch.optim.AdamW(self._adapter_params("judge"), lr=self.cfg["lr_judge"], weight_decay=0.0)

    def _single_token_id(self, char):
        for candidate in (char, " " + char):
            token_ids = self.tok(candidate, add_special_tokens=False)["input_ids"]
            if len(token_ids) == 1:
                return token_ids[0]
        raise ValueError(f"Token is not a single ID: {char}")

    def _adapter_params(self, name):
        params = [p for n, p in self.model.named_parameters() if f".{name}." in n and "lora_" in n]
        if not params:
            raise ValueError(f"No LoRA parameters found for adapter {name}")
        return params

    def chat_text(self, user_prompt):
        messages = [{"role": "user", "content": user_prompt}]
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, token_ids):
        return self.tok.decode([t for t in token_ids if t not in self.eos], skip_special_tokens=True).strip()

    def use_adapter(self, name, train=False):
        self.model.set_adapter(name)
        self.model.train(train)

    def _retry_oom(self, key, fn, opt=None):
        while True:
            try:
                return fn()
            except torch.cuda.OutOfMemoryError:
                if self.cfg[key] <= 1:
                    raise
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            self.cfg[key] = max(1, self.cfg[key] // 2)
            self.log({"memory_error": f"{key} halved", key: self.cfg[key]})

    @torch.no_grad()
    def generate(self, prompts, temperature, top_p=1.0, max_new=None):
        self.use_adapter("writer", False)
        outputs = []
        idx = 0
        while idx < len(prompts):
            bs = self.cfg["gen_bs"]
            batch = prompts[idx : idx + bs]
            max_len = max(map(len, batch))
            input_ids = torch.tensor([[self.pad] * (max_len - len(e)) + e for e in batch], device=self.dev)
            mask = torch.tensor([[0] * (max_len - len(e)) + [1] * len(e) for e in batch], device=self.dev)
            try:
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=mask,
                    max_new_tokens=max_new or self.cfg["max_new"],
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.pad,
                )
            except torch.cuda.OutOfMemoryError:
                if bs <= 2:
                    raise
                del input_ids, mask
                gc.collect()
                torch.cuda.empty_cache()
                self.cfg["gen_bs"] = max(2, bs // 2)
                self.log({"memory_error": "gen_bs halved", "gen_bs": self.cfg["gen_bs"]})
                continue
            for row in generated[:, max_len:].tolist():
                completion = []
                for token in row:
                    completion.append(token)
                    if token in self.eos:
                        break
                outputs.append(completion)
            idx += bs
        return outputs

    def _forward_logp(self, prompts, completions):
        lp_len = max(map(len, prompts))
        lc_len = max(map(len, completions))
        input_ids = torch.tensor(
            [[self.pad] * (lp_len - len(p)) + p + c + [self.pad] * (lc_len - len(c)) for p, c in zip(prompts, completions)],
            device=self.dev,
        )
        mask = torch.tensor(
            [[0] * (lp_len - len(p)) + [1] * (len(p) + len(c)) + [0] * (lc_len - len(c)) for p, c in zip(prompts, completions)],
            device=self.dev,
        )
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        try:
            logits = self.model(
                input_ids=input_ids, attention_mask=mask, position_ids=pos, logits_to_keep=lc_len + 1, use_cache=False
            ).logits
        except TypeError:
            logits = self.model(input_ids=input_ids, attention_mask=mask, position_ids=pos, use_cache=False).logits[
                :, -(lc_len + 1) :
            ]
        logits = logits[:, :-1].float() / self.cfg["gen_temp"]
        targets = input_ids[:, lp_len:]
        logp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none").view(
            targets.shape
        )
        return logp, mask[:, lp_len:].float()

    @torch.no_grad()
    def ref_logp(self, prompts, completions):
        self.use_adapter("ref", False)
        results = []
        mb = self.cfg["mb"]
        for i in range(0, len(prompts), mb):
            logp, mask = self._forward_logp(prompts[i : i + mb], completions[i : i + mb])
            results += [logp[j, : int(mask[j].sum())].detach() for j in range(logp.size(0))]
        return results

    def _judge_ids(self, where, text_a, text_b):
        prompt = INTERNAL_JUDGE_USER.format(
            where=where, a=normalise_punctuation(text_a), b=normalise_punctuation(text_b)
        )
        return self.encode(self.chat_text(prompt))

    def _judge_logits(self, items):
        max_len = max(map(len, items))
        input_ids = torch.tensor([[self.pad] * (max_len - len(e)) + e for e in items], device=self.dev)
        mask = torch.tensor([[0] * (max_len - len(e)) + [1] * len(e) for e in items], device=self.dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        try:
            logits = self.model(
                input_ids=input_ids, attention_mask=mask, position_ids=pos, logits_to_keep=1, use_cache=False
            ).logits[:, -1]
        except TypeError:
            logits = self.model(input_ids=input_ids, attention_mask=mask, position_ids=pos, use_cache=False).logits[:, -1]
        return logits[:, [self.tok_a, self.tok_b]].float()

    def judge_prob_a(self, triples):
        encoded = [self._judge_ids(*triple) for triple in triples]

        @torch.no_grad()
        def run_batch():
            self.use_adapter("judge", False)
            probs = []
            for i in range(0, len(encoded), self.cfg["judge_mb"]):
                probs += torch.softmax(self._judge_logits(encoded[i : i + self.cfg["judge_mb"]]), -1)[:, 0].tolist()
            return probs

        return self._retry_oom("judge_mb", run_batch)

    def train_judge(self, pairs):
        examples = []
        for where, revision, human in pairs:
            examples.append((self._judge_ids(where, revision, human), 1))
            examples.append((self._judge_ids(where, human, revision), 0))
        random.shuffle(examples)

        def run_batch():
            self.use_adapter("judge", True)
            total_loss = 0.0
            mb = self.cfg["judge_mb"]
            for i in range(0, len(examples), mb):
                batch = examples[i : i + mb]
                loss = (
                    F.cross_entropy(
                        self._judge_logits([e for e, _ in batch]),
                        torch.tensor([y for _, y in batch], device=self.dev),
                        reduction="sum",
                    )
                    / len(examples)
                )
                loss.backward()
                total_loss += loss.item()
            return total_loss

        total_loss = self._retry_oom("judge_mb", run_batch, self.opt_j)
        torch.nn.utils.clip_grad_norm_(self._adapter_params("judge"), 1.0)
        self.opt_j.step()
        self.opt_j.zero_grad(set_to_none=True)
        return total_loss

    def step(self, batch, update_writer=True, bypass_gates=False):
        cfg = self.cfg
        k_samples = cfg["K"]
        t0 = time.time()
        unique_prompts = [self.encode(self.chat_text(item["prompt"])) for item in batch]
        prompts = [p for p in unique_prompts for _ in range(k_samples)]
        completions = self.generate(prompts, cfg["gen_temp"])
        texts = [self.decode(seq) for seq in completions]

        gate = []
        for i, text in enumerate(texts):
            item = batch[i // k_samples]
            dw = len(item["draft"].split())
            tw = len(text.split())
            passed = (
                bool(completions[i])
                and completions[i][-1] in self.eos
                and extract_numbers(text) == extract_numbers(item["draft"])
                and cfg["len_lo"] * dw <= tw <= cfg["len_hi"] * dw
            )
            gate.append(passed or bypass_gates)

        triples = []
        for i, text in enumerate(texts):
            item = batch[i // k_samples]
            triples += [(item["where"], text, item["human"]), (item["where"], item["human"], text)]
        prob_a = self.judge_prob_a(triples)
        p_author = [(prob_a[2 * i] + 1.0 - prob_a[2 * i + 1]) / 2.0 for i in range(len(texts))]
        acc = sum(((prob_a[2 * i] < 0.5) + (prob_a[2 * i + 1] > 0.5)) / 2.0 for i in range(len(texts))) / len(texts)

        eps = 1e-6
        rewards = [
            max(-cfg["clip_logodds"], min(cfg["clip_logodds"], math.log((p + eps) / (1.0 - p + eps)))) if g else cfg["floor"]
            for p, g in zip(p_author, gate)
        ]
        log_lengths = [
            math.log(max(1, len(text.split())) / max(1, len(batch[i // k_samples]["draft"].split())))
            for i, text in enumerate(texts)
        ]
        sxy, sxx = within_group_length_sums(rewards, log_lengths, gate, k_samples)
        self.len_sxy = cfg["len_decay"] * self.len_sxy + sxy
        self.len_sxx = cfg["len_decay"] * self.len_sxx + sxx
        b_len = max(-cfg["len_clip"], min(cfg["len_clip"], self.len_sxy / (self.len_sxx + cfg["len_prior"])))
        rewards_used = adjust_for_length(rewards, log_lengths, gate, k_samples, b_len) if cfg["len_neutral"] else rewards

        advantages = []
        for group in range(len(batch)):
            vals = rewards_used[group * k_samples : (group + 1) * k_samples]
            mean_v = sum(vals) / k_samples
            std_v = (sum((v - mean_v) ** 2 for v in vals) / k_samples) ** 0.5
            advantages += [
                max(-cfg["adv_clip"], min(cfg["adv_clip"], (v - mean_v) / std_v)) if std_v > 1e-6 else 0.0 for v in vals
            ]

        kl_mean = 0.0
        wrote = False
        if update_writer and any(abs(a) > 0 for a in advantages):
            keep = [i for i, a in enumerate(advantages) if abs(a) > 0 and completions[i]]
            pk = [prompts[i] for i in keep]
            ck = [completions[i] for i in keep]
            ak = [advantages[i] for i in keep]
            ntok = sum(len(seq) for seq in ck)

            def run_writer():
                ref = self.ref_logp(pk, ck)
                self.use_adapter("writer", True)
                kl_total = 0.0
                for i in range(0, len(pk), cfg["mb"]):
                    logp, mask = self._forward_logp(pk[i : i + cfg["mb"]], ck[i : i + cfg["mb"]])
                    ref_pad = torch.zeros_like(logp)
                    for j, tensor in enumerate(ref[i : i + cfg["mb"]]):
                        ref_pad[j, : tensor.numel()] = tensor
                    diff = ref_pad - logp
                    kl = torch.exp(diff) - diff - 1.0
                    adv_t = torch.tensor(ak[i : i + cfg["mb"]], device=self.dev).unsqueeze(1)
                    loss = ((-adv_t * logp + cfg["beta"] * kl) * mask).sum() / ntok
                    loss.backward()
                    kl_total += float((kl.detach() * mask).sum()) / ntok
                return kl_total

            kl_mean = self._retry_oom("mb", run_writer, self.opt_w)
            torch.nn.utils.clip_grad_norm_(self._adapter_params("writer"), 1.0)
            self.opt_w.step()
            self.opt_w.zero_grad(set_to_none=True)
            wrote = True

        self.ema = acc if self.ema is None else cfg["ema"] * self.ema + (1.0 - cfg["ema"]) * acc
        new_pairs = [(batch[i // k_samples]["where"], text, batch[i // k_samples]["human"]) for i, text in enumerate(texts) if text]
        warm = not update_writer
        judge_loss = None
        judge_trained = False
        if warm or self.ema < cfg["gate_acc"] or self.skip >= cfg["max_skip"]:
            replay = random.sample(list(self.buffer), min(cfg["replay"], len(self.buffer)))
            judge_loss = self.train_judge(new_pairs + replay)
            judge_trained = True
            self.skip = 0
        else:
            self.skip += 1
        self.buffer.extend(new_pairs)

        return {
            "reward_p": sum(p for p, g in zip(p_author, gate) if g) / max(1, sum(gate)),
            "gate_pass": sum(gate) / len(gate),
            "judge_acc": acc,
            "judge_ema": self.ema,
            "judge_trained": judge_trained,
            "judge_loss": judge_loss,
            "skip": self.skip,
            "writer_updated": wrote,
            "kl": kl_mean,
            "words": sum(len(text.split()) for text in texts) / len(texts),
            "reward_logodds": sum(r for r, g in zip(rewards, gate) if g) / max(1, sum(gate)),
            "len_ratio": sum(
                len(text.split()) / max(1, len(batch[i // k_samples]["draft"].split())) for i, text in enumerate(texts)
            )
            / len(texts),
            "groups_used": sum(
                any(abs(a) > 0 for a in advantages[g * k_samples : (g + 1) * k_samples]) for g in range(len(batch))
            ),
            "len_slope": b_len,
            "len_slope_step": sxy / sxx if sxx > 1e-9 else None,
            "samples": [
                {
                    "id": batch[i // k_samples].get("id"),
                    "pool": batch[i // k_samples].get("pool"),
                    "text": text,
                    "words": len(text.split()),
                    "draft_words": len(batch[i // k_samples]["draft"].split()),
                    "p_author": round(p_author[i], 4),
                    "passed": bool(gate[i]),
                    "reward": round(rewards[i], 4),
                    "reward_used": round(rewards_used[i], 4),
                    "adv": round(advantages[i], 4),
                }
                for i, text in enumerate(texts)
            ],
            "sec": round(time.time() - t0, 1),
        }

    def evaluate(self, eval_items, step):
        prompts = [self.encode(self.chat_text(item["prompt"])) for item in eval_items]
        completions = self.generate(prompts, self.cfg["eval_temp"], self.cfg["eval_top_p"], self.cfg["eval_max_new"])
        revisions = [self.decode(seq) for seq in completions]
        triples = [
            triple
            for item, rev in zip(eval_items, revisions)
            for triple in ((item["where"], rev, item["human"]), (item["where"], item["human"], rev))
        ]
        prob_a = self.judge_prob_a(triples)
        own = [(prob_a[2 * i] + 1.0 - prob_a[2 * i + 1]) / 2.0 for i in range(len(revisions))]
        out_file = self.out / "eval" / f"step_{step:04d}.jsonl"
        with open(out_file, "w", encoding="utf-8") as handle:
            for item, rev, prob in zip(eval_items, revisions, own):
                handle.write(
                    json.dumps(
                        {
                            "id": item["id"],
                            "set": item.get("set", ""),
                            "step": step,
                            "T": rev,
                            "own_judge_p_author": round(prob, 4),
                            "numbers_kept": extract_numbers(rev) == extract_numbers(item["draft"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary = {
            "eval_step": step,
            "own_judge_fooling": sum(prob > 0.5 for prob in own) / len(own),
            "own_judge_mean_p": sum(own) / len(own),
            "numbers_kept": sum(extract_numbers(rev) == extract_numbers(item["draft"]) for item, rev in zip(eval_items, revisions))
            / len(revisions),
            "len_ratio": sum(len(rev.split()) / max(1, len(item["draft"].split())) for item, rev in zip(eval_items, revisions))
            / len(revisions),
        }
        for cohort in sorted({item.get("set", "") for item in eval_items} - {""}):
            indices = [i for i, item in enumerate(eval_items) if item.get("set") == cohort]
            summary[f"{cohort}_own_judge_mean_p"] = sum(own[i] for i in indices) / len(indices)
        self.log(summary)
        return summary

    def seed_buffer(self, triples):
        self.buffer.extend(tuple(item) for item in triples)

    def save(self, step):
        ckpt_dir = self.out / "ckpt" / f"step_{step:04d}"
        for name in ("writer", "judge"):
            self.model.save_pretrained(ckpt_dir / name, selected_adapters=[name])
        with open(ckpt_dir / "state.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "step": step,
                    "ema": self.ema,
                    "skip": self.skip,
                    "buffer": list(self.buffer)[-400:],
                    "len_sxy": self.len_sxy,
                    "len_sxx": self.len_sxx,
                    "paused": self.paused,
                    "pause_run": self.pause_run,
                },
                handle,
                ensure_ascii=False,
            )
        old_ckpts = [
            path
            for path in sorted((self.out / "ckpt").glob("step_*"))
            if int(path.name.split("_")[1]) % self.cfg["keep_every"]
        ]
        for old in old_ckpts[:-2]:
            for path in sorted(old.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            old.rmdir()

    def resume(self):
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file

        ckpts = sorted((self.out / "ckpt").glob("step_*")) if (self.out / "ckpt").exists() else []
        ckpts = [path for path in ckpts if (path / "state.json").exists()]
        if not ckpts:
            return 0
        latest = ckpts[-1]
        for name in ("writer", "judge"):
            weights = next((latest / name).rglob("adapter_model.safetensors"))
            set_peft_model_state_dict(self.model, load_file(str(weights)), adapter_name=name)
        state = json.loads((latest / "state.json").read_text(encoding="utf-8"))
        self.ema = state["ema"]
        self.skip = state["skip"]
        self.len_sxy = state.get("len_sxy", 0.0)
        self.len_sxx = state.get("len_sxx", 0.0)
        self.paused = state.get("paused", False)
        self.pause_run = state.get("pause_run", 0)
        self.buffer.extend(tuple(item) for item in state["buffer"])
        self.log({"resumed_from": state["step"]})
        return state["step"]

    def run(self, pools, eval_items, bypass_gates=False):
        if sum(count for _, count in pools) != self.cfg["P"]:
            raise ValueError("Pool counts must sum to P")
        cfg = self.cfg
        (self.out / "samples").mkdir(exist_ok=True)
        tested = lambda step: (self.out / "eval" / f"step_{step:04d}.jsonl").exists()
        due = lambda step: step % cfg["eval_every"] == 0 or step == cfg["steps"]
        start = self.resume()
        if due(start) and not tested(start):
            self.evaluate(eval_items, start)
        t0 = time.time()
        last = start
        reason = "done"
        with open(self.out / "log.jsonl", "a", encoding="utf-8") as log_file:
            for step in range(start + 1, cfg["steps"] + 1):
                if (time.time() - t0) / 3600.0 > cfg["max_hours"]:
                    reason = "max_hours"
                    break
                if self.pause_run >= cfg["max_pause"]:
                    reason = "judge_below_floor"
                    break
                random.seed(cfg["seed"] * 7 + step)
                batch = [
                    item
                    for pool_idx, (items, count) in enumerate(pools)
                    for item in draw_from_pool(items, count, step, cfg["seed"], pool_idx)
                ]
                if self.ema is not None:
                    self.paused = self.ema < (cfg["resume_acc"] if self.paused else cfg["pause_acc"])
                self.pause_run = self.pause_run + 1 if self.paused else 0
                warm = step <= cfg["warmup_steps"] and (self.ema is None or self.ema < cfg["warmup_acc"])
                info = dict(
                    step=step,
                    warmup=warm,
                    paused=self.paused,
                    **self.step(batch, update_writer=not (warm or self.paused), bypass_gates=bypass_gates),
                )
                with open(self.out / "samples" / f"step_{step:04d}.jsonl", "w", encoding="utf-8") as sample_file:
                    for sample in info.pop("samples"):
                        sample_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                log_file.write(json.dumps(info) + "\n")
                log_file.flush()
                self.log(info)
                last = step
                if step % cfg["ckpt_every"] == 0 or step == cfg["steps"]:
                    self.save(step)
                if due(step):
                    self.evaluate(eval_items, step)
        self.log({"stopped": reason, "step": last})
        if last > start and not (self.out / "ckpt" / f"step_{last:04d}" / "state.json").exists():
            self.save(last)
        if reason == "judge_below_floor" and not tested(last):
            self.evaluate(eval_items, last)
        return last, reason
