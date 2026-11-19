import argparse
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from adversarial_trainer import AdversarialTrainer

BASE_MODEL = "google/gemma-4-31B-it"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_quantised_base(model_id=BASE_MODEL):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_length=2048):
        self.items = []
        for row in rows:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(row["target"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + target_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_supervised(batch, pad_id):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = [item["input_ids"] + [pad_id] * (max_len - len(item["input_ids"])) for item in batch]
    labels = [item["labels"] + [-100] * (max_len - len(item["labels"])) for item in batch]
    mask = [[1] * len(item["input_ids"]) + [0] * (max_len - len(item["input_ids"])) for item in batch]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def train_supervised_writer(train_path, val_path, output_dir, model_id=BASE_MODEL):
    model, tokenizer = load_quantised_base(model_id)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    train_ds = SupervisedDataset(read_jsonl(train_path), tokenizer)
    val_ds = SupervisedDataset(read_jsonl(val_path), tokenizer)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        num_train_epochs=2,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=150,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=150,
        per_device_eval_batch_size=2,
        optim="paged_adamw_8bit",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate_supervised(batch, tokenizer.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(Path(output_dir) / "adapter_final")


def pretrain_internal_judge(writer_adapter, judge_train_path, output_dir, model_id=BASE_MODEL, lr=1e-4, epochs=2, mb=4, accum=4):
    base, tokenizer = load_quantised_base(model_id)
    model = PeftModel.from_pretrained(base, str(writer_adapter), adapter_name="writer", is_trainable=False)
    model.load_adapter(str(writer_adapter), adapter_name="ref", is_trainable=False)
    writer_cfg = model.peft_config["writer"]
    model.add_adapter(
        "judge",
        LoraConfig(
            r=writer_cfg.r,
            lora_alpha=writer_cfg.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=writer_cfg.target_modules,
        ),
    )
    trainer = AdversarialTrainer(model, tokenizer, output_dir)
    rows = read_jsonl(judge_train_path)
    examples = []
    for row in rows:
        examples.append((trainer._judge_ids(row["where"], row["revision"], row["human"]), 1))
        examples.append((trainer._judge_ids(row["where"], row["human"], row["revision"]), 0))

    opt = torch.optim.AdamW(trainer._adapter_params("judge"), lr=lr, weight_decay=0.0)
    per_epoch = math.ceil(len(examples) / (mb * accum))
    total_steps = epochs * per_epoch
    warm = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)),
    )

    trainer.use_adapter("judge", True)
    step = 0
    for _ in range(epochs):
        random.shuffle(examples)
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(examples), mb * accum):
            chunk = examples[i : i + mb * accum]
            for j in range(0, len(chunk), mb):
                sub = chunk[j : j + mb]
                loss = (
                    F.cross_entropy(
                        trainer._judge_logits([e for e, _ in sub]),
                        torch.tensor([y for _, y in sub], device=trainer.dev),
                        reduction="sum",
                    )
                    / len(chunk)
                )
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer._adapter_params("judge"), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
    model.save_pretrained(Path(output_dir) / "judge_adapter", selected_adapters=["judge"])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    p_sft = sub.add_parser("writer")
    p_sft.add_argument("--train", required=True)
    p_sft.add_argument("--val", required=True)
    p_sft.add_argument("--out", required=True)
    p_sft.add_argument("--model", default=BASE_MODEL)

    p_judge = sub.add_parser("judge")
    p_judge.add_argument("--writer-adapter", required=True)
    p_judge.add_argument("--train", required=True)
    p_judge.add_argument("--out", required=True)
    p_judge.add_argument("--model", default=BASE_MODEL)

    args = parser.parse_args()
    if args.stage == "writer":
        train_supervised_writer(args.train, args.val, args.out, args.model)
    else:
        pretrain_internal_judge(args.writer_adapter, args.train, args.out, args.model)


if __name__ == "__main__":
    main()
