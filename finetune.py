"""Fine-tune a model to answer with the label, then evaluate it. Settings are in config.yaml.

    python finetune.py --model qwen3-4b --method lora
    python finetune.py --model qwen3-4b --method qlora
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          get_linear_schedule_with_warmup)

from llm import (CFG, DATA, EVAL_SETS, PRED, PRIVATE, batches, chat_prompt,
                 classification_messages, generate_answers, predict_file)

FT = CFG["finetune"]
ADAPTERS = Path("outputs/adapters")


def load_base(name, method):
    spec = CFG["models"][name]
    tok = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gpu = torch.cuda.is_available()
    quant = None
    if method == "qlora":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"], quantization_config=quant,
        dtype=torch.bfloat16 if gpu else torch.float32, device_map="cuda" if gpu else "cpu",
        attn_implementation="eager" if spec["family"] == "gemma2" else "sdpa")
    if quant is not None:
        # Same setup as LoRA otherwise: no gradient checkpointing.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    return model, tok, spec


def training_examples(tok, spec, limit):
    """Class-balanced sample of the train split, tokenized as prompt + label word."""
    train = pd.read_csv(DATA / "davidson_train.csv").drop_duplicates("text")
    n = limit or FT["per_class"]
    sample = pd.concat([g.sample(n, replace=len(g) < n, random_state=CFG["seed"])
                        for _, g in train.groupby("label_name")])
    sample = sample.sample(frac=1, random_state=CFG["seed"])
    examples = []
    for text, label in zip(sample.text, sample.label_name):
        prompt = tok(chat_prompt(tok, spec, classification_messages(text, [])), add_special_tokens=False).input_ids
        answer = tok(label, add_special_tokens=False).input_ids
        examples.append((prompt + answer, [-100] * len(prompt) + answer))
    return examples


def collate(batch, pad_id, device):
    width = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), width), pad_id)
    labels = torch.full((len(batch), width), -100)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, y) in enumerate(batch):   # right padding for training
        ids[i, :len(x)], labels[i, :len(y)], mask[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids.to(device), mask.to(device), labels.to(device)


def train(model, tok, spec, limit):
    lc = FT["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    examples = training_examples(tok, spec, limit)
    steps = FT["epochs"] * math.ceil(len(examples) / FT["batch_size"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=FT["lr"], weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, int(FT["warmup"] * steps), steps)
    rng = np.random.default_rng(CFG["seed"])
    model.train()
    losses, step = [], 0
    for epoch in range(FT["epochs"]):
        order = rng.permutation(len(examples))
        for idx in batches(order, FT["batch_size"]):
            ids, mask, labels = collate([examples[i] for i in idx], tok.pad_token_id, model.device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
            step += 1
            if step % 20 == 0 or step == steps:
                print(f"step {step}/{steps}  loss {np.mean(losses[-20:]):.4f}", flush=True)
    info = {"train_examples": len(examples), "steps": steps, "final_loss": float(np.mean(losses[-20:])),
            "trainable_params": sum(p.numel() for p in params),
            "total_params": sum(p.numel() for p in model.parameters())}
    return model, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name from config.yaml")
    ap.add_argument("--method", required=True, choices=["lora", "qlora"])
    ap.add_argument("--limit", type=int, help="quick test: N training examples per class, N posts per set")
    args = ap.parse_args()
    torch.manual_seed(CFG["seed"])
    suffix = "_quick" if args.limit else ""
    name = f"{args.model}+{args.method}"
    adapter = ADAPTERS / f"{name}{suffix}"
    info_path = PRED / f"{name}{suffix}.json"

    model, tok, spec = load_base(args.model, args.method)
    if adapter.exists():
        model = PeftModel.from_pretrained(model, adapter)
        print("loaded adapter:", adapter)
    else:
        gpu = torch.cuda.is_available()
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        weights_gb = round(torch.cuda.memory_allocated() / 2**30, 2) if gpu else None
        start = time.time()
        model, info = train(model, tok, spec, args.limit)
        info |= {"model": spec["id"], "revision": spec["revision"], "method": args.method,
                 "train_seconds": round(time.time() - start, 1), "model_memory_gb": weights_gb,
                 "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if gpu else None}
        model.save_pretrained(adapter)
        PRED.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(info, indent=2))
        print(json.dumps(info))

    model.eval()
    tok.padding_side = "left"
    for dataset in EVAL_SETS:
        out = PRED / f"{name}__{dataset}__0shot{suffix}.csv"
        if out.exists():
            print("exists:", out)
            continue
        predict_file(model, tok, spec, dataset, [], out, args.limit)

    # Refusal test: does fine-tuning change safety behaviour?
    out = PRIVATE / f"{name}__responses{suffix}.csv"
    if out.exists():
        print("exists:", out)
    else:
        req = pd.read_csv(DATA / "refusal_requests.csv")
        if args.limit:
            req = req.groupby("level").head(1)
        req["response"] = generate_answers(model, tok, spec, req["text"].tolist())
        PRIVATE.mkdir(parents=True, exist_ok=True)
        req.to_csv(out, index=False)
        print(f"{len(req)} responses -> {out}")


if __name__ == "__main__":
    main()
