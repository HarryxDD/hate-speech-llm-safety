"""Fine-tune a model to answer with the label, then evaluate it. Settings are in config.yaml.

    python finetune.py --model qwen3-4b --method lora
    python finetune.py --model qwen3-4b --method qlora
    python finetune.py --model qwen3-8b --method qlora --target soft --format two_questions
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

from llm import (CFG, DATA, EVAL_SETS, LABELS, PIPELINE, PRED, PRIVATE, batches, chat_prompt,
                 classification_messages, eval_set, generate_answers, predict_file,
                 run_strategy, save_predictions)

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


def training_sample(limit):
    """Class-balanced sample of the train split."""
    train = pd.read_csv(DATA / "davidson_train.csv").drop_duplicates("text")
    n = limit or FT["per_class"]
    sample = pd.concat([g.sample(n, replace=len(g) < n, random_state=CFG["seed"])
                        for _, g in train.groupby("label_name")])
    return sample.sample(frac=1, random_state=CFG["seed"])


def training_examples(tok, spec, limit):
    """Training sample tokenized as prompt + label word."""
    sample = training_sample(limit)
    examples = []
    for text, label in zip(sample.text, sample.label_name):
        prompt = tok(chat_prompt(tok, spec, classification_messages(text, [])), add_special_tokens=False).input_ids
        answer = tok(label, add_special_tokens=False).input_ids
        examples.append((prompt + answer, [-100] * len(prompt) + answer))
    return examples


def first_token(tok, word):
    return tok(word, add_special_tokens=False).input_ids[0]


def target_examples(tok, spec, limit, target, fmt):
    """Examples of (prompt, answer tokens, target probabilities). soft uses the share of
    annotator votes, two_questions gives two yes/no examples per post."""
    sample = training_sample(limit)
    if target == "soft":
        q = sample[["votes_hate", "votes_offensive", "votes_neither"]].to_numpy(float)
        q = q / q.sum(1, keepdims=True)
    else:
        q = np.eye(3)[sample.label_name.map(LABELS.index).to_numpy()]
    encode_prompt = lambda text, system=None: tok(chat_prompt(tok, spec, classification_messages(text, [], system)),
                                                  add_special_tokens=False).input_ids
    examples = []
    if fmt == "single":
        label_ids = [first_token(tok, lab) for lab in LABELS]
        for text, probs in zip(sample.text, q):
            examples.append((encode_prompt(text), label_ids, probs.tolist()))
    else:
        P = CFG["prompting"]
        yes_no = [first_token(tok, "yes"), first_token(tok, "no")]
        for text, probs in zip(sample.text, q):
            for question, p_yes in (("question_target", probs[0]), ("question_insult", probs[0] + probs[1])):
                examples.append((encode_prompt(text, P[question]), yes_no, [p_yes, 1 - p_yes]))
    return examples


def collate_targets(batch, pad_id, device):
    """Left padding, so the answer is at the last position."""
    width = max(len(ids) for ids, _, _ in batch)
    ids = torch.full((len(batch), width), pad_id)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, _, _) in enumerate(batch):
        ids[i, width - len(x):], mask[i, width - len(x):] = torch.tensor(x), 1
    targets = torch.tensor([t for _, t, _ in batch])
    probs = torch.tensor([p for _, _, p in batch], dtype=torch.float)
    return ids.to(device), mask.to(device), targets.to(device), probs.to(device)


def target_loss(model, batch, pad_id):
    """Cross-entropy between the target probabilities and the answer token probabilities."""
    ids, mask, targets, probs = collate_targets(batch, pad_id, model.device)
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(1, targets)
    return -(probs * logp).sum(1).mean()


def collate(batch, pad_id, device):
    width = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), width), pad_id)
    labels = torch.full((len(batch), width), -100)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, y) in enumerate(batch):   # right padding for training
        ids[i, :len(x)], labels[i, :len(y)], mask[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids.to(device), mask.to(device), labels.to(device)


def train(model, tok, spec, limit, target="hard", fmt="single"):
    lc = FT["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    plain = target == "hard" and fmt == "single"
    examples = training_examples(tok, spec, limit) if plain else target_examples(tok, spec, limit, target, fmt)
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
            if plain:
                ids, mask, labels = collate([examples[i] for i in idx], tok.pad_token_id, model.device)
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            else:
                loss = target_loss(model, [examples[i] for i in idx], tok.pad_token_id)
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
    ap.add_argument("--target", default="hard", choices=["hard", "soft"],
                    help="hard: majority label; soft: share of annotator votes")
    ap.add_argument("--format", default="single", choices=["single", "two_questions"])
    ap.add_argument("--limit", type=int, help="quick test: N training examples per class, N posts per set")
    args = ap.parse_args()
    torch.manual_seed(CFG["seed"])
    suffix = "_quick" if args.limit else ""
    plain = args.target == "hard" and args.format == "single"
    name = f"{args.model}+{args.method}"
    if not plain:
        name += f"-{args.target}" + ("-2q" if args.format == "two_questions" else "")
    out_dir, private_dir = (PRED, PRIVATE) if plain else (PIPELINE, PRIVATE / "pipeline")
    adapter = ADAPTERS / f"{name}{suffix}"
    info_path = out_dir / f"{name}{suffix}.json"

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
        model, info = train(model, tok, spec, args.limit, args.target, args.format)
        info |= {"model": spec["id"], "revision": spec["revision"], "method": args.method,
                 "target": args.target, "format": args.format,
                 "train_seconds": round(time.time() - start, 1), "model_memory_gb": weights_gb,
                 "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if gpu else None}
        model.save_pretrained(adapter)
        out_dir.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(info, indent=2))
        print(json.dumps(info))

    model.eval()
    tok.padding_side = "left"
    strategy = "two_questions" if args.format == "two_questions" else "definitions"
    for dataset in EVAL_SETS:
        for inp in (["none"] if plain else CFG["pipeline"]["inputs"]):
            out = (PRED / f"{name}__{dataset}__0shot{suffix}.csv" if plain
                   else PIPELINE / f"{name}__{dataset}__{inp}{suffix}.csv")
            if out.exists():
                print("exists:", out)
                continue
            if plain:
                predict_file(model, tok, spec, dataset, [], out, args.limit)
            else:
                df = eval_set(dataset, args.limit)
                save_predictions(df, *run_strategy(model, tok, spec, strategy, df["text"].tolist(), inp), out)

    # Refusal test: does fine-tuning change safety behaviour?
    out = private_dir / f"{name}__responses{suffix}.csv"
    if out.exists():
        print("exists:", out)
    else:
        req = pd.read_csv(DATA / "refusal_requests.csv")
        if args.limit:
            req = req.groupby("level").head(1)
        req["response"] = generate_answers(model, tok, spec, req["text"].tolist())
        private_dir.mkdir(parents=True, exist_ok=True)
        req.to_csv(out, index=False)
        print(f"{len(req)} responses -> {out}")


if __name__ == "__main__":
    main()
