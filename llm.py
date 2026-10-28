"""Settings are in config.yaml.

    python llm.py download                       download model weights (login node)
    python llm.py classify --model qwen3-4b      label posts as hate / offensive / neither
    python llm.py guard                          Llama Guard 3 on the same posts
    python llm.py refusal --model qwen3-4b       answer the refusal test requests
    python llm.py judge                          judge all refusal answers

Add --limit N to run on the first N items only (quick test, saved with a _quick suffix).
Results that already exist are not computed again.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

CFG = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
LABELS = CFG["labels"]
DATA = Path("data/processed")
PRED = Path("outputs/predictions")
PRIVATE = Path("outputs/private")   # raw model answers to the refusal test (git-ignored)
EVAL_SETS = {"davidson_test": "orig_id", "hatecheck": "case_id"}


# model helpers

def load_model(name):
    spec = CFG["models"][name]
    tok = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"],
        dtype=torch.bfloat16 if gpu else torch.float32,
        device_map="cuda" if gpu else "cpu",
        # Gemma 2 needs eager attention for its attention soft-capping.
        attn_implementation="eager" if spec["family"] == "gemma2" else "sdpa",
    ).eval()
    return model, tok, spec


def chat_prompt(tok, spec, messages):
    if not spec.get("system_role", True) and messages[0]["role"] == "system":
        first = {"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}
        messages = [first] + messages[2:]
    extra = {"enable_thinking": False} if spec["family"] in ("qwen3", "test") else {}
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def encode(tok, prompts, device):
    return tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)


def save_info(path, spec, seconds):
    info = {"model": spec["id"], "revision": spec["revision"], "seconds": round(seconds, 1),
            "torch": torch.__version__, "transformers": transformers.__version__}
    path.write_text(json.dumps(info, indent=2))


# classification
def pick_fewshot(train, seed):
    t = train["text"]
    pool = train[(train["agreement"] == 1) & (train["dup_count"] == 1)
                 & t.str.len().between(30, 140)
                 & ~t.str.contains('["“”]|URL') & (t.str.count("@USER") <= 1)]
    shots = pd.concat([pool[pool["label_name"] == c].sample(1, random_state=seed) for c in LABELS])
    return shots.sample(frac=1, random_state=seed)[["text", "label_name"]].values.tolist()


def classification_messages(text, shots):
    messages = [{"role": "system", "content": CFG["classification"]["system_prompt"]}]
    for shot_text, shot_label in shots:
        messages += [{"role": "user", "content": f"Post: {shot_text}"},
                     {"role": "assistant", "content": shot_label}]
    return messages + [{"role": "user", "content": f"Post: {text}"}]


@torch.no_grad()
def score_labels(model, tok, prompts, batch_size):
    """Probability of each label as the model's answer.

    The three label words start with different tokens, so the first token of the answer
    decides the label. For every label, the probabilities of the first tokens of its
    spellings ("hate", "Hate") are added up. Returns the probabilities normalised over
    the three labels, and the total probability of all label tokens (label_mass): a low
    mass means the model would rather have answered something else."""
    first_ids = {lab: sorted({tok(w, add_special_tokens=False).input_ids[0]
                              for w in CFG["classification"]["label_words"][lab]})
                 for lab in LABELS}
    all_ids = [i for ids in first_ids.values() for i in ids]
    assert len(all_ids) == len(set(all_ids)), "labels share a first token"

    probs = []
    for batch in batches(prompts, batch_size):
        enc = encode(tok, batch, model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        p = torch.softmax(logits, dim=-1)
        probs.append(torch.stack([p[:, first_ids[lab]].sum(-1) for lab in LABELS], dim=1).cpu())
    probs = torch.cat(probs).numpy()
    mass = probs.sum(1)
    return probs / mass[:, None], mass


def classify(args):
    train = pd.read_csv(DATA / "davidson_train.csv")
    shots = pick_fewshot(train, CFG["seed"])
    suffix = "_quick" if args.limit else ""
    model = None
    for dataset, id_col in EVAL_SETS.items():
        for k in (0, 3):
            out = PRED / f"{args.model}__{dataset}__{k}shot{suffix}.csv"
            if out.exists():
                print("exists:", out)
                continue
            if model is None:
                model, tok, spec = load_model(args.model)
                start = time.time()
            df = pd.read_csv(DATA / f"{dataset}.csv")
            df = df.head(args.limit) if args.limit else df
            prompts = [chat_prompt(tok, spec, classification_messages(t, shots[:k])) for t in df["text"]]
            probs, mass = score_labels(model, tok, prompts, CFG["classification"]["batch_size"])
            res = pd.DataFrame({"id": df[id_col], "label": df["label_name"],
                                "pred": np.array(LABELS)[probs.argmax(1)], "label_mass": mass})
            for i, lab in enumerate(LABELS):
                res[f"p_{lab}"] = probs[:, i]
            PRED.mkdir(parents=True, exist_ok=True)
            res.to_csv(out, index=False)
            print(f"{out.name}: accuracy {(res.pred == res.label).mean():.3f}, "
                  f"mean label mass {mass.mean():.3f}")
    if model is not None:
        save_info(PRED / f"{args.model}{suffix}.json", spec, time.time() - start)


# Llama Guard

@torch.no_grad()
def guard_verdicts(model, tok, conversations, batch_size=32):
    """Llama Guard 3 verdict for each conversation: unsafe (bool), violated categories
    (e.g. "S10"), and p_unsafe = P(unsafe) / (P(safe) + P(unsafe)) at the verdict word."""
    safe_id = tok("safe", add_special_tokens=False).input_ids[0]
    unsafe_id = tok("unsafe", add_special_tokens=False).input_ids[0]
    rows = []
    for batch in batches(conversations, batch_size):
        enc = encode(tok, [tok.apply_chat_template(c, tokenize=False) for c in batch], model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=12, output_scores=True,
                             return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        new = gen.sequences[:, enc.input_ids.shape[1]:]
        for b in range(len(batch)):
            lines = tok.decode(new[b], skip_special_tokens=True).split()
            unsafe = lines[0] == "unsafe" if lines else None
            cats = lines[1] if unsafe and len(lines) > 1 else ""
            # The verdict word comes after a few newline tokens.
            step = next((s for s, t in enumerate(new[b].tolist()) if t in (safe_id, unsafe_id)), None)
            p_unsafe = np.nan
            if step is not None:
                p = torch.softmax(gen.scores[step][b].float(), dim=-1)
                p_unsafe = (p[unsafe_id] / (p[unsafe_id] + p[safe_id])).item()
            rows.append({"unsafe": unsafe, "categories": cats, "p_unsafe": p_unsafe})
    return pd.DataFrame(rows)


def guard(args):
    suffix = "_quick" if args.limit else ""
    model = None
    for dataset, id_col in EVAL_SETS.items():
        out = PRED / f"llama-guard__{dataset}{suffix}.csv"
        if out.exists():
            print("exists:", out)
            continue
        if model is None:
            model, tok, spec = load_model("llama-guard")
            start = time.time()
        df = pd.read_csv(DATA / f"{dataset}.csv")
        df = df.head(args.limit) if args.limit else df
        res = guard_verdicts(model, tok, [[{"role": "user", "content": t}] for t in df["text"]])
        res.insert(0, "id", df[id_col].values)
        res.insert(1, "label", df["label_name"].values)
        # Hate according to Llama Guard: unsafe with category S10 (Hate).
        res["pred_hate"] = res["categories"].str.split(",").apply(lambda c: "S10" in c)
        PRED.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
        print(f"{out.name}: unsafe {res.unsafe.mean():.3f}, S10 {res.pred_hate.mean():.3f}")
    if model is not None:
        save_info(PRED / f"llama-guard{suffix}.json", spec, time.time() - start)


# refusal test

def refusal(args):
    suffix = "_quick" if args.limit else ""
    out = PRIVATE / f"{args.model}__responses{suffix}.csv"
    if out.exists():
        print("exists:", out)
        return
    req = pd.read_csv(DATA / "refusal_requests.csv")
    if args.limit:
        req = req.groupby("level").head(args.limit)
    model, tok, spec = load_model(args.model)
    start = time.time()
    answers = []
    with torch.no_grad():
        for batch in batches(req["text"].tolist(), CFG["refusal"]["batch_size"]):
            prompts = [chat_prompt(tok, spec, [{"role": "user", "content": r}]) for r in batch]
            enc = encode(tok, prompts, model.device)
            seq = model.generate(**enc, do_sample=False, pad_token_id=tok.pad_token_id,
                                 max_new_tokens=CFG["refusal"]["max_new_tokens"])
            answers += tok.batch_decode(seq[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    req["response"] = [a.strip() for a in answers]
    PRIVATE.mkdir(parents=True, exist_ok=True)
    req.to_csv(out, index=False)
    save_info(PRIVATE / f"{args.model}__responses{suffix}.json", spec, time.time() - start)
    print(f"{len(req)} responses -> {out}")


def is_refusal(response):
    head = str(response)[:300].lower().replace("’", "'")
    return any(re.search(p, head) for p in CFG["refusal"]["patterns"])


def judge(args):
    """Add two judgements to every response file: refused (refusal pattern found) and
    harmful (Llama Guard rates the response unsafe)."""
    model = None
    for path in sorted(PRIVATE.glob("*__responses.csv")):
        out = path.with_name(path.name.replace("__responses", "__judged"))
        if out.exists() and out.stat().st_mtime > path.stat().st_mtime:
            print("exists:", out)
            continue
        df = pd.read_csv(path)
        df["response"] = df["response"].fillna("")
        df["refused"] = df["response"].map(is_refusal)
        if not args.no_guard:
            if model is None:
                model, tok, _ = load_model("llama-guard")
            conv = [[{"role": "user", "content": q}, {"role": "assistant", "content": r}]
                    for q, r in zip(df["text"], df["response"])]
            v = guard_verdicts(model, tok, conv)
            df["harmful"] = v["unsafe"].values
            df["harm_categories"] = v["categories"].values
        df.to_csv(out, index=False)
        print(f"{out.name}: refused {df.refused.mean():.3f}"
              + (f", harmful {df.harmful.mean():.3f}" if "harmful" in df else ""))


# download

def download(args):
    from huggingface_hub import snapshot_download
    for name, spec in CFG["models"].items():
        try:
            snapshot_download(spec["id"], revision=spec["revision"],
                              allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"],
                              ignore_patterns=["original/*"])
            print(f"{name}: ok")
        except Exception as e:
            print(f"{name}: FAILED ({type(e).__name__})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "classify", "guard", "refusal", "judge"])
    ap.add_argument("--model", help="model name from config.yaml")
    ap.add_argument("--limit", type=int, help="only the first N items (quick test)")
    ap.add_argument("--no-guard", action="store_true", help="judge: keyword check only")
    args = ap.parse_args()
    torch.manual_seed(CFG["seed"])
    globals()[args.command](args)
