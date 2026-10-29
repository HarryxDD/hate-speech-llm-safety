"""Settings are in config.yaml.

    python llm.py download                       download model weights (login node)
    python llm.py classify --model qwen3-4b      label posts as hate / offensive / neither
                                                 (Davidson test, HateCheck, perturbed posts)
    python llm.py guard                          Llama Guard 3 on the same posts
    python llm.py refusal --model qwen3-4b       answer the refusal test requests
                                                 (original and perturbed)
    python llm.py judge                          judge all refusal answers
    python llm.py probe --model qwen3-4b         hidden states and per-layer probes

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
HIDDEN = Path("outputs/hidden")     # hidden states (.npy, git-ignored)
EVAL_SETS = {"davidson_test": "orig_id", "hatecheck": "case_id", "perturbed": "id"}
REQUEST_SETS = {"": "refusal_requests", "_perturbed": "refusal_requests_perturbed"}


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


def predict_file(model, tok, spec, dataset, shots, out, limit=None):
    """Label every post of an evaluation set and save the predictions to out."""
    df = pd.read_csv(DATA / f"{dataset}.csv")
    df = df.head(limit) if limit else df
    prompts = [chat_prompt(tok, spec, classification_messages(t, shots)) for t in df["text"]]
    probs, mass = score_labels(model, tok, prompts, CFG["classification"]["batch_size"])
    res = pd.DataFrame({"id": df[EVAL_SETS[dataset]], "label": df["label_name"],
                        "pred": np.array(LABELS)[probs.argmax(1)], "label_mass": mass})
    for i, lab in enumerate(LABELS):
        res[f"p_{lab}"] = probs[:, i]
    PRED.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"{out.name}: accuracy {(res.pred == res.label).mean():.3f}, mean label mass {mass.mean():.3f}")


def classify(args):
    train = pd.read_csv(DATA / "davidson_train.csv")
    shots = pick_fewshot(train, CFG["seed"])
    suffix = "_quick" if args.limit else ""
    model = None
    for dataset in EVAL_SETS:
        for k in (0, 3):
            out = PRED / f"{args.model}__{dataset}__{k}shot{suffix}.csv"
            if out.exists():
                print("exists:", out)
                continue
            if model is None:
                model, tok, spec = load_model(args.model)
                start = time.time()
            predict_file(model, tok, spec, dataset, shots[:k], out, args.limit)
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

@torch.no_grad()
def generate_answers(model, tok, spec, requests):
    answers = []
    for batch in batches(requests, CFG["refusal"]["batch_size"]):
        prompts = [chat_prompt(tok, spec, [{"role": "user", "content": r}]) for r in batch]
        enc = encode(tok, prompts, model.device)
        seq = model.generate(**enc, do_sample=False, pad_token_id=tok.pad_token_id,
                             max_new_tokens=CFG["refusal"]["max_new_tokens"])
        answers += tok.batch_decode(seq[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return [a.strip() for a in answers]


def refusal(args):
    suffix = "_quick" if args.limit else ""
    model = None
    for tag, name in REQUEST_SETS.items():
        out = PRIVATE / f"{args.model}__responses{tag}{suffix}.csv"
        if out.exists():
            print("exists:", out)
            continue
        if model is None:
            model, tok, spec = load_model(args.model)
        start = time.time()
        req = pd.read_csv(DATA / f"{name}.csv")
        if args.limit:
            req = req.groupby(["level"] + (["perturbation"] if tag else [])).head(args.limit)
        req["response"] = generate_answers(model, tok, spec, req["text"].tolist())
        PRIVATE.mkdir(parents=True, exist_ok=True)
        req.to_csv(out, index=False)
        save_info(out.with_suffix(".json"), spec, time.time() - start)
        print(f"{len(req)} responses -> {out}")


def is_refusal(response):
    head = str(response)[:300].lower().replace("’", "'")
    return any(re.search(p, head) for p in CFG["refusal"]["patterns"])


def judge(args):
    """Add two judgements to every response file: refused (refusal pattern found) and
    harmful (Llama Guard rates the response unsafe)."""
    model = None
    for path in sorted(PRIVATE.glob("*__responses*.csv")):
        if path.stem.endswith("_quick"):
            continue
        out = path.with_name(path.name.replace("__responses", "__judged"))
        if out.exists():
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


# hidden states and probes

def post_span(prompt, text):
    start = prompt.rfind(text)
    return (start, start + len(text)) if start >= 0 else (0, len(prompt))


@torch.no_grad()
def hidden_states(model, tok, spec, texts, batch_size=16):
    """Hidden states of every layer for the zero-shot classification prompt, pooled in two
    ways: the last prompt token (where the answer starts) and the mean over the tokens of
    the post. Returns two float16 arrays of shape (posts, layers + 1, hidden size)."""
    last, mean = [], []
    for batch in batches(list(texts), batch_size):
        prompts = [chat_prompt(tok, spec, classification_messages(t, [])) for t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                  return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        spans = torch.tensor([post_span(p, t) for p, t in zip(prompts, batch)])
        in_post = ((offsets[:, :, 0] >= spans[:, :1]) & (offsets[:, :, 1] <= spans[:, 1:])
                   & (offsets[:, :, 1] > offsets[:, :, 0]))
        out = model(**enc.to(model.device), output_hidden_states=True, logits_to_keep=1)
        hs = torch.stack(out.hidden_states, dim=1).float()   # (batch, layers + 1, tokens, hidden)
        last.append(hs[:, :, -1].half().cpu())
        w = in_post.to(hs.device, hs.dtype)[:, None, :, None]
        mean.append(((hs * w).sum(2) / w.sum(2).clamp(min=1)).half().cpu())
    return torch.cat(last).numpy(), torch.cat(mean).numpy()


def probe(args):
    """Logistic regression probe on the hidden states of every layer, 5-fold cross-validated
    on the balanced Davidson subset. Control: the same with shuffled labels. Each perturbed
    post is labelled by the probe of the fold in which its original post was held out."""
    import os
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    suffix = "_quick" if args.limit else ""
    out = PRED / f"{args.model}__probe{suffix}.csv"
    if out.exists():
        print("exists:", out)
        return
    clean = pd.read_csv(DATA / "davidson_balanced.csv")
    pert = pd.read_csv(DATA / "perturbed.csv")
    pert = pert[pert.valid & (pert.source == "davidson")]
    if args.limit:
        clean = clean.head(args.limit)
        pert = pert[pert.orig_id.isin(clean.orig_id)]
    pert = pert.reset_index(drop=True)

    HIDDEN.mkdir(parents=True, exist_ok=True)
    model, X = None, {}
    for name, texts in [("clean", clean.text), ("perturbed", pert.text)]:
        paths = {pool: HIDDEN / f"{args.model}__{name}__{pool}{suffix}.npy" for pool in ("last", "mean")}
        if not all(p.exists() for p in paths.values()):
            if model is None:
                model, tok, spec = load_model(args.model)
                start = time.time()
            for pool, arr in zip(("last", "mean"), hidden_states(model, tok, spec, texts)):
                np.save(paths[pool], arr)
        X[name] = {pool: np.load(p) for pool, p in paths.items()}
    if model is not None:
        save_info(HIDDEN / f"{args.model}{suffix}.json", spec, time.time() - start)
        del model
        torch.cuda.empty_cache()

    y = clean.label_name.values
    y_control = np.random.default_rng(CFG["seed"]).permutation(y)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=CFG["seed"]).split(y, y))
    fold_of = dict(zip(clean.orig_id.values[np.concatenate([t for _, t in folds])],
                       np.repeat(range(5), [len(t) for _, t in folds])))
    pert_fold = pert.orig_id.map(fold_of).values
    f1 = lambda a, b: f1_score(a, b, labels=LABELS, average="macro", zero_division=0)

    def run_layer(pool, layer):
        Xc = X["clean"][pool][:, layer].astype(np.float32)
        Xp = X["perturbed"][pool][:, layer].astype(np.float32)
        pred_c, pred_ctrl, pred_p = np.empty(len(y), object), np.empty(len(y), object), np.empty(len(pert), object)
        for k, (tr, te) in enumerate(folds):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xc[tr], y[tr])
            pred_c[te] = clf.predict(Xc[te])
            if (pert_fold == k).any():
                pred_p[pert_fold == k] = clf.predict(Xp[pert_fold == k])
            ctrl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xc[tr], y_control[tr])
            pred_ctrl[te] = ctrl.predict(Xc[te])
        rows = [{"data": "clean", "n": len(y), "macro_f1_after": f1(y, pred_c)},
                {"data": "control", "n": len(y), "macro_f1_after": f1(y_control, pred_ctrl)}]
        before = pd.Series(pred_c, index=clean.orig_id.values)
        for name, idx in pert.groupby("perturbation").indices.items():
            p = pert.iloc[idx]
            b = before.loc[p.orig_id].values
            rows.append({"data": name, "n": len(idx),
                         "macro_f1_before": f1(p.orig_label, b), "macro_f1_after": f1(p.label_name, pred_p[idx]),
                         "hate_rate_before": (b == "hate").mean(), "hate_rate_after": (pred_p[idx] == "hate").mean()})
        return [{"model": args.model, "pooling": pool, "layer": layer, **r} for r in rows]

    n_layers = X["clean"]["last"].shape[1]
    jobs = [(pool, layer) for pool in ("last", "mean") for layer in range(n_layers)]
    results = Parallel(n_jobs=len(os.sched_getaffinity(0)))(delayed(run_layer)(*j) for j in jobs)
    res = pd.DataFrame([r for rows in results for r in rows])
    res.to_csv(out, index=False)
    best = res[res.data == "clean"].sort_values("macro_f1_after").iloc[-1]
    print(f"{out.name}: best layer {best.layer} ({best.pooling}), macro-F1 {best.macro_f1_after:.3f}")


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
    ap.add_argument("command", choices=["download", "classify", "guard", "refusal", "judge", "probe"])
    ap.add_argument("--model", help="model name from config.yaml")
    ap.add_argument("--limit", type=int, help="only the first N items (quick test)")
    ap.add_argument("--no-guard", action="store_true", help="judge: keyword check only")
    args = ap.parse_args()
    torch.manual_seed(CFG["seed"])
    globals()[args.command](args)
