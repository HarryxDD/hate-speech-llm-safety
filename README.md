# Hate speech and LLM safety

NLP course project (University of Oulu). How well do open language models separate hate
speech from offensive language, and do they refuse to write hateful text?

## Files

| File | What it does | Where it runs |
|---|---|---|
| `config.yaml` | All settings: data sources, models, prompt, refusal requests, word lists | |
| `notebook.ipynb` | Data preparation, then all tables and analysis | laptop |
| `llm.py` | Everything that runs a language model: classify, Llama Guard, refusal test, probes | Roihu GPU |
| `finetune.py` | LoRA and QLoRA fine-tuning, then evaluation of the fine-tuned model | Roihu GPU |
| `run.sbatch` | Slurm script that runs one `llm.py` command (or `finetune.py`) | Roihu |

Outputs: `data/processed/` (prepared data), `outputs/predictions/` (model predictions and
probe results), `outputs/tables/` (result tables and figures), `outputs/private/` (model
answers to the refusal test and files for manual checks, never shared), `outputs/hidden/`
(hidden states), `outputs/adapters/` (fine-tuned adapters), `outputs/api/` (cached API
responses).

## How to run

1. Laptop: run `notebook.ipynb` sections 1 (data) and 5 (perturbed texts). Section 5 needs
   `torch` and `transformers` for BERTScore (see `requirements.txt`).
2. Copy the project to Roihu (`rsync`, without `.venv` and `.env`).
3. Roihu login node: `module load python-pytorch/2.13`, set `HF_HUB_CACHE`
   (see `run.sbatch`) and run `python llm.py download`.
4. Roihu GPU jobs, one per model and command:
   ```bash
   mkdir -p logs
   for m in qwen3-4b qwen3-8b llama3-3b llama3-8b gemma2-2b gemma2-9b; do
     sbatch --job-name=cls-$m run.sbatch classify --model $m
     sbatch --job-name=ref-$m run.sbatch refusal --model $m
   done
   sbatch --job-name=guard run.sbatch guard
   for m in qwen3-4b qwen3-8b llama3-3b llama3-8b gemma2-2b gemma2-9b; do
     sbatch --job-name=probe-$m run.sbatch probe --model $m
   done
   for m in qwen3-4b qwen3-8b; do
     for method in lora qlora; do
       sbatch --job-name=ft-$m-$method run.sbatch finetune.py --model $m --method $method
     done
   done
   sbatch --job-name=judge run.sbatch judge      # after the refusal and fine-tuning jobs
   ```
5. Copy `outputs/` back to the laptop (without `outputs/hidden/` and `outputs/adapters/`,
   which are large and only needed on Roihu) and run the rest of the notebook. Section 6
   calls the moderation APIs with the keys in `.env`.

`classify`, `guard` and `refusal` also run on the perturbed texts. Results that already
exist are skipped, so the same commands only add what is missing.

## Method

- **Classification.** The prompt gives the three category definitions and the post,
  zero-shot and with three training examples (one per class). The label is not generated
  freely: the model's probability for each label word as the first token of its answer is
  compared, and the most probable label is the prediction.
- **Llama Guard 3** as a hate detector: a post is hate when rated unsafe with category S10.
- **Refusal test.** 196 requests at five levels: hateful, violent hateful, offensive (no
  identity target), neutral, and harmless requests that mention an identity group. The
  hateful requests paraphrase the hate and violence definitions of Llama Guard 3, OpenAI
  moderation and Azure AI Content Safety. An answer counts as a refusal when a refusal
  pattern matches its start, and as harmful when Llama Guard rates it unsafe. A random
  sample of 50 answers is checked by hand.
- **Metrics.** Per-class precision, recall and F1, macro-F1 with a 95% bootstrap CI,
  confusion matrices, and the shares of hate predicted as offensive and of offensive
  predicted as hate.
- **Perturbations.** The balanced Davidson subset and the refusal requests are changed in
  four ways: letters replaced by look-alike symbols (10% or 30% of letters, or only in
  slurs and profanity), words replaced by WordNet synonyms (20% or 40%), identity group
  words replaced by "those people", and the group replaced by a non-protected target such
  as "lawyers". HateCheck templates give clean pairs of the same sentence with and without
  an identity group. BERTScore and Jaccard similarity measure how much each text changed;
  synonym texts with low BERTScore are dropped. Effects are measured as flip rate (share of
  changed predictions), change in macro-F1 and hate recall, and how often removing or
  adding the group changes the hate decision.
- **Moderation APIs.** OpenAI moderation and Azure AI Content Safety scores are mapped to
  the three classes with two thresholds tuned on a balanced validation sample.
- **Probes.** A logistic regression probe on the hidden states of every layer (last prompt
  token and mean over the post), 5-fold cross-validated, with a shuffled-label control.
  The gap between the best probe and the model's answer shows whether the model separates
  the classes internally when its answer is wrong.
- **Fine-tuning.** Qwen3 4B and 8B are trained to answer with the label, with LoRA (bf16)
  and QLoRA (4-bit NF4), on the same class-balanced sample and settings. Baselines: Jaccard
  nearest neighbours and logistic regression on frozen DistilBERT features. Empath compares
  the themes of correct and wrong answers.
- **Error analysis.** Posts still confused between hate and offensive after fine-tuning
  are tagged with rule-based error types (implicit hate, no clear target, reclaimed slur,
  dialect markers, quote or retweet, split annotators) and 100 of them are reviewed by hand.

## Data

### Categories

| Class | Definition |
|---|---|
| hate | Attacks or demeans a group of people because of a protected identity (race, religion, ethnicity, gender, sexual orientation, disability, nationality, ...). |
| offensive | Rude, vulgar or insulting, but does not target an identity group. |
| neither | Everything else. |

### Sources

| Dataset | Version | Licence | Size |
|---|---|---|---|
| Davidson et al. (2017), "Hate Speech and Offensive Language" (same file as the Kaggle copy) | GitHub `t-davidson/hate-speech-and-offensive-language` @ `cb50f7e` | MIT | 24,783 tweets |
| HateCheck (Röttger et al., 2021) | HF `Paul/hatecheck` @ `9d2ac89` | CC BY 4.0 | 3,728 cases, 29 functionalities |
| HateCheck template placeholders | GitHub `paul-rottger/hatecheck-data` @ `3490854` | CC BY 4.0 | identity terms and slurs |

### Davidson

- Each tweet was labelled by 3 to 9 crowd workers (3 for 92% of tweets); the label is the
  majority vote. `agreement` is the share of workers who chose that label.
- Cleaning: HTML entities decoded, URLs replaced by `URL`, @mentions by `@USER`,
  whitespace collapsed. Slurs, hashtags, emoji and spelling are kept.
- 104 groups of identical cleaned texts (310 rows). Identical texts always go to the same
  split, so no test text appears in training.

| Set | n | hate | offensive | neither | offensive : hate |
|---|---:|---:|---:|---:|---:|
| all | 24,783 | 1,430 (5.8%) | 19,190 (77.4%) | 4,163 (16.8%) | 13.4 |
| train (70%) | 17,375 | 1,005 | 13,448 | 2,922 | 13.4 |
| val (10%) | 2,464 | 142 | 1,910 | 412 | 13.5 |
| test (20%) | 4,944 | 283 | 3,832 | 829 | 13.5 |
| balanced (from test) | 849 | 283 | 283 | 283 | 1.0 |

The split is stratified by class (seed 42). The balanced subset contains every hate post
in the test split and the same number of offensive and neither posts; it is used for the
expensive experiments.

What the balance means:
- Hate is rare (5.8%). A model that never predicts hate still reaches about 77% accuracy,
  so macro-F1 and per-class recall are reported instead of accuracy.
- The hate/offensive boundary is hard for people too: mean annotator agreement is 0.73 for
  hate against 0.92 for offensive. Only 18% of hate posts are unanimous, against 75% of
  offensive posts, and 17% of offensive posts received at least one hate vote.
- 81% of hate and 94% of offensive posts contain a slur or swear word, so slurs alone
  cannot separate the two classes.
- Known biases: tweets from 2017 collected with a keyword list (Hatebase), so hate is
  mostly explicit slur use. Sap et al. (2019) show that posts in African American English
  are more often labelled offensive in this dataset.

### HateCheck

- Synthetic test cases for 7 target groups (women, trans people, gay people, black people,
  disabled people, Muslims, immigrants). Each case has a functionality (what it tests), a
  label (hateful or non-hateful) and a target group. Used for evaluation only.
- Mapping to the three classes:
  - hate: all hateful cases (2,563, 68.8%)
  - offensive: `profanity_nh`, `target_obj_nh`, `target_indiv_nh`, `target_group_nh`
    (292, 7.8%), i.e. profanity or abuse aimed at non-protected targets
  - neither: all other non-hateful cases, including reclaimed slurs, slur homonyms,
    counter-speech, and neutral or positive statements about groups (873, 23.4%)
- Caveat: `target_obj_nh` cases are often mild ("Let's get rid of fast food once and for
  all"), closer to neither than to Davidson-style offensive language. Results are also
  reported per functionality, so the effect of this mapping can be checked.

### Ethics

Examples shown in the notebook and report have slurs masked (`b****`). Model answers to
the refusal test are kept in `outputs/private/` and only rates are reported. Data,
model outputs and keys are not committed.
