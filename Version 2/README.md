# ADAPT-BTS — Reference Implementation

A real, incrementally-built implementation of the ADAPT-BTS framework
from the manuscript *"ADAPT-BTS: Closed-Loop Distributional Fairness
Optimization for Multilingual Foundation Models."* Every module listed
below either (a) has been executed and unit-tested in this sandbox, or
(b) is real, syntax-checked code that needs torch/GPU/network I don't
have here to actually run. Section headers say which is which — nothing
is asserted to work without saying how I know.

**241/241 tests passing.** Run them yourself:
```bash
python3 tests/_minimal_runner.py    # zero extra dependencies needed
# or, with network access:
pip install pytest && pytest tests/ -v
```

## ⚠️ Sentiment: 101 languages configured, but only 6 have real gold labels — read this

`configs/default_config.yaml` now has a genuine 101 languages (18 HR +
37 MR + 46 LR, matching the paper's Table 2 counts exactly, no
duplicates, every code verified present in CC100). But only 6 of them —
en, de, es, fr, ja, zh, the actual real coverage of Amazon Reviews
Multilingual — have real human-annotated sentiment labels anywhere in
this pipeline. TweetEval adds no new languages (English only).

**`datasets/build_full_sentiment_dataset.py`** is the combiner that
makes "101 languages" achievable at all: it builds gold-labeled data for
those 6 languages via `datasets/download_sentiment.py`, and **weak,
lexicon-heuristic labels** for the other 95 via
`datasets/build_weak_sentiment.py` (real CC100 text + NRC Emotion
Lexicon polarity scoring — a real, documented technique, not
fabrication, but not gold annotation either). Every record — gold or
weak — carries `label_source` (`"gold"` or `"weak_lexicon_nrc"`) and
`is_gold_label` (`true`/`false`), and the combiner writes a
`coverage_report.csv` showing exactly which of the 101 languages got
which treatment, so this composition is never invisible.

**What this does and doesn't fix:** it gets you real training signal in
101 languages instead of 6. It does not give you 101 languages of gold
labels — 95 of them are heuristic, with all the caveats
`datasets/weak_labeling.py`'s docstring lays out (misses negation,
sarcasm, mixed sentiment; machine-translated lexicon, not
human-verified per language). A Macro-F1 computed on this combined
dataset is not comparable to the manuscript's Table 5 unless you
report the gold/weak split alongside it.

## The one thing that matters most

**No code in this repository, at any point, was written to reproduce
the manuscript's specific reported numbers** (85.6 Macro-F1, 0.36 BTS,
12.2% BTS reduction, p<0.001, Cohen's d=0.84, etc.). Every metric
function computes its result live from whatever data you feed it.
Running this on real data will produce whatever it produces — that's
the only way a reproduction claim means anything.

I (Claude) built this in a sandbox with no internet access, no
torch/transformers, and no GPU. I cannot download CC100/WikiAnn/mT5
weights, run mT5-base training, or generate the manuscript's actual
figures/tables from real experiments. What I *could* do — and did — is
build every piece as real, correct, individually-verifiable code, test
everything that's testable without a GPU, and be explicit about the
rest.

---

## What's real and tested right now (no GPU/network needed)

### Fairness math (Sec 3–4, Algorithms 1–3)
| Module | Covers |
|---|---|
| `fairness/bias_transfer_score.py` | BTS metric — Eq. 4, 15, 16, 21 |
| `fairness/fairness_controller.py` | Adaptive λ controller — Eq. 12–14, 17–18, Algorithm 2, Appendix C.1 |
| `fairness/ibadr.py` | Selective data refresh — Algorithm 3 |
| `fairness/semantic_validation.py` | Semantic-syntactic acceptance gate — Eq. 3, 8, 9 |
| `fairness/morphology.py` | Structural sanity check + illustrative ES/DE agreement fixer |
| `fairness/counterfactual_generation.py` | Full Algorithm 1; HR-tier lexicon substitution is real and deterministic |
| `fairness/ner_counterfactual_generation.py` | Per-token substitution for WikiAnn, tag-alignment-preserving by construction |
| `fairness/augmentation.py` | D_aug construction — Eq. 10 |
| `fairness/embedding_backends.py` | Real FastText `.vec`-format loader + cosine nearest-neighbor for the MR tier |

### Evaluation math (Table 4–7, Sec 5.2)
| Module | Covers |
|---|---|
| `evaluation/fairness_metrics.py` | CCR, DPG, instance-weighted aggregation — Eq. 16, 19, 20 |
| `evaluation/statistical_tests.py` | Paired t-test, Wilcoxon, bootstrap CI, Cohen's d, ANOVA, Kruskal-Wallis, correlation, OLS |
| `evaluation/metrics.py` | Macro-F1 + a dependency-free `seqeval`-compatible span-F1 |
| `evaluation/leakage.py` | Representation-leakage linear probe (real `sklearn.LogisticRegression`) |
| `evaluation/report.py` | Table 5/6/7/8-shaped `pandas` DataFrames + CSV/markdown export |

### Data pipeline logic (Sec 5.1)
| Module | Covers |
|---|---|
| `datasets/dataset_utils.py` | Clean → dedup → normalize → split → JSONL I/O |
| `datasets/language_filter.py` | Language-verification logic + resource-tier categorization |
| `datasets/tokenizer.py` | NER subword-label alignment + dynamic padding |
| `datasets/vocab.py` | String↔int label vocabularies for MFC/MADL |
| `datasets/weak_labeling.py` | Lexicon-based weak sentiment labeling logic — real scoring/filtering, tested against synthetic lexicons and known-polarity sentences. **See the warning section above before using.** |
| `scripts/config_utils.py` | YAML config merging + CLI overrides |

### Visualization & appendix analysis (Fig. 4–19, Appendix A)
| Module | Covers |
|---|---|
| `visualization/plots.py` | Bar/line/scatter/KDE/residual charts, styled after Fig. 4–19 — **actually executed here**, real PNGs verified |
| `appendix/regression_analysis.py` | BTS-vs-DPG-style regression + residual diagnostics, wired to `evaluation/statistical_tests.py` |

### Training-loop logic
| Module | Covers |
|---|---|
| `optimization/trainer_config.py` | `TrainerConfig`, including the `-CDA`/`-FAPC`/`-IBADR` ablation toggles from `configs/ablation.yaml` |
| `baselines/losses_core.py` | Pure-NumPy core of MFC's exact contrastive loss (Lin et al. 2023, Eq. 2–5) — **hand-computed test included** |

Every one of these has been run in this sandbox against real or
synthetic inputs. The visualization functions in particular produce
actual PNG files I verified (magic bytes, non-trivial size) — not just
"the code parses."

---

## What's real but *not* executable here (needs torch/GPU/network)

### Model & training
- `model/encoder.py`, `classifier.py`, `heads.py`, `mt5.py`, `xlmr.py` — mT5/XLM-R backbone + task heads (Sec 3.2, 5.1–5.2)
- `datasets/dataloaders.py` — `SentimentDataset`, `WikiAnnDataset`, and their paired (`PairedSentimentDataset`, `PairedWikiAnnDataset`) counterfactual-pair variants
- `optimization/trainer.py` — full Algorithm 2 training loop, now branching on the `-CDA`/`-FAPC`/`-IBADR` ablation toggles
- `optimization/losses.py`, `optimizer.py` — task losses, AdamW + warmup/decay
- `evaluation/evaluator.py` — `evaluate_sentiment`/`evaluate_ner`, both computing per-language **and** instance-weighted global metrics (BTS/CCR now computed per-sample for NER too, not just globally)

### Baselines (Table 3) — fidelity graded individually, not uniform
| Baseline | Fidelity |
|---|---|
| `baselines/mt5_ft.py` | Exact (trivially — same model, λ=0) |
| `baselines/mfc.py` | **Exact, verified against the source paper's actual equations** (arXiv:2303.15697, fetched and checked before implementing) |
| `baselines/csd.py`, `madl.py`, `grad_unlearn.py` | Real, standard implementations of the *general mechanism* their papers' titles/abstracts describe — I couldn't retrieve their exact equations this session, and say so |
| `baselines/magnet.py` | **Explicitly not real MAGNET** — I checked the actual paper (arXiv:2407.08818): real MAGNET is a byte-level tokenization architecture, incompatible with this repo's subword-tokenizer models. What's here is a clearly-labeled loss-reweighting proxy with a docstring warning not to report it as MAGNET |
| `baselines/train_all_baselines.py` | Orchestrator; all six baselines fully wired including MFC/MADL's integer id encoding via `datasets/vocab.py` |

### Real cross-lingual backends
- `fairness/translation_backends.py` — real HF `pipeline("translation", ...)` wrapper (OPUS-MT and NLLB variants) for the LR-tier pivot-translation path
- `fairness/embedding_backends.py`'s loader is tested; actually pointing it at a *real aligned* cross-lingual `.vec` file (e.g. from fasttext.cc's aligned-vectors release) is a network-dependent step left to you

### Weak-supervision data pipeline (network-dependent, not executed here)
- `datasets/download_cc100.py` — targets `statmt/cc100` on HF (checked live: the legacy `cc100` loading-script config is broken, this Parquet mirror isn't)
- `datasets/nrc_lexicon.py` — downloads + parses the NRC Emotion Lexicon from its official URL (respects the source's "no redistribution" license term by fetching fresh rather than bundling); the multilingual file's exact column layout is unverified since I never downloaded it, so parsing is defensively done by column name with clear errors on mismatch
- `datasets/build_weak_sentiment.py` — orchestrates the above into weak-labeled sentiment data, output structurally isolated from real data (see warning section up top)

### CLI entrypoints
- `scripts/train.py`, `evaluate.py` — dispatch on `task: sentiment` vs `task: ner`
- `scripts/reproduce_tables.py`, `reproduce_figures.py`, `reproduce_appendix.py` — read real `evaluate.py` JSON output and format it into tables/figures; **do no metric computation of their own** and raise clear errors if the JSON files they need don't exist yet
- `scripts/run_all.sh` — full pipeline orchestration, explicitly marked as never having been executed

None of the above has run end-to-end anywhere. Each piece is
syntax-checked (`ast.parse`, all passing) and, where the logic doesn't
need torch, unit-tested. The first real run on your infrastructure will
surface integration bugs — that's normal and expected for a codebase
this size that's never been executed as a whole.

---

## Known gaps and simplifications (stated plainly, not buried)

- **Demographic dictionaries**: only a small illustrative English gender dictionary exists (`fairness/demographic_dictionaries.py`). Real experiments need curated dictionaries per HR language and ideally more attribute dimensions than binary gender.
- **MR/LR-tier counterfactual generation**: the dispatch logic and real backend *wrappers* exist, but nothing has actually been run against a real embedding space or MT model — that requires network access this sandbox doesn't have.
- **The 101-language list**: the manuscript never publishes it in full (only per-family and per-resource-tier counts). `configs/default_config.yaml`'s list is a documented starting point, not a reproduction of their exact set.
- **`amazon_reviews_multi`**: removed from Hugging Face in 2024 (confirmed via live web search, not stale training data). `datasets/download_sentiment.py` falls back to a mirror and logs which source it used.
- **Ablation `-Filtering`**: not a `TrainerConfig` toggle — it belongs to `CounterfactualEngineConfig.gamma` at data-generation time, not the training loop; documented in `trainer_config.py`.
- **CSD/MADL/Grad-Unl baselines**: general-mechanism reimplementations, not verified against their source papers' exact equations (I checked what I could find and said so where I couldn't).

## Requirements

```
numpy, scipy, scikit-learn, pandas, matplotlib, pyyaml, pytest   # everything in "real and tested" above
datasets, fasttext-langdetect                                     # dataset downloaders
torch, transformers, sentencepiece, accelerate, sentence-transformers  # model/trainer/evaluator/counterfactual encoder
```
See `requirements.txt` for the full pinned list with per-line notes on
which module needs what.

## If you want to actually run this

```bash
pip install -r requirements.txt

# 1. Build data (English, HR-tier lexicon path — the only fully-wired language)
python datasets/build_sentiment.py --languages en
python datasets/build_counterfactual_pairs.py \
    --input data/processed/sentiment/train.jsonl \
    --output data/processed/sentiment/train_pairs.jsonl --language en

# 2. Train
python scripts/train.py --model-config configs/mt5.yaml --task-config configs/sentiment.yaml

# 3. Evaluate
python scripts/evaluate.py --model-config configs/mt5.yaml --task-config configs/sentiment.yaml \
    --checkpoint checkpoints/sentiment/final_model.pt

# 4. Tables & figures, once you have JSON reports from step 3 for multiple methods
python scripts/reproduce_tables.py --reports "ADAPT-BTS=outputs/eval.json"
python scripts/reproduce_figures.py --reports "ADAPT-BTS=outputs/eval.json"
```
Expect to debug things. Report back what breaks and I'll fix it.

## What would make this stronger next

Roughly in order of value:
1. **Run it** — the only way to know if any of the untested-here code actually works.
2. Curate real demographic dictionaries for more HR languages / attribute dimensions.
3. Point `fairness/embedding_backends.py` and `translation_backends.py` at real aligned vectors / MT models and verify the MR/LR tiers produce sensible counterfactuals.
4. Cross-check `csd.py`/`madl.py`/`grad_unlearn.py` against their source papers' actual equations if you can access them.
5. Extend `evaluate_ner` with DPG (currently sentiment-only).
