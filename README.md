# ADAPT-BTS: Adaptive Dual-Phase Framework for Mitigating Demographic and Dialectal Bias in Multilingual Language Models

## Overview

This repository provides the full implementation of **ADAPT-BTS** (Adaptive Distributional Alignment via Probabilistic Transfer using Bias Transfer Score), an adaptive fairness framework for multilingual transformer models. The framework integrates:

1. **Semantically Validated Counterfactual Data Augmentation (CDA)** — generates demographic variants using multilingual masked language modeling with cosine similarity filtering and morphological agreement constraints.
2. **Iterative Bias-Aware Data Refresh (IBADR)** — dynamically regenerates high-divergence counterfactual samples during training.
3. **Fairness-Aware Proportional Control (FAPC)** — feedback-controlled optimization that adapts the fairness regularization weight λ based on real-time BTS measurements.

The **Bias Transfer Score (BTS)** is the core fairness metric, defined as the expected total variation divergence between predictive distributions of counterfactual pairs.

---

## Project Structure

```
adapt_bts/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   └── default_config.yaml          # All hyperparameters
├── data/
│   ├── dataset_loader.py            # CC100 + XTREME data loaders
│   ├── demographic_extractor.py     # Demographic attribute detection
│   ├── counterfactual_generator.py  # CDA generation + validation
│   └── data_refresh.py              # IBADR mechanism
├── models/
│   ├── multilingual_model.py        # mT5/XLM-R backbone wrapper
│   └── bias_transfer_score.py       # BTS metric implementation
├── training/
│   ├── trainer.py                   # Main ADAPT-BTS trainer
│   ├── fairness_controller.py       # Proportional feedback controller
│   └── objectives.py                # Loss functions
├── evaluation/
│   ├── evaluator.py                 # Full evaluation pipeline
│   ├── metrics.py                   # CCR, DPG, EOD, leakage
│   └── statistical_tests.py        # ANOVA, Mann-Whitney, Cohen's d
├── utils/
│   ├── morphological_checker.py    # MAC for agreement validation
│   ├── language_utils.py           # Language family / resource tier
│   └── logging_utils.py            # Experiment logging
└── scripts/
    ├── run_training.py              # Entry point for training
    ├── run_evaluation.py            # Entry point for evaluation
    └── run_ablation.py              # Ablation study runner
```

---

## Installation

```bash
git clone https://github.com/yourname/adapt-bts.git
cd adapt-bts
pip install -r requirements.txt
```

---

## Quick Start

### Training
```bash
python scripts/run_training.py \
    --config configs/default_config.yaml \
    --task sentiment \
    --languages en de fr es hi tr \
    --output_dir outputs/adapt_bts_run1
```

### Evaluation
```bash
python scripts/run_evaluation.py \
    --model_dir outputs/adapt_bts_run1 \
    --task sentiment \
    --output_dir results/
```

### Ablation Study
```bash
python scripts/run_ablation.py \
    --config configs/default_config.yaml \
    --output_dir results/ablation/
```

---

## Key Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| Macro-F1 | Task predictive performance | ↑ Higher is better |
| BTS | Bias Transfer Score (distributional divergence) | ↓ Lower is better |
| CCR | Counterfactual Consistency Rate | ↑ Higher is better |
| DPG | Demographic Parity Gap | ↓ Lower is better |
| EOD | Equalized Odds Difference | ↓ Lower is better |
| Leakage | Linear probe accuracy on frozen encoder | ↓ Lower is better |

---

## Citation

If you use this code, please cite the ADAPT-BTS paper.

---

## License

MIT License
