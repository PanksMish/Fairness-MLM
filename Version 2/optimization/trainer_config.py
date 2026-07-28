"""
Trainer configuration, extracted from optimization/trainer.py into its
own zero-torch-dependency module -- the same split pattern used
elsewhere in this repo (fairness/bias_transfer_score.py's pure-NumPy
core vs. fairness/bts_torch.py's differentiable wrapper). TrainerConfig
is a plain dataclass with no model/tensor logic, so there's no reason
for it to require torch to be importable/testable.

optimization/trainer.py imports TrainerConfig and StepLog from here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainerConfig:
    num_epochs: int = 3
    learning_rate: float = 5e-5           # Sec 5.2
    batch_size: int = 128                 # Sec 5.2 (per-GPU / effective, adjust for your hardware)
    tau: float = 0.40                     # fairness target, Eq. 11
    eta_lambda: float = 0.05              # Eq. 13
    lambda_init: float = 0.20
    lambda_min: float = 0.05
    lambda_max: float = 1.00
    ibadr_refresh_interval: int = 500     # K, Algorithm 3
    ibadr_selection_ratio: float = 0.10   # rho, Algorithm 3
    use_amp: bool = True                  # Sec 5.2: BF16 precision
    log_every: int = 50
    task: str = "sentiment"               # "sentiment" | "ner"

    # Ablation toggles, matching configs/ablation.yaml's vocabulary
    # (Fig. 11b/18/19's "-CDA"/"-Filtering"/"-FAPC"/"-IBADR" variants).
    # `use_semantic_filtering` is NOT read here -- it's a
    # counterfactual-generation-time flag (whether Eq. 9's acceptance
    # gate is applied when building D_aug), not a training-loop
    # decision, so it belongs to fairness/counterfactual_generation.py's
    # CounterfactualEngineConfig.gamma (set gamma=-inf to accept
    # everything, disabling filtering) rather than to this class.
    use_counterfactual_augmentation: bool = True   # "-CDA" when False: no BTS term, task loss only
    use_adaptive_controller: bool = True           # "-FAPC" when False: lambda fixed at lambda_init, no Eq. 13 updates
    use_ibadr: bool = True                         # "-IBADR" when False: Algorithm 3 refresh never triggers

    @classmethod
    def from_ablation_config(cls, ablation: dict, **kwargs) -> "TrainerConfig":
        """Convenience constructor reading configs/ablation.yaml's
        `ablation:` block directly, e.g.:

            cfg = TrainerConfig.from_ablation_config(
                yaml.safe_load(open("configs/ablation.yaml"))["ablation"],
                num_epochs=3, learning_rate=5e-5,
            )
        """
        return cls(
            use_counterfactual_augmentation=ablation.get("use_counterfactual_augmentation", True),
            use_adaptive_controller=ablation.get("use_adaptive_controller", True),
            use_ibadr=ablation.get("use_ibadr", True),
            **kwargs,
        )


@dataclass
class StepLog:
    step: int
    task_loss: float
    bts: float
    lam: float
    total_loss: float
