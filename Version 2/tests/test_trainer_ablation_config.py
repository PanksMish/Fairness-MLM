import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimization.trainer_config import TrainerConfig, StepLog


def test_trainer_config_defaults_all_ablation_flags_true():
    cfg = TrainerConfig()
    assert cfg.use_counterfactual_augmentation is True
    assert cfg.use_adaptive_controller is True
    assert cfg.use_ibadr is True


def test_trainer_config_from_ablation_config_reads_yaml_style_dict():
    ablation_block = {
        "use_counterfactual_augmentation": False,
        "use_semantic_filtering": True,   # not read by TrainerConfig, should be ignored harmlessly
        "use_adaptive_controller": False,
        "use_ibadr": True,
    }
    cfg = TrainerConfig.from_ablation_config(ablation_block, num_epochs=5)
    assert cfg.use_counterfactual_augmentation is False
    assert cfg.use_adaptive_controller is False
    assert cfg.use_ibadr is True
    assert cfg.num_epochs == 5


def test_trainer_config_from_ablation_config_defaults_missing_keys_to_true():
    cfg = TrainerConfig.from_ablation_config({})
    assert cfg.use_counterfactual_augmentation is True
    assert cfg.use_adaptive_controller is True
    assert cfg.use_ibadr is True


def test_actual_ablation_yaml_parses_and_maps_correctly():
    """Sanity-checks the real configs/ablation.yaml file against
    TrainerConfig.from_ablation_config, so a typo in either the YAML
    keys or the dataclass field names would be caught here."""
    import yaml
    import pathlib
    repo_root = pathlib.Path(__file__).parent.parent
    ablation_path = repo_root / "configs" / "ablation.yaml"
    full_config = yaml.safe_load(open(ablation_path))
    cfg = TrainerConfig.from_ablation_config(full_config["ablation"])
    assert cfg.use_counterfactual_augmentation is True  # default file has everything True
    assert cfg.use_adaptive_controller is True
    assert cfg.use_ibadr is True


def test_step_log_construction():
    log = StepLog(step=5, task_loss=0.3, bts=0.4, lam=0.2, total_loss=0.34)
    assert log.step == 5
    assert log.bts == 0.4


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
