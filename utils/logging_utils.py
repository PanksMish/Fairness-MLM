"""
logging_utils.py
================
Experiment logging utilities for ADAPT-BTS.

Provides:
  - Structured JSON logging for all training metrics
  - CSV export for downstream analysis
  - Console formatting with color-coded severity
  - Checkpoint and result tracking
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


def setup_logging(
    output_dir: str = "outputs/",
    log_level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Configure root logger with both console and file handlers.

    Parameters
    ----------
    output_dir : str
        Directory to write log file.
    log_level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    log_to_file : bool
        If True, also write logs to a timestamped file.

    Returns
    -------
    logger : logging.Logger
    """
    os.makedirs(output_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)7s | %(name)20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"adapt_bts_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


class ExperimentLogger:
    """
    Structured experiment logger that records all metrics to JSON
    and exports to CSV for analysis.

    Usage:
        logger = ExperimentLogger("outputs/run1/")
        logger.log_epoch(epoch=1, train={"loss": 0.42}, val={"f1": 0.85})
        logger.save()
    """

    def __init__(self, output_dir: str, experiment_name: str = "adapt_bts"):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        os.makedirs(output_dir, exist_ok=True)

        self.start_time = time.time()
        self.epoch_records: List[Dict] = []
        self.metadata: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
        }

    def log_config(self, config: Dict):
        """Store experiment configuration."""
        self.metadata["config"] = config
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        extra: Optional[Dict] = None,
    ):
        """
        Record metrics for a single training epoch.

        Parameters
        ----------
        epoch : int
        train_metrics : dict of training metrics (loss, bts, lambda, etc.)
        val_metrics : dict of validation metrics (f1, bts, ccr, dpg, etc.)
        extra : optional additional metadata
        """
        record = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self.start_time, 1),
            "train": train_metrics,
            "val": val_metrics,
        }
        if extra:
            record["extra"] = extra
        self.epoch_records.append(record)

    def log_per_language_results(self, results_df):
        """Save per-language evaluation results."""
        import pandas as pd
        if isinstance(results_df, pd.DataFrame):
            path = os.path.join(self.output_dir, "per_language_results.csv")
            results_df.to_csv(path, index=False)

    def log_comparison_table(self, comparison_df):
        """Save model comparison table (Table 3)."""
        import pandas as pd
        if isinstance(comparison_df, pd.DataFrame):
            path = os.path.join(self.output_dir, "comparison_table.csv")
            comparison_df.to_csv(path, index=False)

    def save(self):
        """
        Write all recorded metrics to disk:
          - metrics.json: full structured log
          - metrics.csv: flattened for analysis
        """
        # JSON output
        output = {
            **self.metadata,
            "end_time": datetime.now().isoformat(),
            "total_elapsed_s": round(time.time() - self.start_time, 1),
            "epochs": self.epoch_records,
        }
        json_path = os.path.join(self.output_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

        # CSV output (flattened)
        try:
            import pandas as pd
            rows = []
            for record in self.epoch_records:
                row = {"epoch": record["epoch"], "elapsed_s": record["elapsed_s"]}
                for k, v in record.get("train", {}).items():
                    row[f"train_{k}"] = v
                for k, v in record.get("val", {}).items():
                    row[f"val_{k}"] = v
                rows.append(row)
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(self.output_dir, "metrics.csv"), index=False)
        except ImportError:
            pass

    def print_summary(self):
        """Print a summary of the experiment to console."""
        if not self.epoch_records:
            print("No epochs recorded.")
            return

        best_f1_epoch = max(
            self.epoch_records,
            key=lambda r: r.get("val", {}).get("macro_f1", 0),
        )
        best_bts_epoch = min(
            self.epoch_records,
            key=lambda r: r.get("val", {}).get("bts", 1),
        )

        print("\n" + "=" * 60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Total epochs: {len(self.epoch_records)}")
        print(f"Elapsed: {time.time() - self.start_time:.0f}s")
        print(f"Best val F1: {best_f1_epoch['val'].get('macro_f1', 0):.4f} "
              f"(epoch {best_f1_epoch['epoch']})")
        print(f"Best val BTS: {best_bts_epoch['val'].get('bts', 1):.4f} "
              f"(epoch {best_bts_epoch['epoch']})")
        print("=" * 60)
