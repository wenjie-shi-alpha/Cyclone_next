from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from cyclone_training.config import load_sft_config


class TrainingConfigTests(unittest.TestCase):
    def test_load_sft_config_reads_best_model_and_early_stop_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    model:
                      name_or_path: models/google/gemma-4-E4B-it
                    data:
                      dataset_root: data/training_rebuilt_v2_20260414_guidancefix
                    trainer:
                      output_dir: runs/test_output
                      run_name: test_run
                      per_device_train_batch_size: 1
                      per_device_eval_batch_size: 1
                      gradient_accumulation_steps: 1
                      learning_rate: 0.0001
                      num_train_epochs: 1
                      max_steps: -1
                      warmup_ratio: 0.03
                      weight_decay: 0.01
                      max_grad_norm: 0.3
                      lr_scheduler_type: cosine
                      logging_steps: 5
                      save_steps: 40
                      eval_steps: 40
                      save_total_limit: 4
                      bf16: true
                      fp16: false
                      gradient_checkpointing: true
                      report_to: [none]
                      load_best_model_at_end: true
                      metric_for_best_model: eval_loss
                      greater_is_better: false
                      early_stopping_patience: 3
                      early_stopping_threshold: 0.001
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_sft_config(config_path)

        self.assertTrue(config.trainer.load_best_model_at_end)
        self.assertEqual(config.trainer.metric_for_best_model, "eval_loss")
        self.assertIs(config.trainer.greater_is_better, False)
        self.assertEqual(config.trainer.early_stopping_patience, 3)
        self.assertAlmostEqual(config.trainer.early_stopping_threshold, 0.001)


if __name__ == "__main__":
    unittest.main()
