"""Comparer la préparation avec la fonction originale du réentraînement."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import resample_poly

from pasl_inference.experiment import check_checkpoint_metadata, resolve_experiment
from pasl_inference.preprocessing import RAW_WINDOW_SAMPLES, make_batch, window_layout


class PreprocessingTests(unittest.TestCase):
    def test_matches_original_builder(self):
        root = Path(__file__).resolve().parents[2]
        source = root / "retraining_wav2sleep/code/build_transformer_dataset.py"

        # Charger uniquement la fonction originale, sans lancer le builder.
        tree = ast.parse(source.read_text())
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "extract_respiration"
        )
        scope = {
            "np": np,
            "resample_poly": resample_poly,
            "WINDOW_EPOCHS": 512,
            "EXPECTED_RESP_SAMPLES": 153600,
            "RESP_SEGMENTS": 64,
            "RESP_SAMPLES_PER_SEGMENT": 2400,
        }
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(source), "exec"), scope)

        signal = np.random.default_rng(4).normal(
            loc=10, scale=3, size=(1024, 3000)
        ).astype(np.float32)

        original_source = SimpleNamespace(
            data=signal,
            subjects={"example": {"pos": (0, 1024)}},
        )
        stats = {"mean": 9.2, "std": 2.8}

        actual = make_batch(
            signal.reshape(-1), [(0, 512), (512, 512)], stats
        )

        for index in range(2):
            expected = scope["extract_respiration"](
                {"subject_id": "example", "window_index": index},
                original_source,
                stats,
            )
            np.testing.assert_array_equal(actual[index], expected)

    def test_tail_is_not_padded(self):
        windows, remainder = window_layout(RAW_WINDOW_SAMPLES + 10)
        self.assertEqual(windows, [(0, 512)])
        self.assertEqual(remainder, 10)

        with self.assertRaises(ValueError):
            window_layout(RAW_WINDOW_SAMPLES + 10, tail="error")

        with self.assertRaises(ValueError):
            window_layout(RAW_WINDOW_SAMPLES - 1)

    def test_each_seed_selects_its_own_pair(self):
        for seed in (0, 1, 42):
            experiment = resolve_experiment(Path("weights"), seed)
            self.assertIn(f"seed_{seed}", experiment["transformer"].parts)
            self.assertIn(f"seed_{seed}", experiment["vqgan"].parts)

    def test_wrong_checkpoint_seed_is_rejected(self):
        experiment = resolve_experiment(Path("weights"), 0)
        checkpoint = {"training_config": {"seed": 1}, "model_config": {}}

        with self.assertRaises(ValueError):
            check_checkpoint_metadata(checkpoint, experiment)


if __name__ == "__main__":
    unittest.main()
