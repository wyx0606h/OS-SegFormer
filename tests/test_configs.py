from __future__ import annotations

import unittest

from os_segformer.config import load_yaml_config
from os_segformer.dataset_spec import resolve_dataset_spec


class ConfigTests(unittest.TestCase):
    def test_release_configs_parse(self) -> None:
        expected = {
            "configs/floodnet/os_segformer.yaml": (10, True),
            "configs/floodnet/segformer_b0.yaml": (10, False),
            "configs/rescuenet/os_segformer.yaml": (11, True),
            "configs/rescuenet/segformer_b0.yaml": (11, False),
        }
        for path, (classes, factorized) in expected.items():
            with self.subTest(path=path):
                config = load_yaml_config(path)
                spec = resolve_dataset_spec(config)
                self.assertEqual(spec.num_classes, classes)
                enabled = bool(
                    config["model"].get("state_factorization", {}).get(
                        "enabled", False
                    )
                )
                self.assertEqual(enabled, factorized)


if __name__ == "__main__":
    unittest.main()
