from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from os_segformer.dataset_spec import load_dataset_spec
from os_segformer.models import OSSegFormer


class DummyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(3, 32, 1),
                torch.nn.Conv2d(3, 64, 1),
                torch.nn.Conv2d(3, 160, 1),
                torch.nn.Conv2d(3, 256, 1),
            ]
        )

    def forward(self, *, pixel_values, output_hidden_states, return_dict):
        del output_hidden_states, return_dict
        scales = (4, 8, 16, 32)
        states = tuple(
            projection(F.avg_pool2d(pixel_values, scale))
            for projection, scale in zip(self.projections, scales)
        )
        return SimpleNamespace(hidden_states=states)


class DummyDecodeHead(torch.nn.Module):
    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.classifier = torch.nn.Conv2d(32, num_labels, 1)

    def forward(self, hidden_states):
        return self.classifier(hidden_states[0])


class DummySegFormer(torch.nn.Module):
    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.segformer = DummyEncoder()
        self.decode_head = DummyDecodeHead(num_labels)


class ModelTests(unittest.TestCase):
    def _run_profile(self, profile: str, expected: tuple[int, int, int]) -> None:
        spec = load_dataset_spec(profile)
        assert spec.factorization is not None
        semantic_classes, object_classes, state_channels = expected
        model = OSSegFormer(
            DummySegFormer(semantic_classes),
            hidden_sizes=(32, 64, 160, 256),
            factorization_spec=spec.factorization,
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(output.logits.shape), (2, semantic_classes, 16, 16))
        self.assertEqual(
            tuple(output.auxiliary["object"].shape),
            (2, object_classes, 16, 16),
        )
        self.assertEqual(
            tuple(output.auxiliary["state"].shape),
            (2, state_channels, 16, 16),
        )

    def test_floodnet_model_initialization_and_forward(self) -> None:
        self._run_profile("configs/datasets/floodnet.yaml", (10, 8, 4))

    def test_rescuenet_model_initialization_and_forward(self) -> None:
        self._run_profile("configs/datasets/rescuenet.yaml", (11, 7, 6))


if __name__ == "__main__":
    unittest.main()
