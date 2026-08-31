from __future__ import annotations

import unittest

import torch

from os_segformer.dataset_spec import load_dataset_spec
from os_segformer.state_factorization import (
    compose_hierarchical_probabilities,
    fuse_semantic_and_hierarchical_logits,
)


class FactorizationTests(unittest.TestCase):
    def test_floodnet_composition_is_normalized(self) -> None:
        spec = load_dataset_spec("configs/datasets/floodnet.yaml").factorization
        assert spec is not None
        object_logits = torch.randn(2, 8, 5, 7)
        state_logits = torch.randn(2, 4, 5, 7)
        posterior = compose_hierarchical_probabilities(
            object_logits, state_logits, factorization_spec=spec
        )
        self.assertEqual(tuple(posterior.shape), (2, 10, 5, 7))
        torch.testing.assert_close(
            posterior.sum(dim=1), torch.ones(2, 5, 7), atol=1e-6, rtol=1e-6
        )

    def test_rescuenet_uses_four_and_two_state_groups(self) -> None:
        spec = load_dataset_spec("configs/datasets/rescuenet.yaml").factorization
        assert spec is not None
        self.assertEqual([len(group.semantic_class_ids) for group in spec.state_groups], [4, 2])
        object_logits = torch.randn(1, 7, 4, 6)
        state_logits = torch.randn(1, 6, 4, 6)
        posterior = compose_hierarchical_probabilities(
            object_logits, state_logits, factorization_spec=spec
        )
        torch.testing.assert_close(
            posterior.sum(dim=1), torch.ones(1, 4, 6), atol=1e-6, rtol=1e-6
        )

    def test_fusion_preserves_group_mass_and_non_stateful_classes(self) -> None:
        spec = load_dataset_spec("configs/datasets/floodnet.yaml").factorization
        assert spec is not None
        flat_logits = torch.randn(1, 10, 3, 4)
        factorized = torch.softmax(torch.randn(1, 10, 3, 4), dim=1)
        fused_logits = fuse_semantic_and_hierarchical_logits(
            flat_logits,
            factorized,
            fusion_weight=0.25,
            factorization_spec=spec,
        )
        flat = torch.softmax(flat_logits, dim=1)
        fused = torch.softmax(fused_logits, dim=1)
        for group in spec.state_groups:
            ids = list(group.semantic_class_ids)
            torch.testing.assert_close(
                fused[:, ids].sum(dim=1),
                flat[:, ids].sum(dim=1),
                atol=1e-6,
                rtol=1e-6,
            )
        grouped = {
            semantic_id
            for group in spec.state_groups
            for semantic_id in group.semantic_class_ids
        }
        non_stateful = [
            semantic_id
            for semantic_id in range(len(spec.semantic_to_object))
            if semantic_id not in grouped
        ]
        torch.testing.assert_close(
            fused[:, non_stateful],
            flat[:, non_stateful],
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
