# OS-SegFormer

Official implementation of:

**Object-State Factorization for Post-Flood Aerial Semantic Segmentation**

[Paper link will be added after review]

OS-SegFormer preserves a flat semantic prediction path while factorizing
stateful semantic categories into object identity and object-conditioned state
prediction, followed by group-mass-preserving posterior fusion.

This submission-stage release is centered on the FloodNet experiments. Minimal
RescueNet support is retained only for the paper's secondary cross-dataset
applicability evaluation.

## Overview

OS-SegFormer uses one SegFormer-B0 encoder for two prediction paths. The
standard decoder produces the directly supervised flat semantic posterior. A
separate multi-scale decoder consumes all four encoder stages, predicts object
identity, and uses selected object posteriors to additively condition grouped
state prediction. The factorized posterior is
`P(object|x) P(state|object,x)`.

Fusion preserves the flat posterior mass of every stateful object group and
changes only its within-group state allocation. Non-stateful semantic classes
remain equal to the flat posterior. The released objective contains semantic
CE+Dice plus object and grouped-state auxiliary losses; it contains no
additional loss between the flat and factorized posteriors.

## Repository Structure

```text
configs/          primary FloodNet recipes and secondary RescueNet adaptation
os_segformer/     model, factorization, losses, data, inference, and metrics
splits/           identifier/path manifests only; no images or masks
tools/            supported dataset preparation helper
train.py          unified training entry point
evaluate.py       checkpoint evaluation and optional prediction export
```

## Installation

Create a clean Python environment, install a PyTorch build compatible with your
CUDA driver, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The frozen FloodNet run used Python 3.10.20, PyTorch 2.1.2 with CUDA 12.1, and
an NVIDIA RTX 4090-class GPU. Package versions in `requirements.txt` are kept
minimal because the appropriate PyTorch/CUDA build is machine-specific.
The `numpy<2` guard preserves compatibility with that PyTorch generation.
SegFormer-B0 initialization uses `nvidia/mit-b0` from Hugging Face.

## Dataset Preparation

Datasets, model weights, and generated predictions are not distributed in this
repository. Run commands from the repository root so the relative manifest
paths resolve correctly.

### FloodNet

Prepare the official `FloodNet-Supervised_v1.0` directory:

```text
FloodNet-Supervised_v1.0/
├── train/
│   ├── train-org-img/
│   └── train-label-img/
├── val/
│   ├── val-org-img/
│   └── val-label-img/
└── test/
    ├── test-org-img/
    └── test-label-img/
```

`splits/floodnet/sup398.csv` contains the paper's fixed 398-image training
subset and the official 450/448 validation/test splits. It contains only sample
identifiers and relative paths.

### RescueNet (optional cross-dataset evaluation)

Prepare the official RescueNet layout:

```text
RescueNet/
├── train/{train-org-img,train-label-img}/
├── val/{val-org-img,val-label-img}/
└── test/{test-org-img,test-label-img}/
```

The released manifest uses 3595/449/450 train/validation/test images.
RescueNet JPEG files may be MPO containers; the loader explicitly decodes
frame 0, which is aligned with the mask. To regenerate the manifest:

```bash
python tools/prepare_rescuenet.py \
  --data-root /path/to/RescueNet \
  --output-dir /tmp/rescuenet_manifest
```

Compare the generated manifest summary against
`splits/rescuenet/summary.json` before training.

## Training

FloodNet OS-SegFormer:

```bash
python train.py \
  --config configs/floodnet/os_segformer.yaml \
  --data-root /path/to/FloodNet-Supervised_v1.0
```

Optional RescueNet cross-dataset evaluation:

```bash
python train.py \
  --config configs/rescuenet/os_segformer.yaml \
  --data-root /path/to/RescueNet
```

Use `--dry-run` first to verify the configuration, manifest, dataset root, and
split counts without building a model or training. Checkpoints are selected
only by the configured validation metric.

## SegFormer-B0 Baseline

The same-backbone flat baseline uses the identical data, optimizer, schedule,
augmentation, and evaluation framework:

```bash
python train.py \
  --config configs/floodnet/segformer_b0.yaml \
  --data-root /path/to/FloodNet-Supervised_v1.0
```

External baselines follow their respective public implementations and native
configurations; their third-party source trees are intentionally not vendored.

For the paper's matched cross-dataset control, a RescueNet flat configuration
is also available:

```bash
python train.py \
  --config configs/rescuenet/segformer_b0.yaml \
  --data-root /path/to/RescueNet
```

## Evaluation

```bash
python evaluate.py \
  --config configs/floodnet/os_segformer.yaml \
  --data-root /path/to/FloodNet-Supervised_v1.0 \
  --checkpoint runs/floodnet_os_segformer/checkpoints/best_miou.pth \
  --split validation
```

Use `--split test` only after selecting and freezing a checkpoint from
validation. Add `--save-predictions` to export class-index PNG predictions.
For diagnostics, `--prediction-source` accepts `fused`,
`semantic_direct`, or `hierarchical`.

## Results

All values below are percentages from one fixed-seed run
(`seed=20260702`), not mean ± standard deviation or significance evidence.
Checkpoints were selected on validation; test was used only for final reporting.

### FloodNet (398 training labels)

| Method | Best step | Val mIoU-10 | Test mIoU-10 |
|---|---:|---:|---:|
| SegFormer-B0 | 18,000 | 49.8815 | 47.6675 |
| OS-SegFormer | 30,000 | 61.7060 | 57.4958 |

### RescueNet cross-dataset applicability (official training split)

The RescueNet adaptation, matched baseline configuration, official split
manifest, and reproduction commands are included as a secondary adaptation.
Quantitative RescueNet results are not reported in this release.

## Checkpoints

Pretrained OS-SegFormer checkpoints are not distributed during peer review.
The released training configurations and validation-based selection procedure
are sufficient to reproduce checkpoints locally. Download links may be added
after publication, but weights are not required to use or inspect the code.

## Citation

TODO: replace this placeholder after the ICASSP/arXiv bibliographic metadata
is final.

```bibtex
@inproceedings{os_segformer_todo,
  title     = {Object-State Factorization for Post-Flood Aerial Semantic Segmentation},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```

## Acknowledgements

This implementation builds on SegFormer and the Hugging Face Transformers
implementation of `nvidia/mit-b0`. We thank the authors and maintainers of
FloodNet and RescueNet for making the datasets available.

## Release TODOs

- Add the final paper/arXiv link and citation metadata.
- Choose and add the repository license.
