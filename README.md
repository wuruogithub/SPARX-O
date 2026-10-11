# SPARX-O

**State-space Phenology-Aware Representation with spatial conteXt for Optical satellite time series.**

This repository contains the complete SPARX-O optical model, including the
Calendar-Phase State Space (CPS) encoder, adaptive temporal readout, semantic
decoder, losses, data loading, training and evaluation. The optional panoptic
pipeline couples the same temporal encoder to U-Net and PaPs.

## Architecture

```text
S2 [B,T,10,H,W] + acquisition dates [B,T] + valid mask [B,T]
  -> lightweight spatial Frame Stem
  -> sensor residual adapter
  -> calendar embedding + two CPS layers
  -> 8-head valid-only adaptive temporal readout
  -> aggregated feature [B,160,H,W]
     -> U-Net + boundary/affinity closed-loop refinement (semantic)
     or
     -> U-Net + PaPs (panoptic)
```

CPS uses gaps between consecutive valid observations to drive memory decay
and complex-state phase rotation. Missing observations hold the recurrent
state. Decay rates, frequencies and bounded content corrections are learned.

## Installation

Use Python 3.10 or 3.11 and install a PyTorch build matching your CUDA runtime
from the [official installation instructions](https://pytorch.org/get-started/locally/).
Then:

```bash
git clone https://github.com/wuruogithub/SPARX-O.git
cd SPARX-O
pip install -e '.[dev]'
# Optional panoptic dependencies:
pip install -e '.[panoptic]'
# Optional TFRecord conversion:
pip install -e '.[mtlcc]'
```

The PaPs implementation includes a native PyTorch scatter-reduction fallback.

## Model API

```python
import torch
from sparxo import SPARXO

model = SPARXO().eval()
s2 = torch.randn(1, 3, 10, 32, 32)
# Elapsed days from 2018-01-01, not ordinal observation indices.
dates = torch.tensor([[0., 30., 90.]])
valid = torch.tensor([[True, False, True]])
with torch.no_grad():
    outputs = model(s2, dates, valid, return_aux=True)
prediction = outputs["logits"].argmax(dim=1)
```

Outputs include final/base semantic logits, boundary logits, four-neighbour
affinity logits, decoder/shape features and the refinement gate. Auxiliary
mode additionally exposes attention and temporal features. At least one
observation must be valid in every sample. 

## Datasets and splits

Download PASTIS-R / MTLCC separately under their respective dataset terms.
No dataset or third-party personal assets are redistributed.
PASTIS-R: https://zenodo.org/records/5735646
MTLCC: https://zenodo.org/records/5712933

PASTIS-R needs `metadata.geojson`, `DATA_S2/S2_<id>.npy`,
`ANNOTATIONS/TARGET_<id>.npy` and `NORM_S2_patch.json`.
Edit `data.root` in `configs/pastis.yaml`. Normalization uses only the
training folds; adaptive crop weights and proportions use training labels only.
Void (19) is ignored; background (0) remains an evaluated class.

The default split trains on data folds 1/2/3, validates on 4 and tests on 5.
This is **Fold 1** in the paper's cross-validation order. The full mapping is:

| Paper fold | Train data folds | Validation data fold | Test data fold |
|---|---|---|---|
| 1 | 1,2,3 | 4 | 5 |
| 2 | 2,3,4 | 5 | 1 |
| 3 | 3,4,5 | 1 | 2 |
| 4 | 4,5,1 | 2 | 3 |
| 5 | 5,1,2 | 3 | 4 |

For MTLCC, use the official 2016 full/240 TFRecords and its
`tileids/train_fold0.tileids`, `eval.tileids`, `test_fold0.tileids` split.

## Semantic training and evaluation

```bash
bash scripts/train.sh configs/pastis.yaml
bash scripts/train.sh configs/mtlcc_scratch.yaml

bash scripts/train.sh configs/pastis.yaml \
  --resume outputs/sparxo_v2_u4_pastis_fold5_seed42/checkpoints/last.pt

python -m sparxo.runtime.train --config configs/pastis.yaml \
  --evaluate outputs/sparxo_v2_u4_pastis_fold5_seed42/checkpoints/best_val.pt \
  --evaluate-test
```

The training implementation includes AdamW, warm-up/scheduling, gradient
clipping, AMP, EMA, validation-based selection, checkpoint/resume and per-class
metrics. PASTIS uses CE + Dice + Lovasz + boundary + continuity + affinity losses.
Validation is every two epochs through epoch 50 and every epoch thereafter.
There is no test-time augmentation by default.

The reference PASTIS configuration uses a true global batch of 48
; MTLCC uses 112.
Changing GPU count requires a matching batch/accumulation configuration.
Numerical results can vary with software, worker RNG and update-count differences.

## Panoptic segmentation

```bash
pip install -e '.[panoptic]'
bash scripts/train_panoptic.sh 5 \
  outputs/sparxo_v2_u4_pastis_fold5_seed42/checkpoints/best_val.pt \
  /path/to/PASTIS-R outputs/panoptic_test_data_fold5
```

The script exports only temporal weights from the same-split semantic
checkpoint; PaPs/spatial heads are newly initialized. Don't initialize a fold
from a checkpoint whose training split contains that fold's test samples.
The wrapper defaults to 150 epochs, minimum stopping epoch 70 and patience 30.

## Reported results

| Task | Dataset | Metric |
|---|---|---|
| Pixel semantic segmentation | PASTIS, five folds | mIoU 69.78 ± 1.21% |
| Pixel semantic segmentation, from scratch | MTLCC | mIoU 91.00% |
| Parcel panoptic segmentation | PASTIS, five folds | PQ 53.47 ± 0.45% |

These are the manuscript's reported experiments. Checkpoints are not bundled with the source release.

## Validation

```bash
OMP_NUM_THREADS=2 pytest -q
python scripts/smoke_test.py
python scripts/smoke_panoptic.py  # requires the panoptic extras
python examples/inspect_model.py
```

Tests cover the full forward pass, original parameter count and state-dict
names, missing-observation state hold, real gaps, all six semantic loss terms,
backpropagation, calendar variants and validation/evaluation scheduling.


## License and attribution

SPARX-O is released under the **MIT License**; see `LICENSE` and `NOTICE.md`.
MIT permits reuse but requires preserving copyright and license notices.
If this code informs an academic publication, please cite the SPARX-O paper
once its final bibliographic information is available; no fabricated DOI or
publication venue is supplied here.
