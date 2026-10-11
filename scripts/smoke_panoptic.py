"""Synthetic CPU inference test for the optional U-Net + PaPs task."""
from pathlib import Path
import sys
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "panoptic" / "lib"))
from config import config
from models.sparxo_pano_seg import SPARXOPanopticSegmentor

torch.set_num_threads(2)
cfg = config.clone()
cfg.defrost()
cfg.merge_from_file(str(root / "panoptic" / "configs" / "pastis" / "finetune" /
                        "pastis_panoptic_sparxo_v2_paps_1_Fold0.yml"))
cfg.freeze()
model = SPARXOPanopticSegmentor(cfg, mode="test").eval()
inputs = {"data_S2": torch.randn(1, 11, 3, 32, 32),
          "date_positions_S2": torch.tensor([[[0, 30, 90], [0, 1, 0]]]),
          "label": torch.zeros(1, 8, 32, 32)}
with torch.inference_mode():
    output, _ = model(inputs)
assert output
print("PASS panoptic forward; parameters:", sum(p.numel() for p in model.parameters()))
