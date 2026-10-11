"""CPU smoke tests; no dataset, GPU or checkpoint download is required."""
import torch
from sparxo import SPARXO
from sparxo.runtime.config import load_config
from sparxo.runtime.losses import build_loss
from sparxo.runtime.models import build_model
from sparxo.runtime.train import should_validate_epoch, DistributedEvalSampler

def test_original_checkpoint_names_and_outputs():
    torch.set_num_threads(2)
    settings = dict(num_classes=20, dim=32, state_dim=16, temporal_heads=4,
                    unet_base_channels=8, dropout=0.0, temporal_mask_probability=0.0)
    public = SPARXO(**settings).eval()
    original = build_model(dict(architecture="sparxo", **settings)).eval()
    original.load_state_dict(public.state_dict(), strict=True)
    batch = {"s2": torch.randn(1, 3, 10, 16, 16),
             "s2_dates": torch.tensor([[0., 30., 90.]]),
             "s2_valid": torch.tensor([[True, False, True]])}
    with torch.no_grad():
        a = public(batch)["logits"]
        b = original(batch)["logits"]
    torch.testing.assert_close(a, b)

def test_complete_loss_and_optimizer_backward():
    torch.set_num_threads(2)
    model = SPARXO(dim=32, state_dim=16, temporal_heads=4, unet_base_channels=8,
                   temporal_mask_probability=0.0)
    criterion = build_loss(load_config("configs/pastis.yaml")["loss"], 19)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(2, 3, 10, 16, 16)
    dates = torch.tensor([[0., 30., 90.]]).expand(2, -1)
    valid = torch.ones(2, 3, dtype=torch.bool)
    output = model(x, dates, valid)
    labels = torch.randint(0, 20, (2, 16, 16))
    losses = criterion(output, labels)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.temporal_encoder.layers[0].log_decay.grad is not None
    assert model.decoder.affinity_head.weight.grad is not None
    optimizer.step()

def test_two_feature_calendar_projection():
    model = SPARXO(dim=32, state_dim=16, temporal_heads=4, unet_base_channels=8,
                   calendar_features=2)
    assert model.temporal_encoder.date_encoding.projection.in_features == 2

def test_validation_cadence_and_no_duplicate_eval():
    cfg = load_config("configs/pastis.yaml")["train"]
    assert not should_validate_epoch(cfg, 49)
    assert should_validate_epoch(cfg, 50)
    assert should_validate_epoch(cfg, 51)
    indices = [list(DistributedEvalSampler(13, rank, 4)) for rank in range(4)]
    assert sorted(sum(indices, [])) == list(range(13))
