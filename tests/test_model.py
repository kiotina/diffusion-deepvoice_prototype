import torch

from deepvoice_diffusion.diffusion import DiffusionSchedule, masked_mse
from deepvoice_diffusion.model import NoisePredictorUNet


def test_model_shape_mask_invariance_and_optimizer_step():
    torch.manual_seed(42)
    model = NoisePredictorUNet(8, 16)
    clean = torch.randn(2, 1, 80, 126)
    mask = torch.ones(2, 1, 1, 126)
    mask[1, ..., 80:] = 0
    steps = torch.tensor([20, 500])
    noise = torch.randn_like(clean)
    noisy = DiffusionSchedule().add_noise(clean, steps, noise, mask)
    output = model(noisy, steps, mask)
    assert output.shape == clean.shape
    modified = noisy.clone()
    modified[1, ..., 80:] = 999
    torch.testing.assert_close(output, model(modified, steps, mask))
    assert not torch.allclose(output, model(noisy, steps + 1, mask))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.input.weight.detach().clone()
    loss = masked_mse(output, noise, mask)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.input.weight)
