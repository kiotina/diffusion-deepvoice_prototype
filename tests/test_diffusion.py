import pytest
import torch

from deepvoice_diffusion.diffusion import DiffusionSchedule, masked_mse, seeded_noise


def test_noising_matches_formula_and_keeps_padding_constant():
    schedule = DiffusionSchedule(10, 0.01, 0.1)
    x = torch.ones(2, 1, 2, 3)
    noise = torch.full_like(x, 2)
    mask = torch.tensor([[[[1., 1., 0.]]]]).repeat(2, 1, 1, 1)
    actual = schedule.add_noise(x, torch.tensor([0, 9]), noise, mask)
    for i, t in enumerate((0, 9)):
        alpha = torch.prod(1 - torch.linspace(0.01, 0.1, 10)[:t+1])
        torch.testing.assert_close(actual[i, ..., :2], torch.full((1, 2, 2), alpha.sqrt() + 2*(1-alpha).sqrt()))
    assert (actual[..., 2] == -1).all()


def test_masked_loss_value_and_gradient_ignore_padding():
    prediction = torch.zeros(1, 1, 2, 3, requires_grad=True)
    target = torch.ones_like(prediction)
    target[..., 2] = 999
    mask = torch.tensor([[[[1., 1., 0.]]]])
    loss = masked_mse(prediction, target, mask)
    assert loss.item() == 1
    loss.backward()
    assert (prediction.grad[..., 2] == 0).all()
    assert (prediction.grad[..., :2] != 0).all()
    with pytest.raises(ValueError, match="valid frames"):
        masked_mse(prediction, target, torch.zeros_like(mask))


def test_validation_noise_is_independent_of_batch_size():
    x = torch.zeros(3, 1, 2, 5)
    steps, noises = seeded_noise(x, [0, 1, 2], 1000, 42, 0)
    for i in range(3):
        t, n = seeded_noise(x[i:i+1], [i], 1000, 42, 0)
        torch.testing.assert_close(t[0], steps[i])
        torch.testing.assert_close(n[0], noises[i])
    assert not torch.equal(noises, seeded_noise(x, [0, 1, 2], 1000, 42, 1)[1])
