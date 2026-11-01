"""Minimal synthetic forward/backward check. This is not model evaluation."""
import torch

from models import Objective, create_model


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2027)
    model = create_model('dscnet_fc_v2', seed=2027).train()
    image = torch.randn(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 15:17] = 1

    logits, aux = model(image, return_aux=True)
    assert logits.shape == mask.shape
    loss, parts = Objective('dscnet_fc_v2')(logits, aux, mask)
    assert torch.isfinite(loss), 'Loss must be finite'
    loss.backward()
    assert model.feedback.feedback[-1].weight.grad is not None
    assert torch.isfinite(model.feedback.feedback[-1].weight.grad).all()
    print('Output logits:', tuple(logits.shape))
    print('Loss terms:', ', '.join(sorted(parts)))
    print('Forward and backward: OK')


if __name__ == '__main__':
    main()
