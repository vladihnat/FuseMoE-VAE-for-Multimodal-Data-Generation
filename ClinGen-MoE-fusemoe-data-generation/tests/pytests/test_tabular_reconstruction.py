import sys
from pathlib import Path

import pytest
import torch

from losses.reconstruction import tabular_reconstruction_loss


def test_numeric_only_loss():
    target_num = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pred_num = torch.tensor([[1.0, 1.0], [2.0, 4.0]])

    out = tabular_reconstruction_loss(
        target_num=target_num,
        pred_num=pred_num,
        target_cat=None,
        pred_cat_logits=None,
    )

    assert out["num_loss"].ndim == 0
    assert out["cat_loss"].ndim == 0
    assert torch.isclose(out["total"], out["num_loss"])


def test_categorical_only_loss():
    target_cat = torch.tensor([[0, 1], [2, 0]])
    pred_cat_logits = [
        torch.tensor([[4.0, 1.0, 0.0], [0.1, 0.2, 3.0]]),
        torch.tensor([[0.1, 2.0], [3.0, 0.2]]),
    ]

    out = tabular_reconstruction_loss(
        target_num=None,
        pred_num=None,
        target_cat=target_cat,
        pred_cat_logits=pred_cat_logits,
    )

    assert out["num_loss"].ndim == 0
    assert out["cat_loss"].ndim == 0
    assert torch.isclose(out["total"], out["cat_loss"])


def test_mixed_loss_and_backward():
    target_num = torch.randn(4, 3)
    pred_num = torch.randn(4, 3, requires_grad=True)
    target_cat = torch.tensor([[0, 1], [1, 0], [2, 1], [0, 0]])
    pred_cat_logits = [
        torch.randn(4, 3, requires_grad=True),
        torch.randn(4, 2, requires_grad=True),
    ]

    out = tabular_reconstruction_loss(
        target_num=target_num,
        pred_num=pred_num,
        target_cat=target_cat,
        pred_cat_logits=pred_cat_logits,
    )
    out["total"].backward()

    assert pred_num.grad is not None
    assert pred_cat_logits[0].grad is not None
    assert pred_cat_logits[1].grad is not None
    assert torch.isfinite(pred_num.grad).all()


def test_shape_mismatch_raises():
    target_num = torch.randn(4, 3)
    pred_num = torch.randn(4, 4)

    with pytest.raises(ValueError, match="same shape"):
        tabular_reconstruction_loss(
            target_num=target_num,
            pred_num=pred_num,
            target_cat=None,
            pred_cat_logits=None,
        )


def test_wrong_number_of_heads_raises():
    target_cat = torch.tensor([[0, 1], [1, 0]])
    pred_cat_logits = [torch.randn(2, 3)]

    with pytest.raises(ValueError, match="categorical heads"):
        tabular_reconstruction_loss(
            target_num=None,
            pred_num=None,
            target_cat=target_cat,
            pred_cat_logits=pred_cat_logits,
        )
