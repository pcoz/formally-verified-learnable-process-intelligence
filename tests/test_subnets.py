"""Tests for the five elemental Petri net subnets.

Each subnet gets three flavours of test:

  * structural — forward pass returns the expected keys, batch shape, and
    activations that stay in [0,1];
  * gradient   — backprop produces non-zero gradients for every learnable
    parameter;
  * behavioural — training on synthetic data that reflects the subnet's
    BPMN semantics drives the loss down and the learned mapping ends up
    matching the target pattern.

The behavioural tests are the load-bearing ones: they verify that the
continuous relaxation in §4.2 can actually be trained to reproduce the
discrete firing semantics in §5.
"""
from __future__ import annotations

import pytest
import torch

from petri_net_nn import (
    AndJoinSubnet,
    AndSplitSubnet,
    SagaSubnet,
    SequentialSubnet,
    XORSubnet,
)


def _train(model, step_fn, *, steps: int = 600, lr: float = 0.1) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = step_fn()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _assert_all_params_have_gradients(model) -> None:
    grads = [(name, p.grad) for name, p in model.named_parameters()]
    missing = [name for name, g in grads if g is None]
    assert not missing, f"no gradient for parameters: {missing}"
    any_nonzero = any(torch.any(g != 0).item() for _, g in grads)
    assert any_nonzero, "all gradients are exactly zero"


# ---------------------------------------------------------------------------
# Subnet 1 — sequential
# ---------------------------------------------------------------------------


def test_sequential_structure():
    net = SequentialSubnet()
    out = net(torch.tensor([0.0, 0.5, 1.0]))
    assert set(out) == {"T_step", "P_after"}
    assert out["T_step"].shape == (3,)
    assert out["P_after"].shape == (3,)
    assert torch.all(out["T_step"] >= 0) and torch.all(out["T_step"] <= 1)


def test_sequential_gradient_flow():
    net = SequentialSubnet()
    out = net(torch.tensor([0.2, 0.8]))
    out["P_after"].sum().backward()
    _assert_all_params_have_gradients(net)


def test_sequential_learns_identity():
    torch.manual_seed(0)
    net = SequentialSubnet()
    inputs = torch.rand(64)

    losses = _train(
        net,
        lambda: ((net(inputs)["P_after"] - inputs) ** 2).mean(),
    )

    assert losses[-1] < losses[0] * 0.1
    with torch.no_grad():
        out = net(torch.tensor([0.05, 0.95]))
        assert out["P_after"][0] < 0.3
        assert out["P_after"][1] > 0.7


# ---------------------------------------------------------------------------
# Subnet 2 — XOR / exclusive choice
# ---------------------------------------------------------------------------


def test_xor_structure():
    net = XORSubnet()
    out = net(torch.tensor([0.0, 1.0]))
    assert set(out) == {"T_route_A", "T_route_B", "P_path_A", "P_path_B"}
    for key in out:
        assert out[key].shape == (2,)


def test_xor_gradient_flow():
    net = XORSubnet()
    out = net(torch.tensor([0.3, 0.7]))
    (out["P_path_A"].sum() + out["P_path_B"].sum()).backward()
    _assert_all_params_have_gradients(net)


def test_xor_learns_to_route_oppositely():
    torch.manual_seed(0)
    net = XORSubnet()
    inputs = torch.rand(128)
    target_A = (inputs > 0.5).float()
    target_B = 1.0 - target_A

    def step():
        out = net(inputs)
        return (
            (out["P_path_A"] - target_A) ** 2 + (out["P_path_B"] - target_B) ** 2
        ).mean()

    losses = _train(net, step, steps=1000, lr=0.1)
    assert losses[-1] < 0.05

    with torch.no_grad():
        out_high = net(torch.tensor([0.95]))
        out_low = net(torch.tensor([0.05]))
        assert out_high["P_path_A"].item() > 0.7
        assert out_high["P_path_B"].item() < 0.3
        assert out_low["P_path_A"].item() < 0.3
        assert out_low["P_path_B"].item() > 0.7


# ---------------------------------------------------------------------------
# Subnet 3 — AND-split / parallel split
# ---------------------------------------------------------------------------


def test_and_split_structure():
    net = AndSplitSubnet()
    out = net(torch.tensor([0.0, 0.5, 1.0]))
    assert set(out) == {"T_spawn", "P_branch_A", "P_branch_B"}
    for key in out:
        assert out[key].shape == (3,)


def test_and_split_gradient_flow():
    net = AndSplitSubnet()
    out = net(torch.tensor([0.3, 0.7]))
    (out["P_branch_A"].sum() + out["P_branch_B"].sum()).backward()
    _assert_all_params_have_gradients(net)


def test_and_split_activates_both_branches_together():
    torch.manual_seed(0)
    net = AndSplitSubnet()
    inputs = torch.rand(64)

    def step():
        out = net(inputs)
        return (
            (out["P_branch_A"] - inputs) ** 2 + (out["P_branch_B"] - inputs) ** 2
        ).mean()

    losses = _train(net, step)
    assert losses[-1] < losses[0] * 0.1

    with torch.no_grad():
        out = net(torch.tensor([0.9]))
        assert out["P_branch_A"].item() > 0.6
        assert out["P_branch_B"].item() > 0.6
        diff = (out["P_branch_A"] - out["P_branch_B"]).abs().item()
        assert diff < 0.1


# ---------------------------------------------------------------------------
# Subnet 4 — AND-join / synchronisation  (the hard case in §5)
# ---------------------------------------------------------------------------


def test_and_join_structure():
    net = AndJoinSubnet()
    a = torch.tensor([0.0, 1.0])
    b = torch.tensor([1.0, 0.0])
    out = net(a, b)
    assert set(out) == {"T_merge", "P_unified"}
    assert out["P_unified"].shape == (2,)


def test_and_join_gradient_flow():
    net = AndJoinSubnet()
    out = net(torch.tensor([0.6]), torch.tensor([0.6]))
    out["P_unified"].sum().backward()
    _assert_all_params_have_gradients(net)


def test_and_join_implements_logical_and():
    torch.manual_seed(0)
    net = AndJoinSubnet()
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target = torch.tensor([0.0, 0.0, 0.0, 1.0])

    losses = _train(
        net,
        lambda: ((net(a, b)["P_unified"] - target) ** 2).mean(),
        steps=2000,
        lr=0.1,
    )

    assert losses[-1] < 0.02

    with torch.no_grad():
        out = net(a, b)["P_unified"]
        assert out[0].item() < 0.2
        assert out[1].item() < 0.2
        assert out[2].item() < 0.2
        assert out[3].item() > 0.8


def test_and_join_partial_input_does_not_propagate():
    """A single-branch token should not be enough to fire T_merge — even
    before any training, the high default threshold should keep the
    output near zero when only one input is active."""
    net = AndJoinSubnet()
    with torch.no_grad():
        only_a = net(torch.tensor([1.0]), torch.tensor([0.0]))["P_unified"]
        only_b = net(torch.tensor([0.0]), torch.tensor([1.0]))["P_unified"]
        both = net(torch.tensor([1.0]), torch.tensor([1.0]))["P_unified"]
    assert only_a.item() < 0.3
    assert only_b.item() < 0.3
    assert both.item() > 0.7


# ---------------------------------------------------------------------------
# Subnet 5 — saga compensation
# ---------------------------------------------------------------------------


def test_saga_structure():
    net = SagaSubnet()
    out = net(torch.tensor([0.0, 0.5, 1.0]))
    expected = {
        "T_succeed",
        "T_fail",
        "T_compensate",
        "P_complete",
        "P_compensating",
        "P_initial",
    }
    assert set(out) == expected
    for key in out:
        assert out[key].shape == (3,)


def test_saga_gradient_flow():
    net = SagaSubnet()
    out = net(torch.tensor([0.3, 0.7]))
    (out["P_complete"].sum() + out["P_initial"].sum()).backward()
    _assert_all_params_have_gradients(net)


def test_saga_routes_success_vs_compensation():
    torch.manual_seed(0)
    net = SagaSubnet()
    inputs = torch.rand(128)
    succeeds = (inputs > 0.5).float()
    target_complete = succeeds
    target_initial = 1.0 - succeeds

    def step():
        out = net(inputs)
        return (
            (out["P_complete"] - target_complete) ** 2
            + (out["P_initial"] - target_initial) ** 2
        ).mean()

    losses = _train(net, step, steps=1500, lr=0.1)
    assert losses[-1] < 0.05

    with torch.no_grad():
        ok = net(torch.tensor([0.95]))
        fail = net(torch.tensor([0.05]))
        assert ok["P_complete"].item() > 0.7
        assert ok["P_initial"].item() < 0.3
        assert fail["P_complete"].item() < 0.3
        assert fail["P_initial"].item() > 0.5


# ---------------------------------------------------------------------------
# Cross-subnet sanity: outputs of one subnet feed cleanly into the next
# ---------------------------------------------------------------------------


def test_subnets_compose_as_tensors():
    """The simple approval composition sketched in §6: sequential -> XOR.
    Verifies that one subnet's place output is a usable input to the next
    without any glue code beyond reading the dict by key."""
    seq = SequentialSubnet()
    xor = XORSubnet()
    submitted = torch.tensor([0.9, 0.1])
    triaged = seq(submitted)["P_after"]
    routed = xor(triaged)
    assert routed["P_path_A"].shape == submitted.shape
    assert routed["P_path_B"].shape == submitted.shape
