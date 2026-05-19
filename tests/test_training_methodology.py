"""Tests for Phase 6 training-methodology utilities.

Three knobs that address §8's "discrete-continuous interface" and
"training data requirements" open problems:

  * Straight-through estimator (STE) — forward returns hard 0/1,
    backward uses the sigmoid gradient. Lets the AND-join enforce
    crisp synchronisation without breaking gradient flow.
  * SharpnessScheduler — anneal the sigmoid sharpness from soft to
    near-step during training.
  * sweep_trace_count — empirical training-data-requirement curve per
    subnet shape.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    PetriNet,
    PetriNetModule,
    SharpnessScheduler,
    parse_bpmn,
    parse_xes,
    sweep_trace_count,
    train_on_traces,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _and_join_net() -> PetriNet:
    net = PetriNet()
    for p in ("p_A", "p_B", "p_out"):
        net.add_place(p)
    net.add_transition("t_merge")
    net.add_arc("p_A", "t_merge")
    net.add_arc("p_B", "t_merge")
    net.add_arc("t_merge", "p_out")
    return net


# ---------------------------------------------------------------------------
# Straight-through estimator
# ---------------------------------------------------------------------------


def test_ste_firing_produces_binary_forward_output():
    torch.manual_seed(0)
    net = _and_join_net()
    module = PetriNetModule(net, firing="ste", sharpness=4.0)
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    with torch.no_grad():
        out = module(input_marking={"p_A": a, "p_B": b})
    for v in out["t_merge"]:
        assert v.item() in {0.0, 1.0}


def test_ste_gradients_flow_through_hard_step():
    torch.manual_seed(0)
    net = _and_join_net()
    module = PetriNetModule(net, firing="ste", sharpness=4.0)
    a = torch.tensor([1.0, 1.0])
    b = torch.tensor([1.0, 0.0])
    out = module(input_marking={"p_A": a, "p_B": b})
    out["p_out"].sum().backward()
    grads = [p.grad for p in module.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0).item() for g in grads)


def test_ste_learns_and_truth_table():
    torch.manual_seed(0)
    net = _and_join_net()
    module = PetriNetModule(net, firing="ste", sharpness=4.0)
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target = torch.tensor([0.0, 0.0, 0.0, 1.0])

    opt = torch.optim.Adam(module.parameters(), lr=0.1)
    for _ in range(2000):
        opt.zero_grad()
        loss = (
            (module(input_marking={"p_A": a, "p_B": b})["p_out"] - target) ** 2
        ).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = module(input_marking={"p_A": a, "p_B": b})["p_out"]
    assert out[0].item() == 0.0
    assert out[1].item() == 0.0
    assert out[2].item() == 0.0
    assert out[3].item() == 1.0


def test_ste_invalid_firing_mode_rejected():
    net = _and_join_net()
    with pytest.raises(ValueError, match="firing"):
        PetriNetModule(net, firing="gumbel")


# ---------------------------------------------------------------------------
# Sharpness annealing scheduler
# ---------------------------------------------------------------------------


def test_sharpness_scheduler_linear_anneal():
    net = _and_join_net()
    module = PetriNetModule(net, sharpness=99.0)
    scheduler = SharpnessScheduler(module, start=1.0, end=5.0, num_steps=4, kind="linear")
    assert module.sharpness == pytest.approx(1.0)
    values = [scheduler.step() for _ in range(4)]
    assert values == pytest.approx([2.0, 3.0, 4.0, 5.0])
    assert scheduler.step() == pytest.approx(5.0)


def test_sharpness_scheduler_exponential_anneal_endpoints():
    net = _and_join_net()
    module = PetriNetModule(net)
    scheduler = SharpnessScheduler(
        module, start=1.0, end=16.0, num_steps=4, kind="exponential"
    )
    assert module.sharpness == pytest.approx(1.0)
    final = None
    for _ in range(4):
        final = scheduler.step()
    assert final == pytest.approx(16.0)


def test_sharpness_anneal_helps_and_join_training():
    """Compare AND-join training with fixed low sharpness vs an anneal
    schedule from soft to sharp. The anneal should reach a lower final
    loss because the early-training gradient flow is preserved while
    the late-training activation is closer to a step."""
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target = torch.tensor([0.0, 0.0, 0.0, 1.0])
    steps = 600

    def loss_for(seed: int, *, use_scheduler: bool) -> float:
        torch.manual_seed(seed)
        module = PetriNetModule(_and_join_net(), sharpness=1.0)
        opt = torch.optim.Adam(module.parameters(), lr=0.1)
        scheduler = (
            SharpnessScheduler(module, start=1.0, end=8.0, num_steps=steps)
            if use_scheduler
            else None
        )
        last = None
        for _ in range(steps):
            opt.zero_grad()
            loss = (
                (module(input_marking={"p_A": a, "p_B": b})["p_out"] - target) ** 2
            ).mean()
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            last = loss.item()
        return last

    annealed = loss_for(0, use_scheduler=True)
    fixed = loss_for(0, use_scheduler=False)
    assert annealed < fixed


def test_sharpness_scheduler_rejects_invalid_kind():
    net = _and_join_net()
    module = PetriNetModule(net)
    with pytest.raises(ValueError, match="kind"):
        SharpnessScheduler(module, start=1.0, end=4.0, num_steps=10, kind="cosine")


# ---------------------------------------------------------------------------
# Training-data-requirement sweep
# ---------------------------------------------------------------------------


def test_sweep_trace_count_returns_one_loss_per_sample_size():
    torch.manual_seed(0)
    traces = parse_xes(FIXTURES / "xor_log.xes")

    def factory():
        torch.manual_seed(0)
        return PetriNetModule(parse_bpmn(FIXTURES / "xor_branch.bpmn"))

    def to_marking(trace):
        return {"p_f0": float(trace.attributes["risk_score"])}

    sizes = [2, 6, 12]
    results = sweep_trace_count(
        factory,
        traces,
        attribute_to_marking=to_marking,
        sample_sizes=sizes,
        steps=600,
        lr=0.1,
    )
    assert set(results) == set(sizes)
    for n in sizes:
        assert results[n] >= 0.0


# ---------------------------------------------------------------------------
# Softmax routing for XOR-shape transitions
# ---------------------------------------------------------------------------


def test_softmax_routing_activations_sum_to_one_across_xor_group():
    """In softmax routing mode, the two competing transitions of an
    XOR-split share probability mass — their activations must sum to
    1 for every batch element. That is the structural difference from
    independent sigmoid firing."""
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    module = PetriNetModule(net, routing="softmax")
    inputs = torch.tensor([0.1, 0.5, 0.9])
    out = module(input_marking={"p_f0": inputs})
    summed = out["t_xor_split_0"] + out["t_xor_split_1"]
    assert torch.allclose(summed, torch.ones_like(summed), atol=1e-5)


def test_softmax_routing_does_not_touch_and_join_transition():
    """An AND-join transition has multiple input places, so it is not
    in any XOR group and must continue to use the configured firing
    function. With sharpness=4 and input (1, 1), the activation should
    be high but not 1.0 (it's a sigmoid, not a softmax output)."""
    torch.manual_seed(0)
    net = PetriNet()
    for p in ("p_A", "p_B", "p_out"):
        net.add_place(p)
    net.add_transition("t_merge")
    net.add_arc("p_A", "t_merge")
    net.add_arc("p_B", "t_merge")
    net.add_arc("t_merge", "p_out")
    module = PetriNetModule(net, routing="softmax", sharpness=4.0)
    out = module(
        input_marking={"p_A": torch.tensor([1.0]), "p_B": torch.tensor([1.0])}
    )
    assert 0.6 < out["t_merge"].item() < 1.0


def test_softmax_routing_invalid_value_rejected():
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    with pytest.raises(ValueError, match="routing"):
        PetriNetModule(net, routing="argmax")


def test_softmax_routing_trains_xor_at_least_as_well_as_independent():
    """Softmax routing has the right inductive bias for XOR shapes —
    exclusive choice among competing branches. On the XOR fixture
    with the same seed and training budget, the softmax model should
    reach a loss no worse than the independent sigmoid model."""
    traces = parse_xes(FIXTURES / "xor_log.xes")

    def to_marking(trace):
        return {"p_f0": float(trace.attributes["risk_score"])}

    def final_loss(routing: str) -> float:
        torch.manual_seed(0)
        module = PetriNetModule(
            parse_bpmn(FIXTURES / "xor_branch.bpmn"), routing=routing
        )
        losses = train_on_traces(
            module, traces, attribute_to_marking=to_marking,
            steps=600, lr=0.1,
        )
        return losses[-1]

    softmax_loss = final_loss("softmax")
    independent_loss = final_loss("independent")
    assert softmax_loss <= independent_loss + 0.05


def test_sweep_more_data_drives_xor_loss_down():
    """The XOR log has 6 high-risk traces and 6 low-risk traces, the
    minimal supervision to learn opposite routing. Training on just 2
    traces (both same class) cannot recover both directions; training
    on the full 12 should."""
    torch.manual_seed(0)
    traces = parse_xes(FIXTURES / "xor_log.xes")

    def factory():
        torch.manual_seed(0)
        return PetriNetModule(parse_bpmn(FIXTURES / "xor_branch.bpmn"))

    def to_marking(trace):
        return {"p_f0": float(trace.attributes["risk_score"])}

    results = sweep_trace_count(
        factory,
        traces,
        attribute_to_marking=to_marking,
        sample_sizes=[2, 12],
        steps=800,
        lr=0.1,
    )
    assert results[12] < results[2]
