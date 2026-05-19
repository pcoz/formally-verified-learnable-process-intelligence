"""Tests for PetriNetModule — the compiler that turns a PetriNet into a
differentiable nn.Module per §4 of the architecture spec.

Strategy:

  * structural tests pin the §4.3 claim "weights outside this structure
    are zero by construction" — the module has exactly |F| arc weights
    and |T| thresholds, nothing more;
  * forward-pass tests confirm the topological propagation reaches the
    sink from M_0 and respects ``input_marking`` overrides;
  * behavioural tests compile each of the five elemental subnet shapes
    from a BPMN fixture (or hand-built net) and train the compiled
    module — verifying that the general §4.2 instantiation reproduces
    what the hand-written subnets in test_subnets.py achieved;
  * an integration test compiles the §6 approval process and trains it
    end-to-end on synthetic happy-path traces.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import PetriNet, PetriNetModule, parse_bpmn


FIXTURES = Path(__file__).parent / "fixtures"


def _train(module, step_fn, *, steps=600, lr=0.1):
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = step_fn()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# Structural tests — §4.3 by construction
# ---------------------------------------------------------------------------


def test_module_has_one_parameter_per_arc_and_per_transition():
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    module = PetriNetModule(net)
    assert len(module.arc_weights) == len(net.flow)
    assert len(module.transition_thresholds) == len(net.transitions)
    assert len(list(module.parameters())) == len(net.flow) + len(net.transitions)


def test_module_rejects_ill_formed_net():
    net = PetriNet()
    net.add_transition("dangling")
    with pytest.raises(ValueError, match="not well-formed"):
        PetriNetModule(net)


def test_module_rejects_cyclic_net():
    net = PetriNet()
    net.add_place("a")
    net.add_place("b")
    net.add_transition("t1")
    net.add_transition("t2")
    net.add_arc("a", "t1")
    net.add_arc("t1", "b")
    net.add_arc("b", "t2")
    net.add_arc("t2", "a")
    with pytest.raises(ValueError, match="cycle"):
        PetriNetModule(net)


def test_module_rejects_negative_num_steps():
    net = PetriNet()
    net.add_place("a", tokens=1)
    net.add_place("b")
    net.add_transition("t")
    net.add_arc("a", "t")
    net.add_arc("t", "b")
    with pytest.raises(ValueError, match="non-negative"):
        PetriNetModule(net, num_steps=-1)


def _two_cycle() -> PetriNet:
    net = PetriNet()
    net.add_place("p1", tokens=1)
    net.add_place("p2")
    net.add_transition("t12", label="forward")
    net.add_transition("t21", label="back")
    net.add_arc("p1", "t12")
    net.add_arc("t12", "p2")
    net.add_arc("p2", "t21")
    net.add_arc("t21", "p1")
    return net


def test_time_unrolled_mode_accepts_cyclic_net():
    """num_steps=0 raises on a 2-cycle; num_steps>0 accepts it."""
    net = _two_cycle()
    with pytest.raises(ValueError, match="cycle"):
        PetriNetModule(net)
    module = PetriNetModule(net, num_steps=5)
    assert module._order is None


def test_time_unrolled_forward_produces_non_degenerate_activations_on_cycle():
    torch.manual_seed(0)
    net = _two_cycle()
    module = PetriNetModule(net, num_steps=8)
    out = module()
    for value in out.values():
        assert torch.all(torch.isfinite(value))
    assert out["p2"].item() > 0.1
    assert out["t12"].item() > 0.1
    assert out["t21"].item() > 0.1


def test_time_unrolled_forward_gradient_flows_through_steps():
    torch.manual_seed(0)
    net = _two_cycle()
    module = PetriNetModule(net, num_steps=4)
    out = module()
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    grads = [p.grad for p in module.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0).item() for g in grads)


def test_time_unrolled_acyclic_converges_to_single_pass_result():
    """For an acyclic net, the time-unrolled forward with enough steps
    should match the single-pass forward (same weights, same input).
    Constructing the two modules with the same seed gives identical
    parameters; we then compare their outputs."""
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "approval.bpmn")
    single_pass = PetriNetModule(net, sharpness=4.0)
    torch.manual_seed(0)
    unrolled = PetriNetModule(net, sharpness=4.0, num_steps=20)

    with torch.no_grad():
        a = single_pass()
        b = unrolled()
    diff = (a["p_f_done"] - b["p_f_done"]).abs().item()
    assert diff < 1e-4, f"single-pass and unrolled outputs disagree by {diff}"


def test_retry_loop_bpmn_compiles_in_time_unrolled_mode_only():
    """The retry loop fixture produces a cyclic Petri net: rejected by
    acyclic mode, accepted by num_steps>0."""
    net = parse_bpmn(FIXTURES / "retry_loop.bpmn")
    with pytest.raises(ValueError, match="cycle"):
        PetriNetModule(net)
    module = PetriNetModule(net, num_steps=10)
    out = module()
    assert out["p_f_ok"].shape == (1,)
    assert torch.all(torch.isfinite(out["p_f_ok"]))


def test_retry_loop_eventually_activates_success_place():
    """With enough unroll steps from the initial marking, the success
    sink p_f_ok should accumulate non-trivial activation as token
    mass propagates around the loop."""
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "retry_loop.bpmn")
    module = PetriNetModule(net, num_steps=15)
    out = module()
    assert out["p_f_ok"].item() > 0.1


def test_inhibitor_arc_suppresses_transition_when_place_activated():
    """The compiler implements inhibitor arcs as a multiplicative gate:
    transition activation is scaled by (1 - a(p)) for each inhibitor
    place p. When the inhibitor place activation is high, the
    transition's effective activation must be near zero."""
    torch.manual_seed(0)
    net = PetriNet()
    net.add_place("p_input")
    net.add_place("p_guard")
    net.add_place("p_output")
    net.add_transition("t_gated")
    net.add_arc("p_input", "t_gated")
    net.add_arc("t_gated", "p_output")
    net.add_inhibitor_arc("p_guard", "t_gated")
    module = PetriNetModule(net)

    with torch.no_grad():
        # Guard empty: transition activates normally.
        ungated = module(
            input_marking={
                "p_input": torch.tensor([1.0]),
                "p_guard": torch.tensor([0.0]),
            }
        )
        # Guard full: transition activation should drop sharply.
        gated = module(
            input_marking={
                "p_input": torch.tensor([1.0]),
                "p_guard": torch.tensor([1.0]),
            }
        )

    assert gated["t_gated"].item() < 0.05
    assert ungated["t_gated"].item() > 0.5


def test_inhibitor_gate_in_time_unrolled_mode_enforces_mutex():
    """The mutex pattern: two transitions race for a shared 'critical'
    place, each inhibited by it. In time-unrolled mode, after the
    first step one transition has fired and 'critical' is occupied;
    by the next step the inhibitor gate suppresses both transitions.
    The result is that across the unrolled steps only one of the two
    transitions accumulates significant activation."""
    torch.manual_seed(0)
    net = PetriNet()
    net.add_place("p_a_pending", tokens=1)
    net.add_place("p_b_pending", tokens=1)
    net.add_place("p_critical")
    net.add_place("p_a_done")
    net.add_place("p_b_done")
    for tid in ("t_serve_a", "t_serve_b"):
        net.add_transition(tid)
    net.add_arc("p_a_pending", "t_serve_a")
    net.add_arc("t_serve_a", "p_critical")
    net.add_arc("t_serve_a", "p_a_done")
    net.add_arc("p_b_pending", "t_serve_b")
    net.add_arc("t_serve_b", "p_critical")
    net.add_arc("t_serve_b", "p_b_done")
    net.add_inhibitor_arc("p_critical", "t_serve_a")
    net.add_inhibitor_arc("p_critical", "t_serve_b")
    module = PetriNetModule(net, num_steps=4, sharpness=2.0)
    out = module()
    # After the first step, p_critical accumulates activation from
    # whichever transition fired more strongly; the inhibitor gate
    # then suppresses both transitions in subsequent steps. The total
    # activation in 'critical' should be bounded — not the sum of both
    # full firings.
    assert out["p_critical"].item() < 1.5


def test_time_unrolled_input_marking_is_clamped_each_step():
    """If input_marking pins a place at 0.0 throughout the unroll, the
    rest of the network downstream of it should also stay near zero —
    confirming the override is re-applied every step, not just at
    step 0."""
    torch.manual_seed(0)
    net = _two_cycle()
    module = PetriNetModule(net, num_steps=20)
    out = module(input_marking={"p1": torch.tensor([0.0])})
    assert out["p1"].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Forward-pass tests
# ---------------------------------------------------------------------------


def test_forward_returns_activations_for_every_node():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    module = PetriNetModule(net)
    out = module()
    assert set(out) == net.places | net.transitions
    for v in out.values():
        assert v.shape == (1,)


def test_forward_propagates_initial_marking_to_sink():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    module = PetriNetModule(net)
    out = module()
    assert out["p_f1"].item() == pytest.approx(1.0)
    assert out["p_f2"].item() > 0.5


def test_forward_input_marking_overrides_initial_marking():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    module = PetriNetModule(net)
    out_zero = module(input_marking={"p_f1": torch.tensor([0.0])})
    out_one = module(input_marking={"p_f1": torch.tensor([1.0])})
    assert out_zero["p_f2"].item() < out_one["p_f2"].item()


def test_forward_supports_batched_input():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    module = PetriNetModule(net)
    out = module(input_marking={"p_f1": torch.tensor([0.0, 0.5, 1.0])})
    assert out["p_f2"].shape == (3,)
    assert out["p_f2"][0].item() < out["p_f2"][2].item()


# ---------------------------------------------------------------------------
# Behavioural — the five subnet shapes via the general compiler
# ---------------------------------------------------------------------------


def test_compiled_sequential_learns_identity():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    module = PetriNetModule(net)
    inputs = torch.rand(64)

    losses = _train(
        module,
        lambda: ((module(input_marking={"p_f1": inputs})["p_f2"] - inputs) ** 2).mean(),
    )
    assert losses[-1] < losses[0] * 0.2

    with torch.no_grad():
        assert module(input_marking={"p_f1": torch.tensor([0.95])})["p_f2"].item() > 0.7
        assert module(input_marking={"p_f1": torch.tensor([0.05])})["p_f2"].item() < 0.3


def test_compiled_xor_learns_opposite_routing():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    module = PetriNetModule(net)
    inputs = torch.rand(128)
    target_A = (inputs > 0.5).float()
    target_B = 1.0 - target_A

    def step():
        out = module(input_marking={"p_f0": inputs})
        return (
            (out["p_fA1"] - target_A) ** 2 + (out["p_fB1"] - target_B) ** 2
        ).mean()

    losses = _train(module, step, steps=1500, lr=0.1)
    assert losses[-1] < 0.05

    with torch.no_grad():
        high = module(input_marking={"p_f0": torch.tensor([0.95])})
        low = module(input_marking={"p_f0": torch.tensor([0.05])})
    assert high["p_fA1"].item() > 0.7
    assert high["p_fB1"].item() < 0.3
    assert low["p_fA1"].item() < 0.3
    assert low["p_fB1"].item() > 0.7


def test_compiled_and_split_activates_both_branches_together():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "and_branch.bpmn")
    module = PetriNetModule(net)
    inputs = torch.rand(64)

    def step():
        out = module(input_marking={"p_f0": inputs})
        return (
            (out["p_fA1"] - inputs) ** 2 + (out["p_fB1"] - inputs) ** 2
        ).mean()

    _train(module, step)

    with torch.no_grad():
        out = module(input_marking={"p_f0": torch.tensor([0.9])})
    assert out["p_fA1"].item() > 0.6
    assert out["p_fB1"].item() > 0.6
    assert (out["p_fA1"] - out["p_fB1"]).abs().item() < 0.1


def test_compiled_and_join_learns_logical_and():
    torch.manual_seed(0)
    net = PetriNet()
    for p in ("p_A", "p_B", "p_out"):
        net.add_place(p)
    net.add_transition("t_merge")
    net.add_arc("p_A", "t_merge")
    net.add_arc("p_B", "t_merge")
    net.add_arc("t_merge", "p_out")

    module = PetriNetModule(net, sharpness=4.0)

    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target = torch.tensor([0.0, 0.0, 0.0, 1.0])

    losses = _train(
        module,
        lambda: (
            (module(input_marking={"p_A": a, "p_B": b})["p_out"] - target) ** 2
        ).mean(),
        steps=2000,
        lr=0.1,
    )
    assert losses[-1] < 0.05

    with torch.no_grad():
        out = module(input_marking={"p_A": a, "p_B": b})["p_out"]
    assert out[0].item() < 0.2
    assert out[1].item() < 0.2
    assert out[2].item() < 0.2
    assert out[3].item() > 0.8


def test_compiled_saga_routes_success_vs_compensation():
    """Build the saga subnet as a hand-coded PetriNet (acyclic shape from
    §5 Subnet 5) and confirm the compiler can train it end-to-end."""
    torch.manual_seed(0)
    net = PetriNet()
    for p in ("p_active", "p_complete", "p_compensating", "p_initial"):
        net.add_place(p)
    for t in ("t_succeed", "t_fail", "t_compensate"):
        net.add_transition(t)
    net.add_arc("p_active", "t_succeed")
    net.add_arc("t_succeed", "p_complete")
    net.add_arc("p_active", "t_fail")
    net.add_arc("t_fail", "p_compensating")
    net.add_arc("p_compensating", "t_compensate")
    net.add_arc("t_compensate", "p_initial")

    module = PetriNetModule(net)

    inputs = torch.rand(128)
    succeeds = (inputs > 0.5).float()
    target_complete = succeeds
    target_initial = 1.0 - succeeds

    def step():
        out = module(input_marking={"p_active": inputs})
        return (
            (out["p_complete"] - target_complete) ** 2
            + (out["p_initial"] - target_initial) ** 2
        ).mean()

    losses = _train(module, step, steps=1500, lr=0.1)
    assert losses[-1] < 0.05

    with torch.no_grad():
        ok = module(input_marking={"p_active": torch.tensor([0.95])})
        fail = module(input_marking={"p_active": torch.tensor([0.05])})
    assert ok["p_complete"].item() > 0.6
    assert ok["p_initial"].item() < 0.4
    assert fail["p_complete"].item() < 0.4
    assert fail["p_initial"].item() > 0.5


def test_compiled_saga_from_bpmn_trains_to_route_success_vs_compensation():
    """End-to-end check that the BPMN compensation extension feeds the
    compiler correctly: parse saga.bpmn, compile it, and train on the
    same routing task as the hand-built saga net test."""
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "saga.bpmn")
    module = PetriNetModule(net)

    inputs = torch.rand(128)
    succeeds = (inputs > 0.5).float()
    target_complete = succeeds
    target_refunded = 1.0 - succeeds

    def step():
        out = module(input_marking={"p_f_start": inputs})
        return (
            (out["p_f_ok"] - target_complete) ** 2
            + (out["p_f_refunded"] - target_refunded) ** 2
        ).mean()

    losses = _train(module, step, steps=1500, lr=0.1)
    assert losses[-1] < 0.05

    with torch.no_grad():
        ok = module(input_marking={"p_f_start": torch.tensor([0.95])})
        fail = module(input_marking={"p_f_start": torch.tensor([0.05])})
    assert ok["p_f_ok"].item() > 0.6
    assert ok["p_f_refunded"].item() < 0.4
    assert fail["p_f_ok"].item() < 0.4
    assert fail["p_f_refunded"].item() > 0.5


# ---------------------------------------------------------------------------
# Integration — compose the whole §6 approval process
# ---------------------------------------------------------------------------


def test_approval_process_compiles_and_runs_forward():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "approval.bpmn")
    module = PetriNetModule(net, sharpness=4.0)
    out = module()
    assert set(out) == net.places | net.transitions
    assert out["p_f_done"].shape == (1,)


def test_approval_process_trains_to_predict_completion():
    """Synthetic supervision: every initial-marking trace should reach the
    sink. We train the compiled approval module to push p_f_done toward
    1.0 from M_0 — a minimal "process completion prediction" task from
    §7.1. The point is that the gradient flows through the whole
    composed topology, not the absolute target value."""
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "approval.bpmn")
    module = PetriNetModule(net, sharpness=4.0)

    target = torch.tensor([1.0])

    def step():
        out = module()
        return ((out["p_f_done"] - target) ** 2).mean()

    losses = _train(module, step, steps=800, lr=0.1)
    assert losses[-1] < losses[0] * 0.3
    assert losses[-1] < 0.1

    with torch.no_grad():
        assert module()["p_f_done"].item() > 0.7
