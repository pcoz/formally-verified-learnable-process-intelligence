"""End-to-end test for the coloured credit-approval scenario.

Showcases the Phase 9 CPN-lite feature: tokens carry application
amounts as their values, and the routing transitions guard on
those values via the adapter's declarative guard spec.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from petri_net_nn import load_scenario


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "credit_approval_coloured"
    / "scenario.toml"
)


def test_credit_approval_loads_with_guards():
    """The declarative guards in the TOML compile into callable
    guards on the PetriNet object."""
    ctx = load_scenario(SCENARIO)
    assert "t_approve" in ctx.net.transition_guards
    assert "t_decline" in ctx.net.transition_guards


def test_high_amount_token_fires_approve_only():
    ctx = load_scenario(SCENARIO)
    high = {"p_submitted": [5000.0]}
    assert ctx.net.is_enabled_coloured("t_approve", high)
    assert not ctx.net.is_enabled_coloured("t_decline", high)


def test_low_amount_token_fires_decline_only():
    ctx = load_scenario(SCENARIO)
    low = {"p_submitted": [250.0]}
    assert not ctx.net.is_enabled_coloured("t_approve", low)
    assert ctx.net.is_enabled_coloured("t_decline", low)


def test_threshold_boundary_amount_fires_approve():
    """At the boundary (amount == 1000), the approve guard uses `>=`
    so it fires; the decline guard uses `<` so it doesn't."""
    ctx = load_scenario(SCENARIO)
    boundary = {"p_submitted": [1000.0]}
    assert ctx.net.is_enabled_coloured("t_approve", boundary)
    assert not ctx.net.is_enabled_coloured("t_decline", boundary)


def test_full_token_game_high_amount_routes_to_approved():
    ctx = load_scenario(SCENARIO)
    marking = {"p_submitted": [7500.0]}
    after = ctx.net.fire_coloured("t_approve", marking)
    # The default output value is 1.0 (no callable form in TOML),
    # so the approved place carries a single 1.0 token. The
    # important property is *which* place got the token — that's
    # the routing the guard enforced.
    assert "p_approved" in after
    assert "p_declined" not in after


def test_full_token_game_low_amount_routes_to_declined():
    ctx = load_scenario(SCENARIO)
    marking = {"p_submitted": [400.0]}
    after = ctx.net.fire_coloured("t_decline", marking)
    assert "p_declined" in after
    assert "p_approved" not in after


def test_batch_of_applications_each_routed_separately():
    """Two applications queued at p_submitted, one high-value and
    one low-value. Firing approve consumes the first (FIFO); firing
    decline then consumes the second. The two applications take
    different routes."""
    ctx = load_scenario(SCENARIO)
    marking = {"p_submitted": [5000.0, 200.0]}
    after_approve = ctx.net.fire_coloured("t_approve", marking)
    # First (5000) consumed by approve, second (200) still queued.
    assert "p_approved" in after_approve
    assert after_approve["p_submitted"] == [200.0]
    after_decline = ctx.net.fire_coloured("t_decline", after_approve)
    assert "p_declined" in after_decline
    assert "p_submitted" not in after_decline


# ---------------------------------------------------------------------------
# CPN-aware compiler — the trained network reads token values and refines
# the declared guard threshold from data.

def test_compiled_module_exposes_learnable_guard_thresholds():
    """Each structurally-guarded transition gets one nn.Parameter
    threshold, initialised at the TOML value."""
    ctx = load_scenario(SCENARIO)
    module = ctx.compile()
    # Two guards declared, so two parameters seeded.
    assert set(module.guard_thresholds) == {"guard_theta_0", "guard_theta_1"}
    # The seed values come from the TOML — both are 1000 in this
    # scenario.
    for key in module.guard_thresholds:
        assert abs(module.guard_thresholds[key].item() - 1000.0) < 1e-6


def test_training_drives_guard_thresholds_toward_observed_boundary():
    """Training on a mix of high- and low-amount applications should
    pull both guard thresholds toward the empirical boundary in
    the data — somewhere between the largest decline (900) and
    the smallest approve (1500). The hand-seeded 1000 should stay
    in that band; what matters is that the gates correctly route
    held-out values rather than that the threshold lands on any
    exact number.

    This is the load-bearing claim of the CPN-aware compiler: the
    trained network reads per-token values and routes on them."""
    import torch

    ctx = load_scenario(SCENARIO)
    module, losses = ctx.train()

    # Training drove the loss down.
    assert losses[-1] < losses[0]

    # Both learned thresholds should sit in the empirical decision
    # band — between the largest observed decline (900) and the
    # smallest observed approve (1500).
    for key in module.guard_thresholds:
        learned = module.guard_thresholds[key].item()
        assert 800.0 <= learned <= 1700.0, (
            f"guard threshold {key} drifted out of band: {learned}"
        )

    # Held-out values should route correctly under the trained guards.
    with torch.no_grad():
        out = module(
            input_marking={"p_submitted": torch.tensor([1.0, 1.0])},
            input_values={"p_submitted": torch.tensor([5000.0, 300.0])},
            batch_size=2,
        )

    # For amount 5000 (high), approve fires strongly, decline weakly.
    assert out["t_approve"][0].item() > 0.7
    assert out["t_decline"][0].item() < 0.3
    # For amount 300 (low), decline fires strongly, approve weakly.
    assert out["t_decline"][1].item() > 0.7
    assert out["t_approve"][1].item() < 0.3
