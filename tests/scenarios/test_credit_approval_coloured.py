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
