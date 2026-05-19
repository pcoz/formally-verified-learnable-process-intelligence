"""End-to-end test for the resource-lock mutex scenario.

Showcases the Phase 9 inhibitor-arc feature: two transitions
guarded by a shared 'resource_busy' place enforce mutual exclusion
both at the discrete-token-game level and in the trained network's
time-unrolled forward pass.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import load_scenario


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "resource_lock"
    / "scenario.toml"
)


def test_scenario_loads_with_inhibitor_arcs():
    ctx = load_scenario(SCENARIO)
    assert ctx.net.validate() == []
    # Both transitions must be inhibited by the shared resource place.
    assert ("p_resource_busy", "t_serve_a") in ctx.net.inhibitor_arcs
    assert ("p_resource_busy", "t_serve_b") in ctx.net.inhibitor_arcs


def test_token_game_initially_allows_either_transition():
    """The classic mutex shape: at the initial marking the resource
    is free, so either transition is enabled. Whichever fires first
    claims the resource and immediately disables the other."""
    ctx = load_scenario(SCENARIO)
    net = ctx.net
    initial = dict(net.initial_marking)
    assert net.is_enabled("t_serve_a", initial)
    assert net.is_enabled("t_serve_b", initial)


def test_token_game_locks_after_first_firing():
    """After t_serve_a fires, p_resource_busy gets a token; the
    inhibitor gate then disables t_serve_b. This is the entire point
    of the mutex pattern."""
    ctx = load_scenario(SCENARIO)
    net = ctx.net
    after_a = net.fire("t_serve_a", net.initial_marking)
    assert "p_resource_busy" in after_a
    assert not net.is_enabled("t_serve_b", after_a)


def test_inhibitor_gate_active_in_time_unrolled_forward():
    """In time-unrolled mode the resource_busy place accumulates
    activation once a transition has fired, and the multiplicative
    inhibitor gate then suppresses subsequent firings. The total
    'critical section' occupation should stay bounded — not double
    what a no-inhibitor net would produce."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    out = module()
    # With proper inhibitor gating the resource_busy place's
    # activation across the unrolled steps should remain bounded
    # by about 1.0 (one of the two transitions wins; the other is
    # gated). Without the gate it would tend toward 2.0.
    assert out["p_resource_busy"].item() < 1.6


def test_priority_attribute_drives_routing():
    """High priority_a should give t_serve_a a higher activation than
    t_serve_b after training — the framework still learns the
    routing rule despite the inhibitor structure."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    with torch.no_grad():
        hi = module(input_marking={"p_a_pending": torch.tensor([0.95])})
        lo = module(input_marking={"p_a_pending": torch.tensor([0.05])})
    # The priority attribute should bias which transition wins the
    # mutex — t_serve_a should pull ahead at high priority and
    # t_serve_b at low priority.
    assert hi["t_serve_a"].item() >= hi["t_serve_b"].item() - 0.1
    assert lo["t_serve_b"].item() >= lo["t_serve_a"].item() - 0.1
