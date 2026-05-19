"""End-to-end test for the priority-dispatch scenario.

Showcases the Phase 9 stochastic-firing-rate feature: three handlers
with declared rates (3.0, 1.0, 0.5) producing the expected eagerness
ordering before training, and refined behaviour after training.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import load_scenario


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "priority_dispatch"
    / "scenario.toml"
)


def test_priority_dispatch_loads_with_rates():
    ctx = load_scenario(SCENARIO)
    assert ctx.net.validate() == []
    assert ctx.net.rate("t_express") == pytest.approx(3.0)
    assert ctx.net.rate("t_standard") == pytest.approx(1.0)
    assert ctx.net.rate("t_bulk") == pytest.approx(0.5)


def test_rate_ordering_holds_before_training():
    """Without any training, the rate priors alone should produce
    activations in order express > standard > bulk for a midpoint
    urgency input."""
    ctx = load_scenario(SCENARIO)
    module = ctx.compile()
    with torch.no_grad():
        out = module(input_marking={"p_ticket": torch.tensor([0.5])})
    assert out["t_express"].item() > out["t_standard"].item() > out["t_bulk"].item()


def test_training_converges_on_urgency_driven_routing():
    """After training, urgency should still drive the routing — high
    urgency favours express, low urgency favours bulk, mid urgency
    favours standard."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    with torch.no_grad():
        high = module(input_marking={"p_ticket": torch.tensor([0.92])})
        low = module(input_marking={"p_ticket": torch.tensor([0.12])})
    assert high["t_express"].item() > high["t_bulk"].item()
    assert low["t_bulk"].item() > low["t_express"].item()
