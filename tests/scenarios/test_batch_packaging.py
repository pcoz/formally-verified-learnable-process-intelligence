"""End-to-end test for the batch packaging scenario.

Showcases the Phase 9 multi-token-markings feature alongside the
existing training / interpretability / anomaly pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    XESEvent,
    XESTrace,
    load_scenario,
)


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "batch_packaging"
    / "scenario.toml"
)


def test_batch_packaging_scenario_loads_with_multi_token_arc():
    ctx = load_scenario(SCENARIO)
    assert ctx.net.validate() == []
    assert ctx.net.weight("p_filled", "t_crate") == 6
    assert ctx.net.weight("t_crate", "p_crate") == 1


def test_crate_transition_waits_for_six_bottles():
    """t_crate has input weight 6 — it must NOT be enabled until at
    least 6 tokens accumulate at p_filled. This is the multi-token
    behaviour that 1-bounded nets could not express."""
    ctx = load_scenario(SCENARIO)
    net = ctx.net
    marking = dict(net.initial_marking)
    for count in range(5):
        marking = net.fire("t_fill", marking)
        assert not net.is_enabled("t_crate", marking), (
            f"t_crate should NOT be enabled with {count + 1} bottles"
        )
    marking = net.fire("t_fill", marking)
    assert net.is_enabled("t_crate", marking)
    assert marking["p_filled"] == 6


def test_crate_firing_consumes_exactly_six_bottles():
    ctx = load_scenario(SCENARIO)
    net = ctx.net
    marking = dict(net.initial_marking)
    for _ in range(8):
        marking = net.fire("t_fill", marking)
    assert marking["p_filled"] == 8

    marking = net.fire("t_crate", marking)
    assert marking["p_filled"] == 2
    assert marking["p_crate"] == 1


def test_full_packaging_run_completes():
    """Walk a complete production run: fill six bottles, crate them,
    inspect, ship."""
    ctx = load_scenario(SCENARIO)
    net = ctx.net
    marking = dict(net.initial_marking)
    for _ in range(6):
        marking = net.fire("t_fill", marking)
    marking = net.fire("t_crate", marking)
    marking = net.fire("t_inspect", marking)
    marking = net.fire("t_ship", marking)
    assert marking["p_shipped"] == 1


def test_training_recovers_quality_routing():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    with torch.no_grad():
        good = module(input_marking={"p_inspected": torch.tensor([0.95])})
        bad = module(input_marking={"p_inspected": torch.tensor([0.05])})
    assert good["t_ship"].item() > good["t_reject"].item()
    assert bad["t_reject"].item() > bad["t_ship"].item()


def test_anomaly_detected_when_low_quality_shipped():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    good = XESTrace(
        attributes={"quality_score": "0.93"},
        events=[
            *[XESEvent(name="fill bottle")] * 6,
            XESEvent(name="crate 6 bottles"),
            XESEvent(name="inspect crate"),
            XESEvent(name="ship crate"),
        ],
    )
    misshipped = XESTrace(
        attributes={"quality_score": "0.05"},
        events=[
            *[XESEvent(name="fill bottle")] * 6,
            XESEvent(name="crate 6 bottles"),
            XESEvent(name="inspect crate"),
            XESEvent(name="ship crate"),
        ],
    )
    ok = sum(ctx.anomaly_score(module, good).values())
    bad = sum(ctx.anomaly_score(module, misshipped).values())
    assert bad > ok + 0.3
