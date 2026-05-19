"""End-to-end test for the manufacturing cell scenario."""
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
    / "manufacturing_cell"
    / "scenario.toml"
)


def test_manufacturing_cell_loads():
    ctx = load_scenario(SCENARIO)
    assert ctx.net.validate() == []
    assert ctx.training.seed == 0


def test_manufacturing_cell_learns_quality_routing():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    with torch.no_grad():
        high = module(input_marking={"p_inspection_ready": torch.tensor([0.95])})
        low = module(input_marking={"p_inspection_ready": torch.tensor([0.05])})
    assert high["t_ship"].item() > high["t_rework"].item()
    assert low["t_rework"].item() > low["t_ship"].item()


def test_manufacturing_cell_distils_ship_rule():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    rules = ctx.extract_rules(module)
    ship_rule = next(
        (r for r in rules["xor"] if "ship part" in (r.label_above, r.label_below)),
        None,
    )
    assert ship_rule is not None
    assert ship_rule.label_above == "ship part"
    assert ship_rule.label_below == "rework part"


def test_manufacturing_cell_anomaly_on_low_quality_ship():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    legitimate = XESTrace(
        attributes={"quality_score": "0.92"},
        events=[
            XESEvent(name="station 1 processing"),
            XESEvent(name="station 2 processing"),
            XESEvent(name="inspect part"),
            XESEvent(name="ship part"),
        ],
    )
    misshipped = XESTrace(
        attributes={"quality_score": "0.05"},
        events=[
            XESEvent(name="station 1 processing"),
            XESEvent(name="station 2 processing"),
            XESEvent(name="inspect part"),
            XESEvent(name="ship part"),
        ],
    )
    ok = ctx.anomaly_score(module, legitimate)
    bad = ctx.anomaly_score(module, misshipped)
    assert sum(bad.values()) > sum(ok.values()) + 0.3
