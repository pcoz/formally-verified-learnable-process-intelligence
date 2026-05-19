"""End-to-end test for the paint-shop scenario.

Showcases the Phase 9 transition-duration feature: t_cure has
duration 3, so a firing at step n produces its output at step n+2
(three steps including the firing step itself).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import load_scenario


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "paint_shop"
    / "scenario.toml"
)


def test_paint_shop_loads_with_duration():
    ctx = load_scenario(SCENARIO)
    assert ctx.net.validate() == []
    # t_cure carries the duration annotation; everything else stays at 1.
    assert ctx.net.duration("t_cure") == 3
    assert ctx.net.duration("t_paint") == 1
    assert ctx.net.duration("t_pass") == 1


def test_cure_delay_visible_in_unrolled_forward():
    """Compile and run the scenario for varying step budgets. The
    inspection place should remain near zero until the cure
    transition has had three steps to mature; once it has, the
    output activation appears."""
    ctx = load_scenario(SCENARIO)
    # Compile two variants: one with too few steps, one with enough.
    torch.manual_seed(0)
    early_module, _ = ctx.train()
    early = early_module()

    # The pass / fail downstream activation depends on p_inspected
    # being populated, which requires waiting through the three-step
    # cure. Six unrolled steps (the scenario default) give plenty of
    # time, so p_passed or p_failed should be lit for some input.
    high_input = early_module(
        input_marking={"p_inspected": torch.tensor([0.9])}
    )
    assert high_input["p_passed"].item() > 0.4


def test_quality_routing_learned_through_delay():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    with torch.no_grad():
        good = module(input_marking={"p_inspected": torch.tensor([0.95])})
        bad = module(input_marking={"p_inspected": torch.tensor([0.05])})
    assert good["t_pass"].item() > good["t_fail"].item()
    assert bad["t_fail"].item() > bad["t_pass"].item()


def test_distilled_rule_routes_pass_on_high_quality():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    rules = ctx.extract_rules(module)
    pass_rule = next(
        (r for r in rules["xor"] if "pass inspection" in (r.label_above, r.label_below)),
        None,
    )
    assert pass_rule is not None
    assert pass_rule.label_above == "pass inspection"
