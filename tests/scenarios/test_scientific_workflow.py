"""End-to-end test for the PCR scientific workflow scenario."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    XESEvent,
    XESTrace,
    drop_event,
    load_scenario,
)


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "scientific_workflow"
    / "scenario.toml"
)


def test_pcr_loads_and_validates():
    ctx = load_scenario(SCENARIO)
    assert ctx.net.validate() == []


def test_pcr_learns_quality_gate():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    with torch.no_grad():
        good = module(input_marking={"p_measured": torch.tensor([0.95])})
        bad = module(input_marking={"p_measured": torch.tensor([0.05])})
    assert good["t_accept"].item() > good["t_reject"].item()
    assert bad["t_reject"].item() > bad["t_accept"].item()


def test_pcr_extracts_accept_threshold_rule():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    rules = ctx.extract_rules(module)
    accept_rule = next(
        (r for r in rules["xor"] if "accept sample" in (r.label_above, r.label_below)),
        None,
    )
    assert accept_rule is not None
    assert accept_rule.label_above == "accept sample"


def test_pcr_deviation_flags_skipped_denaturation():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    normal = XESTrace(
        attributes={"amplification_quality": "0.93"},
        events=[
            XESEvent(name="denature DNA"),
            XESEvent(name="anneal primers"),
            XESEvent(name="extend DNA"),
            XESEvent(name="measure amplification"),
            XESEvent(name="accept sample"),
        ],
    )
    skipped = drop_event(normal, index=0)
    normal_score = sum(
        ctx.anomaly_score(module, normal).values()
    )
    skipped_score = sum(
        ctx.anomaly_score(module, skipped).values()
    )
    assert skipped_score > normal_score + 0.3
