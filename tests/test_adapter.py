"""Tests for the config-driven adapter framework.

The adapter is the bridge between "data lives somewhere on disk" and
"PetriNetModule.train()". Each scenario is a TOML config plus a data
file (BPMN, XES, or inline declarations) and goes through the same
pipeline. These tests pin the adapter contract: parsing, attribute
mapping, the train/extract/score methods, and the round-trip
equivalence with hand-coded scenario tests.
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


EXAMPLES = Path(__file__).parent.parent / "examples"


def test_load_biological_cascade_from_config():
    ctx = load_scenario(EXAMPLES / "biological_signalling" / "scenario.toml")
    assert ctx.name == "biological_signalling_cascade"
    assert "p_signal" in ctx.net.places
    assert "t_phosphorylate" in ctx.net.transitions
    assert len(ctx.traces) == 16
    assert ctx.training.seed == 0


def test_attribute_to_marking_resolves_per_trace_attributes():
    ctx = load_scenario(EXAMPLES / "biological_signalling" / "scenario.toml")
    trace = XESTrace(
        attributes={"signal_strength": "0.42"},
        events=[XESEvent(name="phosphorylate kinase")],
    )
    marking = ctx.attribute_to_marking(trace)
    assert marking == {"p_signal": pytest.approx(0.42)}


def test_attribute_to_marking_raises_for_missing_attribute():
    ctx = load_scenario(EXAMPLES / "biological_signalling" / "scenario.toml")
    trace = XESTrace(attributes={}, events=[])
    with pytest.raises(KeyError, match="signal_strength"):
        ctx.attribute_to_marking(trace)


def test_scenario_train_recovers_strength_dependent_routing():
    """End-to-end: the config alone is enough to drive training,
    and the trained module recovers the same fast/slow routing as
    the hand-coded version in test_non_bpmn_substrate."""
    ctx = load_scenario(EXAMPLES / "biological_signalling" / "scenario.toml")
    module, losses = ctx.train()
    assert losses[-1] < losses[0]

    with torch.no_grad():
        strong = module(input_marking={"p_signal": torch.tensor([0.95])})
        weak = module(input_marking={"p_signal": torch.tensor([0.05])})
    assert strong["t_fast_pathway"].item() > strong["t_slow_pathway"].item()
    assert weak["t_slow_pathway"].item() > weak["t_fast_pathway"].item()


def test_scenario_extract_rules_returns_xor_rules_when_configured():
    ctx = load_scenario(EXAMPLES / "biological_signalling" / "scenario.toml")
    module, _ = ctx.train()
    rules = ctx.extract_rules(module)
    assert "xor" in rules
    assert "and_join" not in rules
    routing_rules = [r for r in rules["xor"] if r.input_place == "p_kinase_active"]
    assert len(routing_rules) == 1
    assert {routing_rules[0].label_above, routing_rules[0].label_below} == {
        "fast pathway",
        "slow pathway",
    }


def test_constant_input_marking_entry():
    """A place can be pinned to a constant instead of an attribute,
    useful for tasks that don't depend on per-trace data."""
    ctx = load_scenario(EXAMPLES / "biological_signalling" / "scenario.toml")
    ctx.input_marking_spec["p_signal"] = {"constant": 0.5}
    trace = XESTrace(attributes={}, events=[])
    assert ctx.attribute_to_marking(trace) == {"p_signal": 0.5}


def test_bpmn_file_net_source_loads_from_disk():
    """If a config points at a BPMN file the adapter must round-trip
    through parse_bpmn. Verify by pointing at one of the test
    fixtures."""
    fixtures = Path(__file__).parent / "fixtures"
    cfg = tmp = Path(__file__).parent / "fixtures" / "_bpmn_adapter_test.toml"
    tmp.write_text(f"""
[scenario]
name = "bpmn_from_disk"

[net]
source = "bpmn_file"
path = "{(fixtures / 'simple_sequence.bpmn').as_posix()}"

[training.input_marking]
p_f1 = {{ constant = 1.0 }}

[training]
steps = 0
seed = 0
""")
    try:
        ctx = load_scenario(cfg)
        assert "p_f1" in ctx.net.places
        assert "t_do_work" in ctx.net.transitions
    finally:
        tmp.unlink()


def test_invalid_net_source_raises():
    fixtures = Path(__file__).parent / "fixtures"
    cfg = fixtures / "_bad_adapter_test.toml"
    cfg.write_text("""
[scenario]
name = "bad"
[net]
source = "nope"
""")
    try:
        with pytest.raises(ValueError, match="net.source"):
            load_scenario(cfg)
    finally:
        cfg.unlink()
