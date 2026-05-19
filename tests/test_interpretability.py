"""Tests for Phase 8 interpretability — distilling trained weights back
into readable rules and pinning anomalies to named BPMN elements."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    PetriNet,
    PetriNetModule,
    XESEvent,
    XESTrace,
    explain_anomaly,
    extract_and_join_rule,
    extract_and_join_rules,
    extract_routing_partitions,
    extract_routing_rules,
    extract_xor_partition,
    extract_xor_rule,
    find_and_join_transitions,
    find_xor_groups,
    parse_bpmn,
    parse_xes,
    swap_event_labels,
    train_on_traces,
)


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# XOR-shape detection
# ---------------------------------------------------------------------------


def test_find_xor_groups_identifies_the_split_in_xor_branch():
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    groups = find_xor_groups(net)

    split_groups = [g for g in groups if g[0] == "p_f0"]
    assert len(split_groups) == 1
    place, transitions = split_groups[0]
    assert place == "p_f0"
    assert len(transitions) == 2


def test_find_xor_groups_excludes_and_join_pattern():
    """The AND-branch fixture has a join with two input arcs but only
    one consuming transition, so it must NOT register as an XOR
    group."""
    net = parse_bpmn(FIXTURES / "and_branch.bpmn")
    groups = find_xor_groups(net)
    for place, transitions in groups:
        assert len(transitions) >= 2
        for t in transitions:
            assert len(net.preset(t)) == 1


def test_find_xor_groups_on_sequential_net_returns_empty():
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    assert find_xor_groups(net) == []


# ---------------------------------------------------------------------------
# XOR rule extraction on a trained model
# ---------------------------------------------------------------------------


def _xor_marking(trace):
    return {"p_f0": float(trace.attributes["risk_score"])}


def _trained_xor_module():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    module = PetriNetModule(net)
    traces = parse_xes(FIXTURES / "xor_log.xes")
    train_on_traces(
        module,
        traces,
        attribute_to_marking=_xor_marking,
        steps=1500,
        lr=0.1,
    )
    return module


def test_extract_xor_rule_recovers_crossover_near_05():
    """The training data splits at risk_score 0.5 (high → Path A,
    low → Path B). The learned routing should recover a crossover
    threshold close to 0.5."""
    module = _trained_xor_module()
    groups = find_xor_groups(module.net)
    split_place = "p_f0"
    transitions = next(t for p, t in groups if p == split_place)

    rule = extract_xor_rule(module, split_place, transitions[0], transitions[1])
    assert 0.3 < rule.crossover < 0.7
    assert rule.confidence > 0.1
    assert rule.label_above in {"Path A", "Path B"}
    assert rule.label_below in {"Path A", "Path B"}
    assert rule.label_above != rule.label_below


def test_extract_xor_rule_high_input_goes_to_path_a():
    """Sanity-check the direction. With the training data we used,
    high risk_score routes to Path A."""
    module = _trained_xor_module()
    groups = find_xor_groups(module.net)
    split_place = "p_f0"
    transitions = next(t for p, t in groups if p == split_place)
    rule = extract_xor_rule(module, split_place, transitions[0], transitions[1])
    assert rule.label_above == "Path A"
    assert rule.label_below == "Path B"


def test_extract_routing_rules_finds_all_xor_pairs():
    """The XOR fixture has one XOR-split and one XOR-join, but the
    join's two transitions go to the same output place (shared output,
    not shared input) — only the split is a "binary routing rule"."""
    module = _trained_xor_module()
    rules = extract_routing_rules(module)
    assert len(rules) >= 1
    crossovers = [r.crossover for r in rules if r.input_place == "p_f0"]
    assert len(crossovers) == 1


def test_xor_rule_description_is_readable():
    module = _trained_xor_module()
    rules = extract_routing_rules(module)
    descriptions = [r.description() for r in rules]
    main = next(d for d, r in zip(descriptions, rules) if r.input_place == "p_f0")
    assert "'p_f0'" in main
    assert "Path A" in main or "Path B" in main


# ---------------------------------------------------------------------------
# N-way XOR rule extraction
# ---------------------------------------------------------------------------


def _three_way_xor_net() -> PetriNet:
    net = PetriNet()
    net.add_place("p_in", tokens=1)
    for branch in ("low", "mid", "high"):
        net.add_place(f"p_{branch}")
        net.add_transition(f"t_{branch}", label=branch.title())
        net.add_arc("p_in", f"t_{branch}")
        net.add_arc(f"t_{branch}", f"p_{branch}")
    return net


def _train_three_way_xor(steps: int = 2500, lr: float = 0.1) -> PetriNetModule:
    torch.manual_seed(0)
    module = PetriNetModule(_three_way_xor_net(), sharpness=2.0)
    inputs = torch.linspace(0.05, 0.95, 90)
    target_low = (inputs < 1.0 / 3.0).float()
    target_mid = ((inputs >= 1.0 / 3.0) & (inputs < 2.0 / 3.0)).float()
    target_high = (inputs >= 2.0 / 3.0).float()

    opt = torch.optim.Adam(module.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        out = module(input_marking={"p_in": inputs})
        loss = (
            (out["t_low"] - target_low) ** 2
            + (out["t_mid"] - target_mid) ** 2
            + (out["t_high"] - target_high) ** 2
        ).mean()
        loss.backward()
        opt.step()
    return module


def test_extract_xor_partition_recovers_three_regions():
    module = _train_three_way_xor()
    partition = extract_xor_partition(
        module, "p_in", ["t_low", "t_mid", "t_high"]
    )
    assert len(partition.regions) == 3
    labels = [r.label for r in partition.regions]
    assert labels == ["Low", "Mid", "High"]
    boundaries = [r.upper for r in partition.regions[:-1]]
    assert 0.2 < boundaries[0] < 0.45
    assert 0.55 < boundaries[1] < 0.8


def test_extract_xor_partition_regions_cover_input_range_contiguously():
    module = _train_three_way_xor()
    partition = extract_xor_partition(
        module, "p_in", ["t_low", "t_mid", "t_high"]
    )
    assert partition.regions[0].lower == 0.0
    assert partition.regions[-1].upper == 1.0
    for prev, nxt in zip(partition.regions, partition.regions[1:]):
        assert prev.upper == pytest.approx(nxt.lower)


def test_extract_routing_partitions_handles_binary_and_nway_uniformly():
    """The binary case from the XOR fixture must still produce a
    2-region partition with the right labels."""
    module = _trained_xor_module()
    partitions = extract_routing_partitions(module)
    p_f0_partitions = [p for p in partitions if p.input_place == "p_f0"]
    assert len(p_f0_partitions) == 1
    regions = p_f0_partitions[0].regions
    assert len(regions) == 2
    labels = {r.label for r in regions}
    assert labels == {"Path A", "Path B"}


def test_extract_xor_partition_rejects_n_below_2():
    module = _train_three_way_xor()
    with pytest.raises(ValueError, match="at least 2"):
        extract_xor_partition(module, "p_in", ["t_low"])


def test_xor_partition_description_lists_each_region():
    module = _train_three_way_xor()
    partition = extract_xor_partition(
        module, "p_in", ["t_low", "t_mid", "t_high"]
    )
    desc = partition.description()
    assert "'Low'" in desc
    assert "'Mid'" in desc
    assert "'High'" in desc


# ---------------------------------------------------------------------------
# AND-join rule distillation
# ---------------------------------------------------------------------------


def _and_join_net() -> PetriNet:
    net = PetriNet()
    for p in ("p_A", "p_B", "p_out"):
        net.add_place(p, label=p)
    net.add_transition("t_merge", label="Merge")
    net.add_arc("p_A", "t_merge")
    net.add_arc("p_B", "t_merge")
    net.add_arc("t_merge", "p_out")
    return net


def _trained_and_join(*, sharpness: float = 4.0):
    torch.manual_seed(0)
    module = PetriNetModule(_and_join_net(), sharpness=sharpness)
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    target = torch.tensor([0.0, 0.0, 0.0, 1.0])
    opt = torch.optim.Adam(module.parameters(), lr=0.1)
    for _ in range(2000):
        opt.zero_grad()
        loss = (
            (module(input_marking={"p_A": a, "p_B": b})["p_out"] - target) ** 2
        ).mean()
        loss.backward()
        opt.step()
    return module


def test_find_and_join_transitions_identifies_multi_input_transitions():
    net = _and_join_net()
    assert find_and_join_transitions(net) == ["t_merge"]


def test_find_and_join_excludes_single_input_transitions():
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    assert find_and_join_transitions(net) == []


def test_extract_and_join_rule_reports_all_inputs_required():
    """After training the 2-input AND-join on its truth table the
    learned weights and threshold should correspond to "all inputs
    must be active" — the canonical AND. The summary should say so
    in those words."""
    module = _trained_and_join()
    rule = extract_and_join_rule(module, "t_merge")
    assert "all" in rule.summary.lower()
    assert "2" in rule.summary
    labels = [lbl for (_, lbl, _) in rule.inputs]
    assert set(labels) == {"p_A", "p_B"}


def test_extract_and_join_rules_walks_whole_net():
    module = _trained_and_join()
    rules = extract_and_join_rules(module)
    assert len(rules) == 1
    assert rules[0].transition == "t_merge"


def test_extract_and_join_rule_rejects_single_input_transition():
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    module = PetriNetModule(net)
    with pytest.raises(ValueError, match="at least 2"):
        extract_and_join_rule(module, "t_do_work")


def test_and_join_rule_description_includes_label():
    module = _trained_and_join()
    rule = extract_and_join_rule(module, "t_merge")
    desc = rule.description()
    assert "Merge" in desc


# ---------------------------------------------------------------------------
# Anomaly explanations
# ---------------------------------------------------------------------------


def test_explain_anomaly_names_diverging_transitions_on_branch_flip():
    """A branch-flipped trace produces residuals on the two routing
    transitions. The explanation must mention Path A and Path B by
    label so the analyst can read which BPMN tasks diverged."""
    module = _trained_xor_module()

    anomalous = XESTrace(
        attributes={"risk_score": "0.95"},
        events=[XESEvent(name="Path B")],
    )

    explanation = explain_anomaly(
        module, anomalous, attribute_to_marking=_xor_marking, threshold=0.1
    )
    assert "Path A" in explanation
    assert "Path B" in explanation
    assert "residual" in explanation


def test_explain_anomaly_silent_on_in_distribution_trace():
    module = _trained_xor_module()
    normal = XESTrace(
        attributes={"risk_score": "0.95"},
        events=[XESEvent(name="Path A")],
    )
    explanation = explain_anomaly(
        module, normal, attribute_to_marking=_xor_marking, threshold=0.4
    )
    assert "No significant anomalies" in explanation


def test_explain_anomaly_uses_bpmn_labels_not_internal_ids():
    """The §7.2 interpretability claim is "interpretable at the
    granularity of named BPMN elements". The explanation must not
    leak transition IDs like 't_xor_split_0' to the reader."""
    module = _trained_xor_module()
    anomalous = XESTrace(
        attributes={"risk_score": "0.95"},
        events=[XESEvent(name="Path B")],
    )
    explanation = explain_anomaly(
        module, anomalous, attribute_to_marking=_xor_marking, threshold=0.1
    )
    assert "t_xor_split" not in explanation
    assert "t_taskA" not in explanation
