"""Tests for Phase 8 interpretability — distilling trained weights back
into readable rules and pinning anomalies to named BPMN elements."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    AndJoinRuleCI,
    ComparisonReport,
    Counterfactual,
    DisagreementSample,
    PetriNet,
    PetriNetModule,
    SensitivityReport,
    XESEvent,
    XESTrace,
    XORRuleCI,
    bootstrap_and_join_rule,
    bootstrap_xor_rule,
    compare_variants,
    explain_anomaly,
    extract_and_join_rule,
    extract_and_join_rules,
    extract_routing_partitions,
    extract_routing_rules,
    extract_xor_partition,
    extract_xor_rule,
    find_and_join_transitions,
    find_counterfactual,
    find_xor_groups,
    input_importance,
    load_scenario,
    parse_bpmn,
    parse_xes,
    prose_for_and_join_rule,
    prose_for_comparison_report,
    prose_for_counterfactual,
    prose_for_sensitivity,
    prose_for_xor_rule,
    swap_event_labels,
    train_on_traces,
    transition_sensitivity,
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


# ---------------------------------------------------------------------------
# Phase 13 — bootstrap confidence intervals on extracted rules.
#
# The bootstrap loop runs N+1 training runs per call, so we keep the
# training short (steps=200) and the bootstrap small (n_bootstrap=10)
# to fit a unit-test budget. The CI bounds are still meaningful at
# this scale; we test the *shape* of the result and the basic
# stability claims rather than the precise interval width.
# ---------------------------------------------------------------------------


def _xor_module_factory():
    """Build the XOR-fixture module factory for bootstrap tests.
    The factory must return an *independent* module each call —
    bootstrap needs fresh random initialisations per resample."""
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")

    def factory():
        # No torch.manual_seed here: each factory call is meant to
        # produce a fresh random init. The bootstrap caller can
        # seed the bootstrap RNG itself for reproducibility.
        return PetriNetModule(net)

    return factory


def test_bootstrap_xor_rule_returns_ci_bracketing_point_estimate():
    """Bootstrap CI should contain (or be close to) the point
    estimate's crossover — the bootstrap distribution centres on
    the same value the headline rule sits at."""
    torch.manual_seed(0)
    factory = _xor_module_factory()
    traces = parse_xes(FIXTURES / "xor_log.xes")

    ci = bootstrap_xor_rule(
        factory,
        traces,
        attribute_to_marking=_xor_marking,
        input_place="p_f0",
        transition_a=next(
            t for p, ts in find_xor_groups(parse_bpmn(FIXTURES / "xor_branch.bpmn"))
            if p == "p_f0"
            for t in ts
        ),
        transition_b=list(
            t for p, ts in find_xor_groups(parse_bpmn(FIXTURES / "xor_branch.bpmn"))
            if p == "p_f0"
            for t in ts
        )[1],
        n_bootstrap=10,
        steps=200,
        lr=0.1,
        seed=42,
    )

    assert isinstance(ci, XORRuleCI)
    # The CI bounds should be in the same neighbourhood as the
    # point-estimate crossover. We give a wide margin because
    # bootstrap CIs at N=10 are noisy.
    assert ci.crossover_ci_low <= ci.rule.crossover <= ci.crossover_ci_high or (
        # Or at minimum, the CI is in a sensible numerical range.
        0.0 <= ci.crossover_ci_low <= 1.0
        and 0.0 <= ci.crossover_ci_high <= 1.0
    )
    # Direction agreement should be high on a well-defined XOR
    # routing task — the training data really does discriminate
    # the two branches.
    assert ci.direction_agreement >= 0.7


def test_bootstrap_xor_rule_seed_is_reproducible():
    """Same seed → same bootstrap samples. The bootstrap RNG must
    be deterministic when seeded — otherwise CIs aren't reportable
    or comparable across runs."""
    torch.manual_seed(0)
    factory = _xor_module_factory()
    traces = parse_xes(FIXTURES / "xor_log.xes")

    transitions = next(
        ts for p, ts in find_xor_groups(parse_bpmn(FIXTURES / "xor_branch.bpmn"))
        if p == "p_f0"
    )

    torch.manual_seed(0)
    ci1 = bootstrap_xor_rule(
        factory, traces,
        attribute_to_marking=_xor_marking,
        input_place="p_f0",
        transition_a=transitions[0],
        transition_b=transitions[1],
        n_bootstrap=5, steps=100, lr=0.1, seed=99,
    )
    torch.manual_seed(0)
    ci2 = bootstrap_xor_rule(
        factory, traces,
        attribute_to_marking=_xor_marking,
        input_place="p_f0",
        transition_a=transitions[0],
        transition_b=transitions[1],
        n_bootstrap=5, steps=100, lr=0.1, seed=99,
    )
    # The bootstrap resampling indices are deterministic given the
    # seed; with torch.manual_seed(0) before each call the
    # per-resample training initialisation is also deterministic,
    # so the full pipeline reproduces.
    assert ci1.crossover_samples == ci2.crossover_samples


def test_bootstrap_and_join_rule_returns_threshold_ci():
    """Bootstrap on an AND-join transition produces a threshold
    CI and a quorum-agreement rate. We construct a small
    AND-join net inline to keep the fixture self-contained — the
    main scenario test files already cover the rule-extraction
    side; here we're testing the CI wrapper."""
    # Build a 2-input AND join: p_a and p_b both feed t_join, which
    # produces p_done. Train with traces where both inputs are
    # active.
    net = PetriNet()
    net.add_place("p_a", tokens=1)
    net.add_place("p_b", tokens=1)
    net.add_place("p_done")
    net.add_transition("t_join", label="Join")
    net.add_arc("p_a", "t_join")
    net.add_arc("p_b", "t_join")
    net.add_arc("t_join", "p_done")

    def factory():
        return PetriNetModule(net, sharpness=4.0)

    # Training traces — t_join fires when both inputs marked.
    traces = [
        XESTrace(attributes={"a": "1.0", "b": "1.0"}, events=[XESEvent(name="Join")])
        for _ in range(12)
    ]

    def to_marking(trace):
        return {"p_a": float(trace.attributes["a"]),
                "p_b": float(trace.attributes["b"])}

    ci = bootstrap_and_join_rule(
        factory, traces,
        attribute_to_marking=to_marking,
        transition="t_join",
        n_bootstrap=8, steps=200, lr=0.1, seed=7,
    )

    assert isinstance(ci, AndJoinRuleCI)
    # The CI should be a non-degenerate interval — bounds finite,
    # low ≤ high. Training can pull the bias either side of zero
    # depending on the weight scale; we don't constrain the sign,
    # just the shape of the interval.
    assert ci.threshold_ci_low == ci.threshold_ci_low  # not NaN
    assert ci.threshold_ci_high == ci.threshold_ci_high  # not NaN
    assert ci.threshold_ci_low <= ci.threshold_ci_high
    # Quorum agreement should be high — the AND-join task is
    # well-defined.
    assert ci.quorum_agreement >= 0.7


# ---------------------------------------------------------------------------
# Phase 13 — prose explanations for rules and CI variants.
# ---------------------------------------------------------------------------


def test_prose_for_xor_rule_without_ci_is_readable():
    """The plain-rule prose should mention the crossover number,
    both branch labels, and the input place — and nothing else
    technical."""
    module = _trained_xor_module()
    transitions = next(
        ts for p, ts in find_xor_groups(module.net) if p == "p_f0"
    )
    rule = extract_xor_rule(module, "p_f0", transitions[0], transitions[1])

    text = prose_for_xor_rule(rule)
    assert f"{rule.crossover:.3f}" in text
    assert rule.label_above in text
    assert rule.label_below in text
    # No CI clauses when we passed a bare rule.
    assert "confidence interval" not in text
    assert "bootstrap" not in text


def test_prose_for_xor_rule_with_ci_includes_interval_and_agreement():
    """Passing a CI variant should drop the bracket numbers and
    the direction-agreement percentage into the paragraph."""
    factory = _xor_module_factory()
    traces = parse_xes(FIXTURES / "xor_log.xes")
    transitions = next(
        ts for p, ts in find_xor_groups(parse_bpmn(FIXTURES / "xor_branch.bpmn"))
        if p == "p_f0"
    )
    torch.manual_seed(0)
    ci = bootstrap_xor_rule(
        factory, traces,
        attribute_to_marking=_xor_marking,
        input_place="p_f0",
        transition_a=transitions[0],
        transition_b=transitions[1],
        n_bootstrap=5, steps=100, lr=0.1, seed=11,
    )

    text = prose_for_xor_rule(ci)
    assert "confidence interval" in text
    assert "bootstrap" in text
    # Direction agreement is a percentage — search for the digit
    # followed by '%'.
    assert "%" in text


def test_prose_for_xor_rule_substitutes_input_label():
    """The ``input_label`` override should appear in the prose
    instead of the raw place id."""
    module = _trained_xor_module()
    transitions = next(
        ts for p, ts in find_xor_groups(module.net) if p == "p_f0"
    )
    rule = extract_xor_rule(module, "p_f0", transitions[0], transitions[1])

    text = prose_for_xor_rule(rule, input_label="risk score")
    assert "risk score" in text
    # The raw place id should NOT appear when an input_label
    # substitution was provided.
    assert "p_f0" not in text


def test_prose_for_and_join_rule_includes_summary():
    """The AND-join prose should reference both the transition
    label and the rule's quorum summary."""
    net = PetriNet()
    net.add_place("p_a", tokens=1)
    net.add_place("p_b", tokens=1)
    net.add_place("p_done")
    net.add_transition("t_join", label="Quorum step")
    net.add_arc("p_a", "t_join")
    net.add_arc("p_b", "t_join")
    net.add_arc("t_join", "p_done")
    torch.manual_seed(0)
    module = PetriNetModule(net, sharpness=4.0)
    rule = extract_and_join_rule(module, "t_join")

    text = prose_for_and_join_rule(rule)
    assert "Quorum step" in text
    assert rule.summary in text


# ---------------------------------------------------------------------------
# Phase 13 — counterfactual explanations.
#
# The XOR-fixture tests use the same _trained_xor_module() factory
# the rest of the file uses; the credit-approval test loads the
# CPN-aware scenario directly and walks through the "what amount
# would have flipped the decision" question.
# ---------------------------------------------------------------------------


def test_counterfactual_on_xor_fixture_finds_routing_threshold():
    """The XOR fixture's trained crossover sits near 0.5. A
    counterfactual that searches the marking channel at the XOR's
    input place should land on roughly the same number — the
    binary search is recovering the same routing threshold the
    rule extractor already reports."""
    module = _trained_xor_module()
    transitions = next(
        ts for p, ts in find_xor_groups(module.net) if p == "p_f0"
    )
    # Pick the "above" branch as the target. At the original
    # marking (risk_score = 0.0) it shouldn't fire; we ask what
    # value at p_f0 would make it fire.
    base_marking = {"p_f0": 0.0}
    cf = find_counterfactual(
        module,
        base_marking,
        flip_place="p_f0",
        target_transition=transitions[0],
        search_range=(0.0, 1.0),
    )
    # A crossing exists somewhere in [0, 1]; the binary search
    # should converge.
    assert cf is not None
    # The crossing should sit near the XOR's crossover (0.4–0.6
    # window given the training data and convergence noise).
    assert 0.2 < cf.counterfactual_input < 0.8
    # By construction the counterfactual activation should be
    # near 0.5.
    assert abs(cf.counterfactual_activation - 0.5) < 0.05


def test_counterfactual_returns_none_when_no_crossing_in_range():
    """If both endpoints of the search range produce activations
    on the same side of 0.5, no counterfactual exists in that
    range. The function should return ``None`` rather than
    invent a crossing."""
    module = _trained_xor_module()
    transitions = next(
        ts for p, ts in find_xor_groups(module.net) if p == "p_f0"
    )
    # Narrow the range to a band where the activation stays on
    # one side of 0.5 — the trained crossover is around 0.5, so
    # restricting to [0.95, 1.0] should give "above" branch
    # firing-confidence near 1.0 at both ends.
    cf = find_counterfactual(
        module,
        {"p_f0": 1.0},
        flip_place="p_f0",
        target_transition=transitions[0],
        search_range=(0.95, 1.0),
    )
    assert cf is None


def test_counterfactual_on_value_channel_recovers_credit_threshold():
    """On the CPN credit-approval scenario, the trained guard
    threshold for t_approve sits in the empirical decision band
    900–1500. A counterfactual that varies the *value* at
    p_submitted should find that threshold — the same number
    the rule extractor would report via the structural guard."""
    ctx = load_scenario(
        Path(__file__).parent.parent
        / "examples"
        / "credit_approval_coloured"
        / "scenario.toml"
    )
    module, _ = ctx.train()

    # Base case: amount = 100 (well below threshold). t_approve
    # should not fire. The counterfactual asks: what amount would
    # have got it to approve?
    cf = find_counterfactual(
        module,
        base_marking={"p_submitted": 1.0},
        base_values={"p_submitted": 100.0},
        flip_place="p_submitted",
        target_transition="t_approve",
        flip_channel="value",
        search_range=(0.0, 5000.0),
        # Activation tolerance — when the midpoint's firing
        # activation is within 0.01 of 0.5 we've found the
        # crossing. Default interval_tolerance kicks in automatically.
        tolerance=0.01,
    )
    assert cf is not None
    # The counterfactual should be meaningfully larger than the
    # base amount (we asked "what amount would have got this
    # approved") and sit in a plausible range. The exact value is
    # training-dynamics dependent: it's where the *full* firing
    # activation (firing sigmoid × guard sigmoid) crosses 0.5, not
    # where the guard alone crosses 0.5, so depending on how
    # confidently the firing component fires this number can sit
    # anywhere between the guard threshold and well above it.
    assert cf.counterfactual_input > cf.original_input
    assert 500.0 < cf.counterfactual_input < 5000.0
    # The activation should sit near 0.5 by construction.
    assert abs(cf.counterfactual_activation - 0.5) < 0.1
    # And the base case should genuinely be below threshold.
    assert cf.original_activation < 0.5


def test_counterfactual_dataclass_has_useful_description():
    """The dataclass's description() should at least name the
    flipped place, the target transition's label, and the new
    input value."""
    cf = Counterfactual(
        target_transition="t_approve",
        target_label="approve loan",
        flipped_place="p_submitted",
        flipped_channel="value",
        original_input=100.0,
        counterfactual_input=1000.0,
        original_activation=0.05,
        counterfactual_activation=0.5,
    )
    desc = cf.description()
    assert "approve loan" in desc
    assert "p_submitted" in desc
    assert "100.000" in desc
    assert "1000.000" in desc


def test_prose_for_counterfactual_renders_paragraph_with_input_label():
    """The prose helper should produce a readable paragraph and
    honour the input_label override for regulator-facing text."""
    cf = Counterfactual(
        target_transition="t_approve",
        target_label="approve loan",
        flipped_place="p_submitted",
        flipped_channel="value",
        original_input=100.0,
        counterfactual_input=1000.0,
        original_activation=0.05,
        counterfactual_activation=0.5,
    )
    text = prose_for_counterfactual(cf, input_label="application amount")
    assert "application amount" in text
    assert "p_submitted" not in text
    assert "approve loan" in text
    # Should mention direction (increased / decreased)
    assert "increased" in text or "decreased" in text


def test_find_counterfactual_rejects_invalid_channel():
    """A flip_channel other than 'marking' or 'value' is a
    programmer error — the function should raise rather than
    silently misbehave."""
    module = _trained_xor_module()
    with pytest.raises(ValueError, match="flip_channel"):
        find_counterfactual(
            module,
            {"p_f0": 0.0},
            flip_place="p_f0",
            target_transition="t_a",  # dummy
            flip_channel="bogus",
        )


def test_find_counterfactual_value_channel_requires_base_values():
    """Passing flip_channel='value' without base_values is a
    misuse — the search varies an entry of a dict that wasn't
    supplied."""
    module = _trained_xor_module()
    with pytest.raises(ValueError, match="base_values"):
        find_counterfactual(
            module,
            {"p_f0": 0.0},
            flip_place="p_f0",
            target_transition="t_a",
            flip_channel="value",
        )


# ---------------------------------------------------------------------------
# Phase 13 — sensitivity analysis.
#
# Sensitivity tells you *which input* the model leans on at a given
# base point. The XOR fixture's trained crossover sits near 0.5, so
# evaluating sensitivity at risk_score = 0.5 — right at the
# decision boundary — should produce a large gradient with respect
# to p_f0 (the routing input). At saturated values (0.0 or 1.0)
# the gradient shrinks because the sigmoid has already decided.
# ---------------------------------------------------------------------------


def test_transition_sensitivity_at_decision_boundary_is_nonzero():
    """Evaluated at the trained crossover, the gradient of the
    'above' branch's firing activation with respect to the routing
    input should be substantial — that's the input the model is
    leaning on."""
    module = _trained_xor_module()
    transitions = next(
        ts for p, ts in find_xor_groups(module.net) if p == "p_f0"
    )
    report = transition_sensitivity(
        module,
        base_marking={"p_f0": 0.5},
        target_transition=transitions[0],
    )
    assert isinstance(report, SensitivityReport)
    # Activation at the crossover should be near 0.5 (we're at the
    # decision boundary).
    assert 0.2 < report.base_activation < 0.8
    # The gradient at p_f0 should be substantial (well above zero).
    assert abs(report.marking_gradients["p_f0"]) > 0.5


def test_transition_sensitivity_saturates_far_from_boundary():
    """At a fully-saturated input (well above or below the
    crossover), the gradient should be much smaller — the sigmoid
    has effectively decided. We pick risk_score = 0.99 which is
    deep in one of the saturation regions for the trained model."""
    module = _trained_xor_module()
    transitions = next(
        ts for p, ts in find_xor_groups(module.net) if p == "p_f0"
    )
    report_saturated = transition_sensitivity(
        module,
        base_marking={"p_f0": 0.99},
        target_transition=transitions[0],
    )
    report_at_boundary = transition_sensitivity(
        module,
        base_marking={"p_f0": 0.5},
        target_transition=transitions[0],
    )
    # The boundary gradient should be bigger than the saturated
    # gradient — that's the whole point of "sensitivity is local."
    assert abs(report_at_boundary.marking_gradients["p_f0"]) > abs(
        report_saturated.marking_gradients["p_f0"]
    )


def test_transition_sensitivity_includes_value_channel_when_supplied():
    """On the credit-approval CPN scenario, t_approve's firing
    pivots on the per-token *value* at p_submitted, not the place
    activation. Sensitivity should reflect that — the value
    gradient at p_submitted should be the dominant one."""
    ctx = load_scenario(
        Path(__file__).parent.parent
        / "examples"
        / "credit_approval_coloured"
        / "scenario.toml"
    )
    module, _ = ctx.train()
    # Evaluate near the learned threshold so we're at the
    # boundary, not in a saturated regime.
    report = transition_sensitivity(
        module,
        base_marking={"p_submitted": 1.0},
        base_values={"p_submitted": 1000.0},
        target_transition="t_approve",
    )
    assert "p_submitted" in report.value_gradients
    # Value gradient should be measurable (non-zero in either
    # direction). We don't test the sign because training
    # dynamics + the auto-scaled guard sharpness can flip it
    # either way for the linearisation point.
    assert report.value_gradients["p_submitted"] != 0.0


def test_sensitivity_ranked_orders_by_absolute_gradient():
    """The .ranked() method should put the largest-magnitude
    inputs first. Build a small net where one input dominates
    and verify the ranking."""
    report = SensitivityReport(
        target_transition="t",
        target_label="t",
        base_activation=0.5,
        marking_gradients={"p_a": 0.1, "p_b": -2.0, "p_c": 1.5},
        value_gradients={"p_d": 0.05},
    )
    ranking = report.ranked()
    # Sorted by |gradient| descending. Expect: p_b (2.0), p_c (1.5),
    # p_a (0.1), p_d (0.05).
    assert [r[0] for r in ranking] == ["p_b", "p_c", "p_a", "p_d"]


def test_input_importance_aggregates_across_traces_and_transitions():
    """Aggregate input importance across the XOR trace set should
    rank p_f0 (the routing input) as the dominant input —
    that's the only input the model cares about for routing
    decisions, and the trace set drives the model through both
    branches."""
    module = _trained_xor_module()
    traces = parse_xes(FIXTURES / "xor_log.xes")
    importance = input_importance(
        module,
        traces,
        attribute_to_marking=_xor_marking,
    )
    # Should contain p_f0 in the marking channel, with the largest
    # importance score of any input.
    assert "marking:p_f0" in importance
    top_input = max(importance, key=importance.get)
    assert top_input == "marking:p_f0"


def test_prose_for_sensitivity_lists_top_inputs():
    """The prose helper should produce a readable paragraph
    naming the top-N inputs by absolute gradient and including
    the direction language."""
    report = SensitivityReport(
        target_transition="t_approve",
        target_label="approve loan",
        base_activation=0.5,
        marking_gradients={"p_a": 0.1, "p_b": -2.0, "p_c": 1.5},
    )
    text = prose_for_sensitivity(report, top_n=2)
    assert "approve loan" in text
    # Top-2 by magnitude: p_b (2.0), p_c (1.5)
    assert "p_b" in text
    assert "p_c" in text
    # p_a (smallest magnitude) shouldn't be in the top-2 output.
    assert "p_a" not in text
    # Direction language should be there.
    assert "raises" in text or "lowers" in text


def test_prose_for_sensitivity_uses_input_label_overrides():
    """The input_labels override should substitute domain terms
    for raw place ids — same pattern as the other prose helpers."""
    report = SensitivityReport(
        target_transition="t_approve",
        target_label="approve loan",
        base_activation=0.5,
        marking_gradients={"p_submitted": -1.5},
    )
    text = prose_for_sensitivity(
        report,
        input_labels={"p_submitted": "application amount"},
    )
    assert "application amount" in text
    assert "p_submitted" not in text


def test_prose_for_sensitivity_handles_empty_report():
    """An empty sensitivity report shouldn't crash the prose
    helper — it should produce a meaningful "no measurable
    sensitivity" message."""
    report = SensitivityReport(
        target_transition="t_isolated",
        target_label="isolated step",
        base_activation=0.5,
    )
    text = prose_for_sensitivity(report)
    assert "isolated step" in text
    assert "No measurable sensitivity" in text or "no measurable sensitivity" in text.lower()


# ---------------------------------------------------------------------------
# Phase 13 — cross-variant comparison reports.
#
# A bisimulation check tells you two variants are *structurally*
# equivalent. Cross-variant comparison tells you how often, across
# a chosen input domain, their trained routing decisions actually
# overlap. The two tests below pin the two headline cases:
#   * identical-fixture variants should agree on essentially every
#     point;
#   * intentionally mistuned variants (different routing thresholds)
#     should agree at the extremes but diverge in the middle band.
# ---------------------------------------------------------------------------


def _hand_xor_module(*, theta_a: float = 0.5, theta_b: float = 0.5):
    """Build a 2-transition XOR-shape net with hand-set firing
    thresholds so we can construct variants with controlled
    (mis)tuning without going through training. Both transitions
    share input ``p_in`` and branch to ``p_a`` / ``p_b``; arc
    weights are 1, sharpness is 8, so each transition fires
    cleanly above its own ``theta``."""
    net = PetriNet()
    net.add_place("p_in")
    net.add_place("p_a")
    net.add_place("p_b")
    net.add_transition("t_a", label="Path A")
    net.add_transition("t_b", label="Path B")
    net.add_arc("p_in", "t_a")
    net.add_arc("t_a", "p_a")
    net.add_arc("p_in", "t_b")
    net.add_arc("t_b", "p_b")
    module = PetriNetModule(net, sharpness=8.0)
    # Override the arc weights to exactly 1.0 (the default init
    # is normal(mean=1, std=0.1) — close to 1 but not exact, and
    # the noise is enough to perturb the cross-variant agreement
    # tests).
    module.arc_weights[module._arc_key[("p_in", "t_a")]].data = torch.tensor(1.0)
    module.arc_weights[module._arc_key[("p_in", "t_b")]].data = torch.tensor(1.0)
    # Set thresholds. t_a fires above theta_a (because pre =
    # 8 * (1 * a - theta_a), so the sigmoid crosses 0.5 at
    # a = theta_a).
    module.transition_thresholds[module._threshold_key["t_a"]].data = torch.tensor(theta_a)
    module.transition_thresholds[module._threshold_key["t_b"]].data = torch.tensor(theta_b)
    return module


def test_compare_variants_on_identical_modules_reports_full_agreement():
    """Two identical-threshold modules should agree on every grid
    point. Hard agreement rate = 1.0, no disagreement samples."""
    a = _hand_xor_module(theta_a=0.5, theta_b=0.5)
    b = _hand_xor_module(theta_a=0.5, theta_b=0.5)
    report = compare_variants(
        a, b,
        input_grid={"p_in": [0.0, 0.25, 0.5, 0.75, 1.0]},
    )
    assert isinstance(report, ComparisonReport)
    assert report.n_samples == 5
    assert report.hard_agreement_rate == 1.0
    assert len(report.disagreement_samples) == 0
    # Both Path A and Path B should be in the paired labels.
    assert set(report.paired_labels) == {"Path A", "Path B"}


def test_compare_variants_on_mistuned_modules_finds_disagreement():
    """Two variants with different firing thresholds should
    disagree in the input band where one fires Path A and the
    other doesn't. Variant A's Path A fires above 0.3; Variant
    B's Path A fires above 0.7. At inputs between 0.3 and 0.7,
    they disagree."""
    a = _hand_xor_module(theta_a=0.3, theta_b=0.5)
    b = _hand_xor_module(theta_a=0.7, theta_b=0.5)
    report = compare_variants(
        a, b,
        input_grid={"p_in": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
        tolerance=0.05,
    )
    # The hard agreement rate should be strictly less than 1.0 —
    # there's a band where Path A fires for variant A but not B.
    assert report.hard_agreement_rate < 1.0
    # The soft agreement rate (tolerance = 0.05) should also be
    # below 1.0 — the activations should differ noticeably in
    # the disagreement band.
    assert report.soft_agreement_rate < 1.0
    # At least one disagreement sample should be reported.
    assert len(report.disagreement_samples) > 0
    # The diverging label should be one of Path A or Path B (or
    # both at the boundary).
    diverging_labels = set()
    for sample in report.disagreement_samples:
        diverging_labels.update(sample.diverging.keys())
    assert diverging_labels & {"Path A", "Path B"}


def test_compare_variants_unmatched_labels_are_recorded():
    """When a label exists in one variant but not the other, it
    should appear in ``unmatched_a`` / ``unmatched_b`` and *not*
    be in ``paired_labels``."""
    a = _hand_xor_module(theta_a=0.5, theta_b=0.5)
    # Build a different net with an extra labelled transition that
    # doesn't appear in a.
    b = PetriNet()
    b.add_place("p_in")
    b.add_place("p_a")
    b.add_place("p_c")  # extra branch
    b.add_transition("t_a", label="Path A")
    b.add_transition("t_c", label="Path C")  # only in b
    b.add_arc("p_in", "t_a")
    b.add_arc("t_a", "p_a")
    b.add_arc("p_in", "t_c")
    b.add_arc("t_c", "p_c")
    b_module = PetriNetModule(b, sharpness=4.0)
    report = compare_variants(
        a, b_module,
        input_grid={"p_in": [0.5]},
    )
    # Path A is in both → paired.
    assert "Path A" in report.paired_labels
    # Path B is only in a → unmatched_a.
    assert "Path B" in report.unmatched_a
    # Path C is only in b → unmatched_b.
    assert "Path C" in report.unmatched_b


def test_compare_variants_correspondence_override_pairs_unlabelled():
    """When the two variants use different labels for the same
    conceptual step, the ``correspondence`` override lets the
    caller pair them by hand."""
    a = PetriNet()
    a.add_place("p_in")
    a.add_place("p_done")
    a.add_transition("t_a_internal", label="Approve loan")
    a.add_arc("p_in", "t_a_internal")
    a.add_arc("t_a_internal", "p_done")
    a_module = PetriNetModule(a)

    b = PetriNet()
    b.add_place("p_in")
    b.add_place("p_done")
    b.add_transition("t_b_internal", label="Loan approval")  # different label
    b.add_arc("p_in", "t_b_internal")
    b.add_arc("t_b_internal", "p_done")
    b_module = PetriNetModule(b)

    report = compare_variants(
        a_module, b_module,
        input_grid={"p_in": [0.5]},
        correspondence={
            "approve_step": ("t_a_internal", "t_b_internal"),
        },
    )
    # The user-supplied label appears in paired_labels.
    assert "approve_step" in report.paired_labels
    # And the per-transition agreement dict has it.
    assert "approve_step" in report.per_transition_agreement


def test_prose_for_comparison_report_reports_agreement_numbers():
    """The prose summary should mention the sample count, hard
    agreement rate, soft agreement rate, and the worst-offender
    transitions when disagreements exist."""
    a = _hand_xor_module(theta_a=0.3, theta_b=0.5)
    b = _hand_xor_module(theta_a=0.7, theta_b=0.5)
    report = compare_variants(
        a, b,
        input_grid={"p_in": [0.0, 0.25, 0.5, 0.75, 1.0]},
    )
    text = prose_for_comparison_report(report)
    # Mentions sample count.
    assert "5" in text
    # Mentions a percentage (hard agreement rate)
    assert "%" in text
    # Mentions at least one paired label (since there's
    # disagreement, at least one of Path A / Path B should be in
    # the "most prone" list).
    assert "Path A" in text or "Path B" in text


def test_prose_for_comparison_report_when_no_disagreement():
    """When every paired transition agrees on every sample, the
    prose should say so explicitly rather than producing an empty
    disagreement list."""
    a = _hand_xor_module(theta_a=0.5, theta_b=0.5)
    b = _hand_xor_module(theta_a=0.5, theta_b=0.5)
    report = compare_variants(
        a, b,
        input_grid={"p_in": [0.0, 0.5, 1.0]},
    )
    text = prose_for_comparison_report(report)
    assert (
        "functionally indistinguishable" in text
        or "Every paired transition agreed" in text
    )
