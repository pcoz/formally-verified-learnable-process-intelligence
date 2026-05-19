"""Tests for Phase 7 anomaly detection evaluation.

Three pieces:

  * generators (`drop_event`, `insert_event`, `swap_event_labels`,
    `shuffle_events`) produce controlled corruptions of normal traces
    without mutating the originals;
  * AUC characterisation — train a `PetriNetModule` on normal traces,
    generate matched anomalous traces, and confirm the structured
    detector produces a meaningfully separable score distribution;
  * baseline comparison — the structured detector outperforms a
    frequency-only baseline on the branch-flip case, isolating the
    contribution of the structural prior plus attribute conditioning.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    FrequencyBaseline,
    PetriNetModule,
    XESEvent,
    XESTrace,
    auc,
    drop_event,
    insert_event,
    parse_bpmn,
    parse_xes,
    shuffle_events,
    swap_event_labels,
    trace_anomaly_score,
    train_on_traces,
)


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def test_drop_event_removes_one_event_and_leaves_original_intact():
    original = XESTrace(
        attributes={"name": "case"},
        events=[XESEvent(name="A"), XESEvent(name="B"), XESEvent(name="C")],
    )
    dropped = drop_event(original, index=1)
    assert [e.name for e in dropped.events] == ["A", "C"]
    assert [e.name for e in original.events] == ["A", "B", "C"]


def test_drop_event_on_empty_trace_is_a_noop():
    original = XESTrace()
    assert drop_event(original).events == []


def test_insert_event_adds_one_event():
    original = XESTrace(events=[XESEvent(name="A")])
    extended = insert_event(original, "X")
    assert [e.name for e in extended.events] == ["A", "X"]
    assert [e.name for e in original.events] == ["A"]


def test_swap_event_labels_flips_only_the_named_labels():
    original = XESTrace(
        events=[XESEvent(name="A"), XESEvent(name="B"), XESEvent(name="C")]
    )
    swapped = swap_event_labels(original, "A", "B")
    assert [e.name for e in swapped.events] == ["B", "A", "C"]


def test_shuffle_events_is_deterministic_with_seed():
    original = XESTrace(
        events=[XESEvent(name=str(i)) for i in range(5)]
    )
    a = shuffle_events(original, seed=42)
    b = shuffle_events(original, seed=42)
    assert [e.name for e in a.events] == [e.name for e in b.events]
    assert {e.name for e in a.events} == {"0", "1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# AUC characterisation on the XOR fixture
# ---------------------------------------------------------------------------


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
    return module, traces


def _xor_marking(trace):
    return {"p_f0": float(trace.attributes["risk_score"])}


def test_branch_flip_anomalies_have_high_auc():
    """Branch flipping — replacing the recorded "Path A" with "Path B"
    given a high risk_score (and vice versa) — should be the most
    detectable anomaly type for the XOR shape, because routing was
    the structural learning target."""
    module, normal_traces = _trained_xor_module()
    anomalous = [
        swap_event_labels(t, "Path A", "Path B") for t in normal_traces
    ]
    normal_scores = [
        trace_anomaly_score(module, t, attribute_to_marking=_xor_marking)
        for t in normal_traces
    ]
    anomalous_scores = [
        trace_anomaly_score(module, t, attribute_to_marking=_xor_marking)
        for t in anomalous
    ]
    assert auc(anomalous_scores, normal_scores) > 0.9


def test_inserted_event_anomalies_are_detected():
    module, normal_traces = _trained_xor_module()
    anomalous = [insert_event(t, "Unknown step") for t in normal_traces]
    normal_scores = [
        trace_anomaly_score(module, t, attribute_to_marking=_xor_marking)
        for t in normal_traces
    ]
    anomalous_scores = [
        trace_anomaly_score(module, t, attribute_to_marking=_xor_marking)
        for t in anomalous
    ]
    assert auc(anomalous_scores, normal_scores) >= 0.5


def test_dropped_event_anomalies_are_detected():
    """A trace with no events at all has no transitions matching any
    label, so every expected firing shows up as a residual. The
    score should be strictly higher than the lowest in-distribution
    score."""
    module, normal_traces = _trained_xor_module()
    anomalous = [drop_event(t) for t in normal_traces]
    normal_scores = [
        trace_anomaly_score(module, t, attribute_to_marking=_xor_marking)
        for t in normal_traces
    ]
    anomalous_scores = [
        trace_anomaly_score(module, t, attribute_to_marking=_xor_marking)
        for t in anomalous
    ]
    assert max(anomalous_scores) > min(normal_scores)
    assert auc(anomalous_scores, normal_scores) > 0.5


# ---------------------------------------------------------------------------
# AUC helper itself
# ---------------------------------------------------------------------------


def test_auc_perfect_separation():
    assert auc([1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]) == pytest.approx(1.0)


def test_auc_no_separation():
    assert auc([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.5)


def test_auc_empty_input_returns_nan():
    import math
    assert math.isnan(auc([], [1.0]))
    assert math.isnan(auc([1.0], []))


# ---------------------------------------------------------------------------
# Baseline comparison: structural prior + attribute conditioning vs
# pure marginal-frequency detection
# ---------------------------------------------------------------------------


def test_structured_detector_beats_frequency_baseline_on_branch_flip():
    """Branch flipping is invisible to a frequency baseline: both
    "Path A" and "Path B" are common labels seen in training, so the
    baseline assigns similar likelihood to either. The structured
    detector, conditioning on each trace's risk_score, knows which
    branch is *expected* for a given input and flags the wrong-branch
    traces. This is the load-bearing comparison for the structural
    prior's value-add."""
    module, normal_traces = _trained_xor_module()
    anomalous = [
        swap_event_labels(t, "Path A", "Path B") for t in normal_traces
    ]

    structured_normal = [
        trace_anomaly_score(module, t, attribute_to_marking=_xor_marking)
        for t in normal_traces
    ]
    structured_anomalous = [
        trace_anomaly_score(module, t, attribute_to_marking=_xor_marking)
        for t in anomalous
    ]
    structured_auc = auc(structured_anomalous, structured_normal)

    baseline = FrequencyBaseline().fit(normal_traces)
    baseline_normal = [baseline.score(t) for t in normal_traces]
    baseline_anomalous = [baseline.score(t) for t in anomalous]
    baseline_auc = auc(baseline_anomalous, baseline_normal)

    assert structured_auc > 0.9
    assert baseline_auc == pytest.approx(0.5)
    assert structured_auc > baseline_auc + 0.3


def test_frequency_baseline_does_detect_unseen_labels():
    """Sanity check: the frequency baseline isn't useless. It catches
    anomalies that introduce unseen event labels, where the structural
    prior is not strictly needed."""
    _, normal_traces = _trained_xor_module()
    baseline = FrequencyBaseline().fit(normal_traces)
    anomalous = [insert_event(t, "TOTALLY UNKNOWN LABEL") for t in normal_traces]

    normal_scores = [baseline.score(t) for t in normal_traces]
    anomalous_scores = [baseline.score(t) for t in anomalous]
    assert auc(anomalous_scores, normal_scores) > 0.9


def test_frequency_baseline_requires_fit_first():
    baseline = FrequencyBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        baseline.score(XESTrace(events=[XESEvent(name="A")]))
