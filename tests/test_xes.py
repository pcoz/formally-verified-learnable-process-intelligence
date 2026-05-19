"""Tests for the XES log loader and trace-driven training utilities."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    PetriNetModule,
    XESEvent,
    XESTrace,
    anomaly_score,
    parse_bpmn,
    parse_xes,
    trace_occurrence_vector,
    train_on_traces,
)


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# XES parser
# ---------------------------------------------------------------------------


def test_parse_xes_returns_traces_and_events():
    traces = parse_xes(FIXTURES / "simple_log.xes")
    assert len(traces) == 4
    assert all(len(t.events) == 1 for t in traces)
    assert all(t.events[0].name == "Do work" for t in traces)
    assert traces[0].name == "case-001"


def test_parse_xes_collects_trace_attributes():
    traces = parse_xes(FIXTURES / "xor_log.xes")
    assert len(traces) == 12
    assert "risk_score" in traces[0].attributes
    assert float(traces[0].attributes["risk_score"]) > 0.5


def test_parse_xes_rejects_non_log_root():
    bad = "<?xml version='1.0'?><notlog/>"
    with pytest.raises(ValueError, match="expected XES root element"):
        parse_xes(bad)


def test_parse_xes_accepts_documents_without_namespace():
    xml = """<?xml version='1.0'?>
        <log>
          <trace>
            <string key="concept:name" value="t1"/>
            <event><string key="concept:name" value="A"/></event>
            <event><string key="concept:name" value="B"/></event>
          </trace>
        </log>"""
    traces = parse_xes(xml)
    assert len(traces) == 1
    assert [e.name for e in traces[0].events] == ["A", "B"]


# ---------------------------------------------------------------------------
# Trace → transition mapping
# ---------------------------------------------------------------------------


def test_trace_occurrence_vector_matches_by_label():
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    scored = sorted(
        t for t in net.transitions if "->" not in net.transition_labels.get(t, t)
    )
    trace = XESTrace(events=[XESEvent(name="Path A")])
    vec = trace_occurrence_vector(net, trace, scored)
    assert vec.sum().item() == 1.0
    fired_idx = vec.argmax().item()
    assert net.transition_labels[scored[fired_idx]] == "Path A"


def test_trace_occurrence_vector_ignores_unknown_events():
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    scored = sorted(net.transitions)
    trace = XESTrace(events=[XESEvent(name="not a real task")])
    vec = trace_occurrence_vector(net, trace, scored)
    assert vec.sum().item() == 0.0


# ---------------------------------------------------------------------------
# Training on traces — spec §10 Step 3, sequential and XOR
# ---------------------------------------------------------------------------


def test_train_on_traces_sequential_log():
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    module = PetriNetModule(net)
    traces = parse_xes(FIXTURES / "simple_log.xes")

    losses = train_on_traces(
        module,
        traces,
        attribute_to_marking=lambda trace: {"p_f1": 1.0},
        steps=300,
        lr=0.1,
    )
    assert losses[-1] < losses[0] * 0.3

    with torch.no_grad():
        out = module(input_marking={"p_f1": torch.tensor([1.0])})
    assert out["t_do_work"].item() > 0.7


def test_train_on_traces_xor_log_learns_conditional_routing():
    """Train xor_branch on the XOR log where each trace's risk_score
    determines whether Path A or Path B fired. After training, a high
    risk_score should activate Path A's transition more than Path B's."""
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    module = PetriNetModule(net)
    traces = parse_xes(FIXTURES / "xor_log.xes")

    def to_marking(trace):
        return {"p_f0": float(trace.attributes["risk_score"])}

    losses = train_on_traces(
        module,
        traces,
        attribute_to_marking=to_marking,
        steps=1500,
        lr=0.1,
    )
    assert losses[-1] < 0.2

    label_to_t = {net.transition_labels[t]: t for t in net.transitions}
    t_pathA = label_to_t["Path A"]
    t_pathB = label_to_t["Path B"]

    with torch.no_grad():
        high = module(input_marking={"p_f0": torch.tensor([0.95])})
        low = module(input_marking={"p_f0": torch.tensor([0.05])})
    assert high[t_pathA].item() > high[t_pathB].item()
    assert low[t_pathB].item() > low[t_pathA].item()


# ---------------------------------------------------------------------------
# Anomaly detection — §7.2
# ---------------------------------------------------------------------------


def test_anomaly_score_flags_off_path_trace():
    """After training on traces that all routed high-risk -> Path A and
    low-risk -> Path B, a trace that takes the *opposite* route for its
    risk_score should produce a larger total residual on the diverging
    arcs than a trace that takes the expected route. This is the §7.2
    fingerprint-vs-deviation pattern."""
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    module = PetriNetModule(net)
    traces = parse_xes(FIXTURES / "xor_log.xes")

    def to_marking(trace):
        return {"p_f0": float(trace.attributes["risk_score"])}

    train_on_traces(
        module, traces, attribute_to_marking=to_marking, steps=1500, lr=0.1
    )

    in_distribution = XESTrace(
        attributes={"risk_score": "0.9"},
        events=[XESEvent(name="Path A")],
    )
    anomalous = XESTrace(
        attributes={"risk_score": "0.9"},
        events=[XESEvent(name="Path B")],
    )

    ok_scores = anomaly_score(module, in_distribution, attribute_to_marking=to_marking)
    bad_scores = anomaly_score(module, anomalous, attribute_to_marking=to_marking)

    assert sum(bad_scores.values()) > sum(ok_scores.values()) + 0.5

    label_to_t = {net.transition_labels[t]: t for t in net.transitions}
    t_pathA = label_to_t["Path A"]
    t_pathB = label_to_t["Path B"]
    diverging = bad_scores[t_pathA] + bad_scores[t_pathB]
    other_residuals = sum(
        v for t, v in bad_scores.items() if t not in {t_pathA, t_pathB}
    )
    assert diverging > other_residuals
