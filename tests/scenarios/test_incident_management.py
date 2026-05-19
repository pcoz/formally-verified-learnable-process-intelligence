"""End-to-end test against real BPI Challenge 2013 data.

The scenario points at a 1.3 MB compressed XES file committed to
the repo — Volvo IT's public incident-management log, 7,554
real ITIL incident tickets. The adapter reads gzipped XES
transparently and trains on the full corpus.

This is the strongest "works on actual real data" piece of
evidence in the test suite: not a hand-crafted fixture inspired
by BPI Challenge, but the literal public dataset.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import XESEvent, XESTrace, load_scenario, train_on_traces


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "incident_management"
    / "scenario.toml"
)

# The BPI 2013 dataset uses string-valued impact labels rather than
# numeric ones. Map them onto a [0, 1] severity scale the network
# can read as an input-marking activation.
IMPACT_TO_SEVERITY = {
    "Low": 0.2,
    "Medium": 0.5,
    "High": 0.8,
    "Major": 0.95,
}


def _impact_to_marking(trace: XESTrace) -> dict[str, float]:
    raw = trace.attributes.get("impact", "Medium")
    return {"p_opened": IMPACT_TO_SEVERITY.get(raw, 0.5)}


def test_real_bpi_data_loads_from_gzipped_xes():
    """The committed BPI 2013 incidents log loads as the full
    7,554-trace public release. Each trace carries the promoted
    `impact` attribute lifted from the event level, and each
    event's name has been rewritten to the lifecycle state."""
    ctx = load_scenario(SCENARIO)
    assert len(ctx.traces) == 7554
    impacts = {t.attributes.get("impact") for t in ctx.traces}
    # The BPI 2013 dataset uses these four impact strings.
    assert impacts <= {"Low", "Medium", "High", "Major", None}
    # And the trace concept:name (the actual incident ID) should
    # also be present from the source XES.
    assert all("concept:name" in t.attributes for t in ctx.traces)
    # Event names should now be lifecycle states, not the original
    # high-level concept:name tags.
    lifecycle_states = {e.name for t in ctx.traces[:50] for e in t.events}
    assert "In Progress" in lifecycle_states
    assert "Resolved" in lifecycle_states or "Closed" in lifecycle_states


def test_real_data_training_runs():
    """Train PETRA against the real BPI 2013 traces. The custom
    impact->severity mapping bridges the categorical impact label
    to a numeric input-marking activation."""
    torch.manual_seed(0)
    ctx = load_scenario(SCENARIO)
    module = ctx.compile()
    losses = train_on_traces(
        module,
        ctx.traces,
        attribute_to_marking=_impact_to_marking,
        steps=ctx.training.steps,
        lr=ctx.training.lr,
    )
    # Training should make real progress on real traces — final loss
    # noticeably below initial.
    assert losses[-1] < losses[0]


def test_real_canonical_path_predicted_after_training():
    """After training on the real log the canonical In Progress →
    Resolved → Closed transitions should all show meaningful
    activation under a typical input. We pick a medium-impact
    input and look at the activations of the modelled transitions."""
    torch.manual_seed(0)
    ctx = load_scenario(SCENARIO)
    module = ctx.compile()
    train_on_traces(
        module,
        ctx.traces,
        attribute_to_marking=_impact_to_marking,
        steps=ctx.training.steps,
        lr=ctx.training.lr,
    )
    with torch.no_grad():
        out = module(input_marking={"p_opened": torch.tensor([0.5])})
    # The most common modelled transition in the real log is
    # "In Progress" — it fires in essentially every trace. Its
    # activation should be high after training.
    assert out["t_in_progress"].item() > 0.5


def test_anomalous_trace_skipping_resolved_is_flagged():
    """A trace that goes In Progress → Closed without going through
    Resolved is a real conformance violation. PETRA's residual
    on `t_resolved` should pick it up."""
    from petri_net_nn import anomaly_score

    torch.manual_seed(0)
    ctx = load_scenario(SCENARIO)
    module = ctx.compile()
    train_on_traces(
        module,
        ctx.traces,
        attribute_to_marking=_impact_to_marking,
        steps=ctx.training.steps,
        lr=ctx.training.lr,
    )

    normal = XESTrace(
        attributes={"impact": "Medium"},
        events=[
            XESEvent(name="In Progress"),
            XESEvent(name="Resolved"),
            XESEvent(name="Closed"),
        ],
    )
    anomalous = XESTrace(
        attributes={"impact": "Medium"},
        events=[
            XESEvent(name="In Progress"),
            XESEvent(name="Closed"),
        ],
    )
    normal_scores = anomaly_score(
        module, normal, attribute_to_marking=_impact_to_marking
    )
    anomalous_scores = anomaly_score(
        module, anomalous, attribute_to_marking=_impact_to_marking
    )
    # The anomalous trace's residual on t_resolved (the skipped
    # step) must be larger than the normal trace's.
    assert anomalous_scores["t_resolved"] > normal_scores["t_resolved"]
