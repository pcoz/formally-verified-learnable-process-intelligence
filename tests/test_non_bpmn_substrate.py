"""Validate the framing claim that the substrate is not BPMN-specific.

The ROADMAP framing section claims (point 2) that the cyclic compiler
covers "any discrete-event dynamical system that can be expressed as a
place/transition graph", with biological signalling pathways named
explicitly. This module instantiates a small kinase signalling cascade
as a hand-coded Petri net — no BPMN XML, no parser, no business
process language anywhere — and runs the full pipeline against it:

  * compile to PetriNetModule (the substrate);
  * train on synthetic traces with signal-strength attribute
    conditioning (the §4.2 learning loop);
  * distil the learned routing into a readable rule (Phase 8
    interpretability, in biological terms);
  * detect mismatched-route anomalies (§7.2);
  * verify two structurally-isomorphic cascades are bisimilar
    (Phase 2 §7.3).

Together these confirm: the primitives travel out of the BPMN domain
without modification.
"""
from __future__ import annotations

import pytest
import torch

from petri_net_nn import (
    PetriNet,
    PetriNetModule,
    XESEvent,
    XESTrace,
    anomaly_score,
    are_bisimilar,
    extract_routing_rules,
    find_xor_groups,
    train_on_traces,
)


def _signalling_cascade(*, prefix: str = "") -> PetriNet:
    """Two-step kinase signalling cascade with signal-strength-dependent
    routing.

    Biology in scope:
      * an incoming ligand signal binds a receptor;
      * the activated receptor phosphorylates a downstream kinase;
      * strong signals favour a fast pathway (direct effector);
      * weak signals favour a slow pathway (full cascade with an
        intermediate kinase).

    Petri-net mapping:
      * places carry the molecular state ("kinase active",
        "intermediate active", "fast effector active", ...);
      * transitions carry biological reactions (phosphorylation,
        downstream activation);
      * the signal strength is encoded as the activation of the
        ``p_signal`` source place — high activation drives the fast
        pathway; low activation drives the slow pathway.
    """
    net = PetriNet()
    net.add_place(f"{prefix}p_signal", tokens=1, label="bound ligand")
    net.add_place(f"{prefix}p_kinase_active", label="kinase phosphorylated")
    net.add_place(f"{prefix}p_intermediate_active", label="intermediate kinase active")
    net.add_place(f"{prefix}p_fast_effector", label="fast effector active")
    net.add_place(f"{prefix}p_slow_effector", label="slow effector active")

    net.add_transition(f"{prefix}t_phosphorylate", label="phosphorylate kinase")
    net.add_arc(f"{prefix}p_signal", f"{prefix}t_phosphorylate")
    net.add_arc(f"{prefix}t_phosphorylate", f"{prefix}p_kinase_active")

    net.add_transition(f"{prefix}t_fast_pathway", label="fast pathway")
    net.add_arc(f"{prefix}p_kinase_active", f"{prefix}t_fast_pathway")
    net.add_arc(f"{prefix}t_fast_pathway", f"{prefix}p_fast_effector")

    net.add_transition(f"{prefix}t_slow_pathway", label="slow pathway")
    net.add_arc(f"{prefix}p_kinase_active", f"{prefix}t_slow_pathway")
    net.add_arc(f"{prefix}t_slow_pathway", f"{prefix}p_intermediate_active")

    net.add_transition(f"{prefix}t_slow_finalise", label="slow effector activation")
    net.add_arc(f"{prefix}p_intermediate_active", f"{prefix}t_slow_finalise")
    net.add_arc(f"{prefix}t_slow_finalise", f"{prefix}p_slow_effector")
    return net


def _signal_marking(trace: XESTrace) -> dict[str, float]:
    return {"p_signal": float(trace.attributes["signal_strength"])}


def _synthetic_traces() -> list[XESTrace]:
    """Strong-signal traces fire the fast pathway; weak-signal traces
    fire the slow pathway. The kinase phosphorylation event always
    fires (it is upstream of the routing decision)."""
    traces: list[XESTrace] = []
    for strength in (0.92, 0.85, 0.78, 0.95, 0.68, 0.71, 0.83, 0.88):
        traces.append(
            XESTrace(
                attributes={"signal_strength": str(strength)},
                events=[
                    XESEvent(name="phosphorylate kinase"),
                    XESEvent(name="fast pathway"),
                ],
            )
        )
    for strength in (0.08, 0.21, 0.33, 0.12, 0.04, 0.27, 0.15, 0.31):
        traces.append(
            XESTrace(
                attributes={"signal_strength": str(strength)},
                events=[
                    XESEvent(name="phosphorylate kinase"),
                    XESEvent(name="slow pathway"),
                    XESEvent(name="slow effector activation"),
                ],
            )
        )
    return traces


# ---------------------------------------------------------------------------
# Compilation — the substrate accepts hand-coded non-BPMN nets
# ---------------------------------------------------------------------------


def test_signalling_cascade_validates_and_compiles():
    net = _signalling_cascade()
    assert net.validate() == []
    torch.manual_seed(0)
    module = PetriNetModule(net)
    out = module()
    assert "p_fast_effector" in out
    assert "p_slow_effector" in out


def test_signalling_cascade_xor_groups_are_detected_in_biological_net():
    """find_xor_groups operates on the Petri-net structure, not BPMN
    syntax. It should identify the routing decision at p_kinase_active
    regardless of whether the net came from BPMN or hand-coded biology."""
    net = _signalling_cascade()
    groups = find_xor_groups(net)
    routing_groups = [(p, t) for p, t in groups if p == "p_kinase_active"]
    assert len(routing_groups) == 1
    _, transitions = routing_groups[0]
    assert set(transitions) == {"t_fast_pathway", "t_slow_pathway"}


# ---------------------------------------------------------------------------
# Training — §4.2 learning loop works on the biological net
# ---------------------------------------------------------------------------


def _train_cascade() -> PetriNetModule:
    torch.manual_seed(0)
    module = PetriNetModule(_signalling_cascade())
    train_on_traces(
        module,
        _synthetic_traces(),
        attribute_to_marking=_signal_marking,
        steps=1500,
        lr=0.1,
    )
    return module


def test_cascade_learns_strength_dependent_routing():
    module = _train_cascade()
    with torch.no_grad():
        strong = module(input_marking={"p_signal": torch.tensor([0.95])})
        weak = module(input_marking={"p_signal": torch.tensor([0.05])})
    assert strong["t_fast_pathway"].item() > strong["t_slow_pathway"].item()
    assert weak["t_slow_pathway"].item() > weak["t_fast_pathway"].item()


# ---------------------------------------------------------------------------
# Interpretability — distilled rule reads in biological terms
# ---------------------------------------------------------------------------


def test_extracted_routing_rule_uses_biological_labels():
    """The interpretability module knows nothing about BPMN. Given the
    trained cascade it must produce a rule whose labels are the
    biological pathway names supplied to add_transition, not internal
    IDs."""
    module = _train_cascade()
    rules = extract_routing_rules(module)
    cascade_rules = [r for r in rules if r.input_place == "p_kinase_active"]
    assert len(cascade_rules) == 1
    rule = cascade_rules[0]
    assert rule.label_above == "fast pathway"
    assert rule.label_below == "slow pathway"
    assert rule.confidence > 0.5


# ---------------------------------------------------------------------------
# Anomaly detection — §7.2 catches off-pathway traces
# ---------------------------------------------------------------------------


def test_cascade_anomaly_detection_flags_mismatched_route():
    """A strong-signal trace that fires the slow pathway is
    biologically inconsistent (we trained the network to expect fast
    pathway for strong signals). The residuals should concentrate on
    the two routing transitions."""
    module = _train_cascade()
    normal = XESTrace(
        attributes={"signal_strength": "0.95"},
        events=[
            XESEvent(name="phosphorylate kinase"),
            XESEvent(name="fast pathway"),
        ],
    )
    anomalous = XESTrace(
        attributes={"signal_strength": "0.95"},
        events=[
            XESEvent(name="phosphorylate kinase"),
            XESEvent(name="slow pathway"),
            XESEvent(name="slow effector activation"),
        ],
    )
    normal_scores = anomaly_score(
        module, normal, attribute_to_marking=_signal_marking
    )
    anomalous_scores = anomaly_score(
        module, anomalous, attribute_to_marking=_signal_marking
    )
    assert sum(anomalous_scores.values()) > sum(normal_scores.values()) + 0.5

    routing_residuals = (
        anomalous_scores["t_fast_pathway"] + anomalous_scores["t_slow_pathway"]
    )
    other_residuals = sum(
        v for t, v in anomalous_scores.items()
        if t not in {"t_fast_pathway", "t_slow_pathway"}
    )
    assert routing_residuals > other_residuals


# ---------------------------------------------------------------------------
# Bisimulation — §7.3 holds for the non-BPMN substrate
# ---------------------------------------------------------------------------


def test_two_isomorphic_cascades_are_bisimilar():
    """Build the same biological cascade twice with disjoint prefixes;
    Phase 2's structural bisimulation check should say they are
    behaviourally equivalent. The mechanism is identical to the BPMN
    case because bisimulation operates on the labelled transition
    system, not on any BPMN structure."""
    a = _signalling_cascade(prefix="a_")
    b = _signalling_cascade(prefix="b_")
    assert are_bisimilar(a, b)
