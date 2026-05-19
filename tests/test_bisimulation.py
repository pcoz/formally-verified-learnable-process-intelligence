"""Tests for the bisimulation checker (Phase 2 of docs/ROADMAP.md).

The first group of tests pins the algorithm itself — reachability graph
construction, partition refinement, the four canonical
bisimilar / not-bisimilar shapes.

The final test is the §7.3 load-bearing one: two bisimilar nets,
trained on the same XES data, converge to forward-pass outputs that
agree on the bisimulation correspondence — "structural equivalence
verified by the compiler, behavioural equivalence confirmed by
identical learned weights" (or in our case, identical learned
functions; weights are random-init-noise apart).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    PetriNet,
    PetriNetModule,
    XESEvent,
    XESTrace,
    are_bisimilar,
    bisimulation_equivalence_classes,
    parse_bpmn,
    parse_xes,
    reachability_graph,
    train_on_traces,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _sequence(*, labels: tuple[str, ...], prefix: str = "") -> PetriNet:
    """Build a linear net p0 -t0-> p1 -t1-> ... -t{n-1}-> p_n.
    The token starts at p0; each transition takes its label from
    ``labels``. ``prefix`` namespaces the IDs so two structurally
    identical nets can live side-by-side with disjoint names."""
    net = PetriNet()
    n = len(labels)
    for i in range(n + 1):
        net.add_place(f"{prefix}p{i}", tokens=(1 if i == 0 else 0))
    for i, label in enumerate(labels):
        tid = f"{prefix}t{i}"
        net.add_transition(tid, label=label)
        net.add_arc(f"{prefix}p{i}", tid)
        net.add_arc(tid, f"{prefix}p{i+1}")
    return net


# ---------------------------------------------------------------------------
# Reachability graph
# ---------------------------------------------------------------------------


def test_reachability_graph_simple_sequence():
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    lts = reachability_graph(net)
    assert len(lts.states) == 2
    assert len(lts.transitions) == 1
    initial_token_place, _ = next(iter(lts.initial))
    assert initial_token_place == "p_f1"


def test_reachability_graph_xor_branch_explores_both_paths():
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    lts = reachability_graph(net)
    assert len(lts.states) >= 5
    label_set = {label for _, label, _ in lts.transitions}
    assert "Path A" in label_set
    assert "Path B" in label_set


def test_reachability_graph_max_states_raises():
    """Build a deliberately unbounded net: a transition that consumes
    nothing and produces a token on a fresh place each fire would be
    impossible without dynamic places, so we instead build a small net
    and pass a tiny cap to trigger the guard."""
    net = parse_bpmn(FIXTURES / "approval.bpmn")
    with pytest.raises(ValueError, match="exceeded"):
        reachability_graph(net, max_states=2)


# ---------------------------------------------------------------------------
# Bisimulation algorithm
# ---------------------------------------------------------------------------


def test_net_is_bisimilar_to_itself():
    net = parse_bpmn(FIXTURES / "simple_sequence.bpmn")
    assert are_bisimilar(net, net)


def test_isomorphic_renamings_are_bisimilar():
    a = _sequence(labels=("X", "Y"), prefix="a_")
    b = _sequence(labels=("X", "Y"), prefix="b_")
    assert are_bisimilar(a, b)


def test_different_labels_are_not_bisimilar():
    a = _sequence(labels=("X", "Y"), prefix="a_")
    b = _sequence(labels=("X", "Z"), prefix="b_")
    assert not are_bisimilar(a, b)


def test_different_branching_is_not_bisimilar():
    """Two paths of length 2 vs one path of length 2 — net B can do
    something net A cannot mirror after the initial X step."""
    branching = PetriNet()
    branching.add_place("p0", tokens=1)
    branching.add_place("p1")
    branching.add_place("p2a")
    branching.add_place("p2b")
    branching.add_transition("t_x", label="X")
    branching.add_transition("t_a", label="A")
    branching.add_transition("t_b", label="B")
    branching.add_arc("p0", "t_x")
    branching.add_arc("t_x", "p1")
    branching.add_arc("p1", "t_a")
    branching.add_arc("t_a", "p2a")
    branching.add_arc("p1", "t_b")
    branching.add_arc("t_b", "p2b")

    single = _sequence(labels=("X", "A"), prefix="s_")
    assert not are_bisimilar(branching, single)


def test_redundant_parallel_transitions_with_same_label_are_bisimilar():
    """Net A has one X-labelled transition; net B has two X-labelled
    transitions in parallel (same source, same target). They're
    structurally distinct but behaviourally equivalent."""
    a = _sequence(labels=("X",), prefix="a_")

    b = PetriNet()
    b.add_place("p0", tokens=1)
    b.add_place("p1")
    b.add_transition("t_x1", label="X")
    b.add_transition("t_x2", label="X")
    b.add_arc("p0", "t_x1")
    b.add_arc("t_x1", "p1")
    b.add_arc("p0", "t_x2")
    b.add_arc("t_x2", "p1")

    assert are_bisimilar(a, b)


def test_equivalence_classes_for_sequential_net():
    net = _sequence(labels=("X", "Y"), prefix="")
    classes = bisimulation_equivalence_classes(net)
    assert len(classes) == 3


# ---------------------------------------------------------------------------
# §7.3 integration — bisimilar nets converge to the same trained function
# ---------------------------------------------------------------------------


def test_two_bisimilar_xor_nets_learn_the_same_function():
    """Build two structurally isomorphic XOR nets (same labels, disjoint
    IDs), verify the algorithm says they're bisimilar, then train each
    on the same XES log under the same seed. After training, the two
    compiled networks should agree on Path A / Path B routing for any
    risk_score input — i.e. they compute the same function."""

    def make_xor_net(prefix: str) -> PetriNet:
        net = PetriNet()
        net.add_place(f"{prefix}p_in", tokens=1)
        net.add_place(f"{prefix}p_A")
        net.add_place(f"{prefix}p_B")
        net.add_transition(f"{prefix}t_A", label="Path A")
        net.add_transition(f"{prefix}t_B", label="Path B")
        net.add_arc(f"{prefix}p_in", f"{prefix}t_A")
        net.add_arc(f"{prefix}t_A", f"{prefix}p_A")
        net.add_arc(f"{prefix}p_in", f"{prefix}t_B")
        net.add_arc(f"{prefix}t_B", f"{prefix}p_B")
        return net

    net1 = make_xor_net("n1_")
    net2 = make_xor_net("n2_")
    assert are_bisimilar(net1, net2)

    traces = parse_xes(FIXTURES / "xor_log.xes")

    torch.manual_seed(0)
    module1 = PetriNetModule(net1)
    torch.manual_seed(0)
    module2 = PetriNetModule(net2)

    def marking_for(prefix):
        def to_marking(trace):
            return {f"{prefix}p_in": float(trace.attributes["risk_score"])}
        return to_marking

    train_on_traces(
        module1, traces, attribute_to_marking=marking_for("n1_"),
        steps=1500, lr=0.1,
    )
    train_on_traces(
        module2, traces, attribute_to_marking=marking_for("n2_"),
        steps=1500, lr=0.1,
    )

    test_inputs = torch.linspace(0.0, 1.0, 11)

    with torch.no_grad():
        out1 = module1(input_marking={"n1_p_in": test_inputs})
        out2 = module2(input_marking={"n2_p_in": test_inputs})

    diff_A = (out1["n1_t_A"] - out2["n2_t_A"]).abs().max().item()
    diff_B = (out1["n1_t_B"] - out2["n2_t_B"]).abs().max().item()
    assert diff_A < 0.05, f"Path A diverged by {diff_A}"
    assert diff_B < 0.05, f"Path B diverged by {diff_B}"
