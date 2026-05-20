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


# ---------------------------------------------------------------------------
# Weak bisimulation — Phase 11
#
# The strong checker above rejects refactorings that introduce an
# internal silent transition (logging step, no-op gate, internal
# handoff). The weak checker collapses such transitions before the
# match — which is what makes the cost-ranked refactoring story
# actually work for real-world redesigns, where almost every variant
# differs from the reference by some τ-step the modeller wants to
# treat as invisible.
# ---------------------------------------------------------------------------


from petri_net_nn import (  # noqa: E402
    are_weakly_bisimilar,
    weak_bisimulation_equivalence_classes,
)


def _sequence_with_silent(
    *,
    labels: tuple[str, ...],
    silent_after: int | None = None,
    prefix: str = "",
) -> PetriNet:
    """Build a linear net like ``_sequence`` but optionally insert
    one silent (τ) transition immediately after position
    ``silent_after`` in the original transition sequence. The
    silent transition consumes the place at ``silent_after`` and
    produces a *new* fresh place that the next visible transition
    then consumes.

    Letting tests parameterise the silent insertion point keeps the
    fixtures one-liners — there are several cases below that need
    "the same visible sequence but with a τ-step somewhere in the
    middle"."""
    net = PetriNet()
    n = len(labels)
    # Build the regular place / transition skeleton first.
    for i in range(n + 1):
        net.add_place(f"{prefix}p{i}", tokens=(1 if i == 0 else 0))
    for i, label in enumerate(labels):
        tid = f"{prefix}t{i}"
        net.add_transition(tid, label=label)
        net.add_arc(f"{prefix}p{i}", tid)
        net.add_arc(tid, f"{prefix}p{i+1}")

    # If requested, splice in a silent transition between p{k+1}
    # (the output of the silent_after'th visible transition) and
    # the input of the next visible one — i.e. replace the direct
    # path p{k+1} → t{k+1} with p{k+1} → t_tau → p{k+1}_post → t{k+1}.
    if silent_after is not None:
        if not (0 <= silent_after < n - 1):
            raise ValueError(
                "silent_after must be a valid insertion point "
                "between two visible transitions"
            )
        k = silent_after
        intermediate = f"{prefix}p{k+1}_post"
        net.add_place(intermediate)
        tau_id = f"{prefix}t_tau_{k}"
        net.add_transition(tau_id, label="τ", silent=True)
        # The silent transition consumes the intermediate marking
        # and forwards to a fresh place, which becomes the input
        # of the next visible transition. We rewire t{k+1}'s input
        # arc to the new place.
        net.add_arc(f"{prefix}p{k+1}", tau_id)
        net.add_arc(tau_id, intermediate)
        # Remove the original p{k+1} -> t{k+1} arc and replace it
        # with intermediate -> t{k+1}. flow is a set so just
        # discard + add.
        original_arc = (f"{prefix}p{k+1}", f"{prefix}t{k+1}")
        net.flow.discard(original_arc)
        net.add_arc(intermediate, f"{prefix}t{k+1}")

    return net


def test_strong_bisimilar_nets_are_also_weakly_bisimilar():
    """Sanity check: weak bisimulation accepts anything strong
    bisimulation accepts. The matching condition for visible actions
    is looser under weak (τ-paths allowed before and after), but
    direct ``→^a`` matches are a degenerate case of the weak ``⇒^a``
    relation, so the strong-bisim acceptances carry through."""
    a = _sequence(labels=("X", "Y"), prefix="a_")
    b = _sequence(labels=("X", "Y"), prefix="b_")
    assert are_bisimilar(a, b)
    assert are_weakly_bisimilar(a, b)


def test_silent_transition_inserted_is_weakly_but_not_strongly_bisimilar():
    """The load-bearing claim of Phase 11: a refactoring that adds
    an internal silent step (a logging hook, a no-op routing gate,
    a structural artefact) is rejected by strong bisimulation but
    accepted by weak. This is the case the strong checker is too
    strict for."""
    # Net A: X → Y (two visible transitions in sequence).
    a = _sequence(labels=("X", "Y"), prefix="a_")
    # Net B: X → τ → Y (the same visible sequence with a silent
    # transition spliced between).
    b = _sequence_with_silent(
        labels=("X", "Y"), silent_after=0, prefix="b_",
    )

    # Strong rejects — the τ-step is a labelled edge in net B that
    # doesn't appear in net A.
    assert not are_bisimilar(a, b)
    # Weak accepts — τ-edges are collapsed before comparison.
    assert are_weakly_bisimilar(a, b)


def test_silent_self_loop_is_invisible_to_weak_bisimulation():
    """A τ self-loop on some state lets the state τ-step back to
    itself indefinitely. Because the saturation step makes every
    state self-τ-reachable anyway, the explicit self-loop should
    not affect the equivalence."""
    base = _sequence(labels=("X",), prefix="a_")

    # Same visible behaviour, but with a silent self-loop hanging
    # off the start state via a "ping" transition that consumes
    # and re-produces p0.
    with_loop = PetriNet()
    with_loop.add_place("p0", tokens=1)
    with_loop.add_place("p1")
    with_loop.add_transition("t_x", label="X")
    with_loop.add_transition("t_ping", label="τ", silent=True)
    with_loop.add_arc("p0", "t_x")
    with_loop.add_arc("t_x", "p1")
    with_loop.add_arc("p0", "t_ping")
    with_loop.add_arc("t_ping", "p0")

    assert are_weakly_bisimilar(base, with_loop)


def test_tau_chains_of_different_lengths_are_weakly_bisimilar():
    """Two τ-chains compress to the same observable behaviour even
    when their lengths differ. The saturation step folds an
    arbitrary number of consecutive τ-edges into a single weak
    transition."""
    # Net A: X → τ → Y (one silent step between visibles).
    a = _sequence_with_silent(
        labels=("X", "Y"), silent_after=0, prefix="a_",
    )

    # Net B: X → τ → τ → Y (two silent steps).
    b = PetriNet()
    b.add_place("b_p0", tokens=1)
    b.add_place("b_p1")
    b.add_place("b_p1a")
    b.add_place("b_p1b")
    b.add_place("b_p2")
    b.add_transition("b_tx", label="X")
    b.add_transition("b_tau1", label="τ", silent=True)
    b.add_transition("b_tau2", label="τ", silent=True)
    b.add_transition("b_ty", label="Y")
    b.add_arc("b_p0", "b_tx")
    b.add_arc("b_tx", "b_p1")
    b.add_arc("b_p1", "b_tau1")
    b.add_arc("b_tau1", "b_p1a")
    b.add_arc("b_p1a", "b_tau2")
    b.add_arc("b_tau2", "b_p1b")
    b.add_arc("b_p1b", "b_ty")
    b.add_arc("b_ty", "b_p2")

    assert are_weakly_bisimilar(a, b)


def test_weak_bisimulation_still_discriminates_visible_differences():
    """Adding τ-steps must not let arbitrary nets get matched —
    visible-action discrimination still has to work. A net that
    does ``X then Y`` versus a net that does ``X then Z`` must
    remain non-bisimilar under weak too."""
    a = _sequence(labels=("X", "Y"), prefix="a_")
    b = _sequence(labels=("X", "Z"), prefix="b_")
    assert not are_weakly_bisimilar(a, b)


def test_weak_bisimulation_distinguishes_branching():
    """Branching behaviour at a state survives τ-collapse. A net
    that can do ``X → A`` or ``X → B`` from the same intermediate
    state is *not* weakly bisimilar to a net that can only do
    ``X → A`` — even if you sprinkle τ-steps into either."""
    # The shape that's branchy at p1.
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
    assert not are_weakly_bisimilar(branching, single)


def test_weak_equivalence_classes_collapse_silent_intermediate_states():
    """``weak_bisimulation_equivalence_classes`` is to weak what
    ``bisimulation_equivalence_classes`` is to strong: a partition
    of reachable markings into "same observable future" buckets.
    A marking that lives only on a silent transition's τ-output
    should land in the same equivalence class as the marking that
    fed into the τ — they exhibit the same visible future."""
    # Net: X → τ → Y. The reachability graph has four markings,
    # but the τ-collapse identifies two of them — the post-τ state
    # is observably indistinguishable from the post-X-pre-τ state
    # because both have the same visible future (just Y).
    net = _sequence_with_silent(
        labels=("X", "Y"), silent_after=0, prefix="",
    )
    classes = weak_bisimulation_equivalence_classes(net)
    # Under strong bisim we'd see four distinct classes (one per
    # marking). Under weak, the τ-step is collapsed, so the
    # post-X-pre-τ marking and the post-τ marking merge.
    assert len(classes) == 3
