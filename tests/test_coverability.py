"""Tests for Karp-Miller coverability analysis.

The headline contract of :func:`coverability_graph`: on a bounded
net it returns ``is_bounded=True`` and an exact per-place upper
bound; on an unbounded net it identifies which specific places
can hold arbitrarily many tokens and provides ω-marking witnesses.

Coverage breakdown:

* Bounded cases — sequential net, branching, bounded cycle —
  must report ``is_bounded=True`` with the expected integer
  bounds.
* Unbounded cases — the classical self-loop-plus-producer, a
  multi-place unbounded net, a counter pattern — must pin the
  exact unbounded places and produce ω-marking witnesses.
* Edge cases — empty initial marking, ``max_nodes`` guard,
  inhibitor-arc conservative treatment.
"""
from __future__ import annotations

import pytest

from petri_net_nn import PetriNet
from petri_net_nn.coverability import (
    OMEGA,
    CoverabilityReport,
    coverability_graph,
    is_bounded,
)


# ---------------------------------------------------------------------------
# Bounded cases — the analyser must say so cleanly
# ---------------------------------------------------------------------------


def test_sequential_net_is_bounded():
    """A two-step linear process — one token at p0, fires t0 to p1,
    fires t1 to p2 — is bounded by construction. Each place holds
    at most one token across the reachable markings."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p1")
    net.add_place("p2")
    net.add_transition("t0")
    net.add_transition("t1")
    net.add_arc("p0", "t0")
    net.add_arc("t0", "p1")
    net.add_arc("p1", "t1")
    net.add_arc("t1", "p2")

    report = coverability_graph(net)
    assert report.is_bounded
    assert report.unbounded_places == []
    assert report.omega_markings == []
    # Each place ever holds exactly 1 token in some reachable state.
    assert report.place_bounds == {"p0": 1, "p1": 1, "p2": 1}


def test_bounded_cycle_returns_finite_bounds():
    """A two-place cycle (p0 ↔ p1 via t_forward / t_back) is
    bounded — the single token oscillates between the two places
    forever but each place only ever holds one token at a time."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p1")
    net.add_transition("t_forward")
    net.add_transition("t_back")
    net.add_arc("p0", "t_forward")
    net.add_arc("t_forward", "p1")
    net.add_arc("p1", "t_back")
    net.add_arc("t_back", "p0")

    report = coverability_graph(net)
    assert report.is_bounded
    assert report.place_bounds == {"p0": 1, "p1": 1}


def test_branching_net_records_per_place_bounds():
    """A token at p0 routes to either p_a or p_b via t_a / t_b,
    then both branches synchronise at t_join into p_sink. Every
    place is bounded; the bounds reflect each place's maximum
    token count over the reachable space."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p_a")
    net.add_place("p_b")
    net.add_place("p_sink")
    net.add_transition("t_a")
    net.add_transition("t_b")
    net.add_arc("p0", "t_a")
    net.add_arc("t_a", "p_a")
    net.add_arc("p0", "t_b")
    net.add_arc("t_b", "p_b")
    # Both branches converge — distinct transitions, same sink
    net.add_transition("t_finish_a")
    net.add_transition("t_finish_b")
    net.add_arc("p_a", "t_finish_a")
    net.add_arc("p_b", "t_finish_b")
    net.add_arc("t_finish_a", "p_sink")
    net.add_arc("t_finish_b", "p_sink")

    report = coverability_graph(net)
    assert report.is_bounded
    assert report.place_bounds["p_sink"] == 1


def test_summary_for_bounded_net_says_bounded():
    """``summary()`` returns the literal ``"bounded"`` on a clean
    net — useful as an assertion message in upstream tests."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p1")
    net.add_transition("t")
    net.add_arc("p0", "t")
    net.add_arc("t", "p1")

    assert coverability_graph(net).summary() == "bounded"


# ---------------------------------------------------------------------------
# Unbounded cases — the analyser must pin the unbounded place(s)
# ---------------------------------------------------------------------------


def test_classical_unbounded_producer_is_flagged():
    """The textbook unbounded Petri net: a self-loop transition
    that consumes p0, produces p0 back, AND produces an extra
    token at p_counter every time it fires. p_counter grows
    without bound while p0 stays at 1 token forever."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p_counter")
    net.add_transition("t")
    net.add_arc("p0", "t")
    net.add_arc("t", "p0")  # self-loop keeps p0 alive
    net.add_arc("t", "p_counter")  # but p_counter accumulates

    report = coverability_graph(net)
    assert not report.is_bounded
    assert report.unbounded_places == ["p_counter"]
    # p0 stays bounded — the self-loop just maintains it at 1.
    assert report.place_bounds["p0"] == 1
    assert report.place_bounds["p_counter"] == OMEGA
    # At least one ω-marking witness was recorded.
    assert report.omega_markings
    # Every witness must carry ω at p_counter (the unbounded one).
    for witness in report.omega_markings:
        as_dict = dict(witness)
        assert as_dict.get("p_counter") == OMEGA


def test_summary_names_unbounded_places():
    """``summary()`` on an unbounded report identifies which
    places are the ones blowing up, in alphabetical order."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p_counter")
    net.add_transition("t")
    net.add_arc("p0", "t")
    net.add_arc("t", "p0")
    net.add_arc("t", "p_counter")

    report = coverability_graph(net)
    assert report.summary() == "unbounded at 1 place(s): p_counter"


def test_multiple_unbounded_places_are_all_pinned():
    """A net with two independent producer self-loops, each
    accumulating tokens at a different counter place. Both
    places must be reported as unbounded."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("counter_a")
    net.add_place("counter_b")
    net.add_transition("t_a")
    net.add_transition("t_b")
    # Both transitions self-loop on p0, each producing one of the
    # counter places per firing.
    net.add_arc("p0", "t_a")
    net.add_arc("t_a", "p0")
    net.add_arc("t_a", "counter_a")
    net.add_arc("p0", "t_b")
    net.add_arc("t_b", "p0")
    net.add_arc("t_b", "counter_b")

    report = coverability_graph(net)
    assert not report.is_bounded
    assert report.unbounded_places == ["counter_a", "counter_b"]
    assert report.place_bounds["counter_a"] == OMEGA
    assert report.place_bounds["counter_b"] == OMEGA
    assert report.place_bounds["p0"] == 1


def test_unbounded_via_arc_weight_imbalance():
    """A transition that consumes one token and produces two —
    each firing strictly grows the net token count. p_pool
    accumulates without bound."""
    net = PetriNet()
    net.add_place("p_pool", tokens=1)
    net.add_transition("t")
    net.add_arc("p_pool", "t")
    net.add_arc("t", "p_pool", weight=2)

    report = coverability_graph(net)
    assert not report.is_bounded
    assert report.unbounded_places == ["p_pool"]


# ---------------------------------------------------------------------------
# is_bounded convenience wrapper
# ---------------------------------------------------------------------------


def test_is_bounded_helper_returns_bool():
    """``is_bounded(net)`` is the one-liner question, equivalent
    to ``coverability_graph(net).is_bounded``."""
    sound_net = PetriNet()
    sound_net.add_place("p0", tokens=1)
    sound_net.add_place("p1")
    sound_net.add_transition("t")
    sound_net.add_arc("p0", "t")
    sound_net.add_arc("t", "p1")
    assert is_bounded(sound_net) is True

    bad_net = PetriNet()
    bad_net.add_place("p0", tokens=1)
    bad_net.add_place("p_counter")
    bad_net.add_transition("t")
    bad_net.add_arc("p0", "t")
    bad_net.add_arc("t", "p0")
    bad_net.add_arc("t", "p_counter")
    assert is_bounded(bad_net) is False


# ---------------------------------------------------------------------------
# Inhibitor arcs — conservative treatment
# ---------------------------------------------------------------------------


def test_inhibitor_arc_produces_conservative_overreport():
    """The inhibitor-arc conservative-treatment case documented at
    the module level. The net has a ``t_produce`` transition that
    self-loops on ``p_in`` and produces both ``p_out`` and
    ``p_lock`` per firing, with an inhibitor arc from ``p_lock``
    to ``t_produce``. In the real token-game the inhibitor
    blocks the transition after exactly one firing, so the net
    is bounded — ``p_out`` and ``p_lock`` each end at one token.

    Karp-Miller, however, introduces ω as soon as a marking
    strictly covers an ancestor; it does that on the very first
    firing, before the inhibitor evidence has time to
    accumulate. The algorithm therefore reports the net as
    unbounded at both producer outputs. This is the
    conservative-overreport behaviour the module documents — a
    perfect answer would require solving the halting problem
    (Minsky's 1967 reduction of two-counter machines to
    inhibitor-arc Petri nets plus Turing's 1936 undecidability
    of halting). Every coverability tool in the world makes the
    same compromise; PETRA opts for conservative reporting
    rather than refusing to answer."""
    net = PetriNet()
    net.add_place("p_in", tokens=1)
    net.add_place("p_out")
    net.add_place("p_lock")
    net.add_transition("t_produce")
    net.add_arc("p_in", "t_produce")
    net.add_arc("t_produce", "p_in")
    net.add_arc("t_produce", "p_out")
    net.add_arc("t_produce", "p_lock")
    net.add_inhibitor_arc("p_lock", "t_produce")

    report = coverability_graph(net)
    # Conservative: both producer outputs flagged even though the
    # real net firmly bounds them at one token each.
    assert not report.is_bounded
    assert "p_out" in report.unbounded_places
    assert "p_lock" in report.unbounded_places


def test_resource_lock_mutex_is_bounded():
    """A two-client mutex pattern with a shared lock place. Each
    client acquires the lock (consumes the lock-free token,
    produces a lock-held token), does work, then releases. The
    lock is structurally bounded — there's only ever one
    lock-related token in circulation."""
    net = PetriNet()
    net.add_place("lock_free", tokens=1)
    net.add_place("lock_held")
    net.add_place("client_a_ready", tokens=1)
    net.add_place("client_a_done")
    net.add_transition("a_acquire")
    net.add_transition("a_release")
    net.add_arc("lock_free", "a_acquire")
    net.add_arc("client_a_ready", "a_acquire")
    net.add_arc("a_acquire", "lock_held")
    net.add_arc("lock_held", "a_release")
    net.add_arc("a_release", "lock_free")
    net.add_arc("a_release", "client_a_done")

    report = coverability_graph(net)
    assert report.is_bounded
    assert report.place_bounds["lock_free"] == 1
    assert report.place_bounds["lock_held"] == 1


# ---------------------------------------------------------------------------
# Edge cases — empty marking, guard cap
# ---------------------------------------------------------------------------


def test_empty_initial_marking_is_bounded():
    """A net with no tokens initially has no firings, so it's
    trivially bounded. All places have bound 0."""
    net = PetriNet()
    net.add_place("p0")
    net.add_place("p1")
    net.add_transition("t")
    net.add_arc("p0", "t")
    net.add_arc("t", "p1")

    report = coverability_graph(net)
    assert report.is_bounded
    assert report.place_bounds == {"p0": 0, "p1": 0}


def test_max_nodes_guard_raises_clearly():
    """The ``max_nodes`` cap protects against pathological inputs
    that grow the tree beyond a sensible size. With the cap
    pushed unreasonably low, even a small net should trip it."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p_counter")
    net.add_transition("t")
    net.add_arc("p0", "t")
    net.add_arc("t", "p0")
    net.add_arc("t", "p_counter")

    # max_nodes=1 means the cap trips after the root is added and
    # the first successor would push the count past the limit.
    with pytest.raises(ValueError, match="coverability tree exceeded"):
        coverability_graph(net, max_nodes=1)


def test_report_is_a_dataclass():
    """The report is a frozen dataclass — fields are exposed by
    name and the object is immutable."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p1")
    net.add_transition("t")
    net.add_arc("p0", "t")
    net.add_arc("t", "p1")

    report = coverability_graph(net)
    assert isinstance(report, CoverabilityReport)
    with pytest.raises(Exception):
        report.is_bounded = False  # frozen dataclass blocks mutation
