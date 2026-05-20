"""Tests for the Aalst soundness checker and deadlock localiser.

The three soundness conditions — option to complete, proper
completion, no dead transitions — get one positive test apiece
(sound net passes) and one negative test apiece (broken net
correctly diagnoses the matching condition). The deadlock
localiser gets a dedicated test pinning a constructed dead-end.

These checks are foundational: every scenario in ``examples/``
should pass soundness, and the test suite also asserts that on
a representative subset.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from petri_net_nn import (
    PetriNet,
    SoundnessReport,
    check_soundness,
    find_deadlocks,
    parse_bpmn,
)


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Soundness — positive cases
# ---------------------------------------------------------------------------


def test_simple_sequence_is_sound():
    """A linear two-step process — initial token at p0, fires t0 to
    p1, fires t1 to p2 — should pass all three soundness checks."""
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

    report = check_soundness(net)
    assert report.is_sound, report.summary()
    assert report.incomplete_markings == []
    assert report.lingering_token_markings == []
    assert report.dead_transitions == []


def test_xor_branching_net_is_sound():
    """A simple BPMN XOR fixture has two routing branches that
    both reach the sink. Both branches are reachable, both reach
    the final marking, so the net is sound."""
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    report = check_soundness(net)
    assert report.is_sound, report.summary()


def test_summary_for_sound_net_says_sound():
    """The summary() helper produces a one-liner suitable for log
    output; on a clean net it just says ``"sound"``."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p1")
    net.add_transition("t0")
    net.add_arc("p0", "t0")
    net.add_arc("t0", "p1")
    assert check_soundness(net).summary() == "sound"


# ---------------------------------------------------------------------------
# Soundness — negative cases (one per failing condition)
# ---------------------------------------------------------------------------


def test_dead_transition_is_flagged():
    """A transition that's structurally well-formed but never
    enabled — its input place is never marked — should appear in
    ``dead_transitions``. We construct this by giving t_dead an
    input place that nobody ever populates."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p1")
    net.add_place("p_isolated")  # never receives a token
    net.add_place("p_dead_out")
    net.add_transition("t_main")
    net.add_transition("t_dead")
    net.add_arc("p0", "t_main")
    net.add_arc("t_main", "p1")
    # t_dead is well-formed but its input is never marked.
    net.add_arc("p_isolated", "t_dead")
    net.add_arc("t_dead", "p_dead_out")

    report = check_soundness(net, final_marking=frozenset({("p1", 1)}))
    assert not report.is_sound
    assert report.dead_transitions == ["t_dead"]


def test_marking_that_cannot_complete_is_flagged():
    """Build a small net where one path leads to a marking from
    which the intended final is unreachable. We model that as a
    forked initial: from p0, t_good goes to the sink; t_bad goes
    to an off-path place that has no exit. After t_bad fires, the
    sink is structurally unreachable — option-to-complete fails."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p_sink")
    net.add_place("p_offpath")  # post-t_bad, no outgoing
    net.add_transition("t_good")
    net.add_transition("t_bad")
    net.add_arc("p0", "t_good")
    net.add_arc("t_good", "p_sink")
    net.add_arc("p0", "t_bad")
    net.add_arc("t_bad", "p_offpath")

    # Explicit final: one token at p_sink.
    report = check_soundness(net, final_marking=frozenset({("p_sink", 1)}))
    assert not report.is_sound
    # The post-t_bad marking can't reach p_sink — that's the
    # incomplete one we expect to be flagged.
    flagged = report.incomplete_markings
    assert any(("p_offpath", 1) in m for m in flagged)


def test_lingering_tokens_at_completion_are_flagged():
    """Two parallel branches that don't synchronise. p0 forks
    into p_a and p_b via t_split; t_done consumes only p_a to
    fill the sink. After t_done fires, the sink has its token
    but p_b still holds a leftover — proper completion fails."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p_a")
    net.add_place("p_b")
    net.add_place("p_sink")
    net.add_transition("t_split")
    net.add_transition("t_done")
    net.add_arc("p0", "t_split")
    net.add_arc("t_split", "p_a")
    net.add_arc("t_split", "p_b")
    net.add_arc("p_a", "t_done")
    net.add_arc("t_done", "p_sink")

    report = check_soundness(net, final_marking=frozenset({("p_sink", 1)}))
    assert not report.is_sound
    # Should flag at least one lingering-token marking: the one
    # where p_sink has a token AND p_b still has a token.
    assert report.lingering_token_markings, (
        "expected a lingering-token diagnosis"
    )
    assert any(
        ("p_sink", 1) in m and ("p_b", 1) in m
        for m in report.lingering_token_markings
    )


# ---------------------------------------------------------------------------
# Final-marking detection
# ---------------------------------------------------------------------------


def test_default_final_marking_picks_unique_sink():
    """When the net has one sink place (no outgoing arcs), the
    default heuristic uses it as the final marking — no explicit
    argument needed."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p_sink")
    net.add_transition("t")
    net.add_arc("p0", "t")
    net.add_arc("t", "p_sink")

    report = check_soundness(net)
    assert report.final_marking == frozenset({("p_sink", 1)})
    assert report.is_sound


def test_explicit_final_marking_overrides_default():
    """The caller can supply ``final_marking`` directly, e.g. to
    declare that the intended completion has two sink tokens
    rather than one."""
    net = PetriNet()
    net.add_place("p0", tokens=2)
    net.add_place("p_sink")
    net.add_transition("t")
    net.add_arc("p0", "t")
    net.add_arc("t", "p_sink")

    # Two firings expected; final has two sink tokens.
    final = frozenset({("p_sink", 2)})
    report = check_soundness(net, final_marking=final)
    assert report.final_marking == final
    assert report.is_sound


def test_no_sink_places_raises_helpful_error():
    """A net with no sink places (every place has an outgoing arc
    — possible in cyclic-only nets) needs an explicit final
    marking; calling without one should raise with a clear
    message."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p1")
    net.add_transition("t_forward")
    net.add_transition("t_back")
    # cycle: p0 -> t_forward -> p1 -> t_back -> p0
    net.add_arc("p0", "t_forward")
    net.add_arc("t_forward", "p1")
    net.add_arc("p1", "t_back")
    net.add_arc("t_back", "p0")

    with pytest.raises(ValueError, match="sink"):
        check_soundness(net)


# ---------------------------------------------------------------------------
# Deadlock localisation
# ---------------------------------------------------------------------------


def test_no_deadlocks_in_sound_net():
    """A sound sequential net has no deadlocks — every reachable
    marking either has an enabled successor or is the final
    marking."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p1")
    net.add_transition("t")
    net.add_arc("p0", "t")
    net.add_arc("t", "p1")
    assert find_deadlocks(net) == []


def test_deadlock_marking_is_pinned_to_specific_state():
    """Reuse the option-to-complete-failure net: after t_bad
    fires, the marking ``{p_offpath: 1}`` has no enabled successors
    and isn't the final marking. ``find_deadlocks`` should pin it."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p_sink")
    net.add_place("p_offpath")
    net.add_transition("t_good")
    net.add_transition("t_bad")
    net.add_arc("p0", "t_good")
    net.add_arc("t_good", "p_sink")
    net.add_arc("p0", "t_bad")
    net.add_arc("t_bad", "p_offpath")

    deadlocks = find_deadlocks(
        net, final_marking=frozenset({("p_sink", 1)})
    )
    assert deadlocks == [frozenset({("p_offpath", 1)})]


def test_final_marking_is_not_a_deadlock():
    """The intended final marking is excluded from the deadlock
    list — it's *supposed* to have no successors."""
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p1")
    net.add_transition("t")
    net.add_arc("p0", "t")
    net.add_arc("t", "p1")
    # The final marking {p1: 1} has no enabled transitions but
    # is the intended completion; not a deadlock.
    assert find_deadlocks(net) == []
