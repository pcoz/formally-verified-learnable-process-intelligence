"""Tests for the PetriNet data structure and its token-game semantics."""
from __future__ import annotations

import pytest

from petri_net_nn import PetriNet


def test_add_place_and_transition_and_arc():
    net = PetriNet()
    net.add_place("p1", tokens=1)
    net.add_place("p2")
    net.add_transition("t1", label="step")
    net.add_arc("p1", "t1")
    net.add_arc("t1", "p2")
    assert net.places == {"p1", "p2"}
    assert net.transitions == {"t1"}
    assert net.flow == {("p1", "t1"), ("t1", "p2")}
    assert net.initial_marking == {"p1": 1}
    assert net.transition_labels["t1"] == "step"


def test_add_arc_rejects_place_to_place():
    net = PetriNet()
    net.add_place("p1")
    net.add_place("p2")
    with pytest.raises(ValueError, match="not place->transition or transition->place"):
        net.add_arc("p1", "p2")


def test_add_arc_rejects_transition_to_transition():
    net = PetriNet()
    net.add_transition("t1")
    net.add_transition("t2")
    with pytest.raises(ValueError):
        net.add_arc("t1", "t2")


def test_add_arc_rejects_unknown_node():
    net = PetriNet()
    net.add_place("p1")
    with pytest.raises(ValueError):
        net.add_arc("p1", "t_does_not_exist")


def test_preset_and_postset():
    net = PetriNet()
    for p in ("p_in_A", "p_in_B", "p_out"):
        net.add_place(p)
    net.add_transition("t_merge")
    net.add_arc("p_in_A", "t_merge")
    net.add_arc("p_in_B", "t_merge")
    net.add_arc("t_merge", "p_out")
    assert net.preset("t_merge") == {"p_in_A", "p_in_B"}
    assert net.postset("t_merge") == {"p_out"}
    assert net.preset("p_out") == {"t_merge"}


def test_is_enabled_requires_token_in_every_input_place():
    net = PetriNet()
    for p in ("a", "b", "c"):
        net.add_place(p)
    net.add_transition("t")
    net.add_arc("a", "t")
    net.add_arc("b", "t")
    net.add_arc("t", "c")
    assert not net.is_enabled("t", {})
    assert not net.is_enabled("t", {"a": 1})
    assert not net.is_enabled("t", {"b": 1})
    assert net.is_enabled("t", {"a": 1, "b": 1})


def test_fire_moves_tokens_per_petri_net_rule():
    net = PetriNet()
    for p in ("a", "b", "c"):
        net.add_place(p)
    net.add_transition("t")
    net.add_arc("a", "t")
    net.add_arc("b", "t")
    net.add_arc("t", "c")
    new = net.fire("t", {"a": 1, "b": 1})
    assert new == {"c": 1}


def test_fire_raises_if_not_enabled():
    net = PetriNet()
    net.add_place("a")
    net.add_place("b")
    net.add_transition("t")
    net.add_arc("a", "t")
    net.add_arc("t", "b")
    with pytest.raises(ValueError, match="not enabled"):
        net.fire("t", {})


def test_validate_flags_dangling_transition():
    net = PetriNet()
    net.add_transition("orphan")
    issues = net.validate()
    assert any("orphan" in issue and "input" in issue for issue in issues)
    assert any("orphan" in issue and "output" in issue for issue in issues)


def test_validate_returns_empty_for_well_formed_net():
    net = PetriNet()
    net.add_place("a", tokens=1)
    net.add_place("b")
    net.add_transition("t")
    net.add_arc("a", "t")
    net.add_arc("t", "b")
    assert net.validate() == []


# ---------------------------------------------------------------------------
# Phase 9 — multi-token markings (arc multiplicities)
# ---------------------------------------------------------------------------


def test_arc_default_weight_is_one():
    net = PetriNet()
    net.add_place("a")
    net.add_place("b")
    net.add_transition("t")
    net.add_arc("a", "t")
    net.add_arc("t", "b")
    assert net.weight("a", "t") == 1
    assert net.weight("t", "b") == 1
    assert net.arc_multiplicities == {}


def test_arc_explicit_weight_stored():
    net = PetriNet()
    net.add_place("a")
    net.add_place("b")
    net.add_transition("t")
    net.add_arc("a", "t", weight=3)
    net.add_arc("t", "b", weight=2)
    assert net.weight("a", "t") == 3
    assert net.weight("t", "b") == 2


def test_add_arc_rejects_non_positive_weight():
    net = PetriNet()
    net.add_place("a")
    net.add_transition("t")
    with pytest.raises(ValueError, match="positive integer"):
        net.add_arc("a", "t", weight=0)
    with pytest.raises(ValueError, match="positive integer"):
        net.add_arc("a", "t", weight=-2)


def test_is_enabled_respects_input_arc_weight():
    """A transition with input weight 3 needs 3 tokens at the input
    place before it is enabled — not just 1."""
    net = PetriNet()
    net.add_place("buffer")
    net.add_place("out")
    net.add_transition("consume_batch")
    net.add_arc("buffer", "consume_batch", weight=3)
    net.add_arc("consume_batch", "out")
    assert not net.is_enabled("consume_batch", {"buffer": 2})
    assert net.is_enabled("consume_batch", {"buffer": 3})
    assert net.is_enabled("consume_batch", {"buffer": 7})


def test_fire_consumes_and_produces_according_to_arc_weights():
    net = PetriNet()
    net.add_place("buffer")
    net.add_place("packaged")
    net.add_transition("box_three")
    net.add_arc("buffer", "box_three", weight=3)
    net.add_arc("box_three", "packaged")
    new = net.fire("box_three", {"buffer": 5})
    assert new == {"buffer": 2, "packaged": 1}


def test_fire_with_output_arc_weight_produces_multiple_tokens():
    """One firing of a transition with output weight 2 produces 2
    tokens at the output place."""
    net = PetriNet()
    net.add_place("trigger", tokens=1)
    net.add_place("clones")
    net.add_transition("spawn_pair")
    net.add_arc("trigger", "spawn_pair")
    net.add_arc("spawn_pair", "clones", weight=2)
    new = net.fire("spawn_pair", net.initial_marking)
    assert new == {"clones": 2}


def test_capacity_three_buffer_scenario():
    """Producer fires three times to fill a buffer; consumer (weight 3)
    can fire exactly once and the buffer is empty afterwards."""
    net = PetriNet()
    net.add_place("supply", tokens=10)
    net.add_place("buffer")
    net.add_place("finished")
    net.add_transition("produce")
    net.add_transition("consume_batch")
    net.add_arc("supply", "produce")
    net.add_arc("produce", "buffer")
    net.add_arc("buffer", "consume_batch", weight=3)
    net.add_arc("consume_batch", "finished")

    marking = dict(net.initial_marking)
    assert not net.is_enabled("consume_batch", marking)

    marking = net.fire("produce", marking)
    marking = net.fire("produce", marking)
    assert not net.is_enabled("consume_batch", marking)

    marking = net.fire("produce", marking)
    assert net.is_enabled("consume_batch", marking)

    marking = net.fire("consume_batch", marking)
    assert "buffer" not in marking
    assert marking["finished"] == 1
    assert marking["supply"] == 7


def test_validate_flags_orphan_multiplicity():
    """An arc multiplicity recorded for an arc that's not in the flow
    relation is a structural inconsistency — should be caught."""
    net = PetriNet()
    net.add_place("a")
    net.add_transition("t")
    net.add_arc("a", "t")
    net.add_place("b")
    net.add_arc("t", "b")
    net.arc_multiplicities[("a", "ghost")] = 2
    issues = net.validate()
    assert any("no such arc" in issue for issue in issues)


# ---------------------------------------------------------------------------
# Phase 9 — inhibitor arcs (fire only when place is empty)
# ---------------------------------------------------------------------------


def _mutex_net() -> PetriNet:
    """Two transitions guarded by inhibitor arcs against a shared
    'critical_section' place. Whichever fires first claims the section;
    the other becomes disabled until the section clears."""
    net = PetriNet()
    net.add_place("p_a_ready", tokens=1)
    net.add_place("p_b_ready", tokens=1)
    net.add_place("p_critical")          # holds a token while either A or B is inside
    net.add_place("p_a_done")
    net.add_place("p_b_done")
    net.add_transition("t_enter_a")
    net.add_transition("t_enter_b")
    net.add_arc("p_a_ready", "t_enter_a")
    net.add_arc("t_enter_a", "p_critical")
    net.add_arc("t_enter_a", "p_a_done")
    net.add_arc("p_b_ready", "t_enter_b")
    net.add_arc("t_enter_b", "p_critical")
    net.add_arc("t_enter_b", "p_b_done")
    # Each transition is inhibited by the critical-section place —
    # neither can fire while the section is occupied.
    net.add_inhibitor_arc("p_critical", "t_enter_a")
    net.add_inhibitor_arc("p_critical", "t_enter_b")
    return net


def test_inhibitor_arc_blocks_when_place_holds_token():
    """The classic test: inhibitor on a place that has a token →
    transition is NOT enabled."""
    net = PetriNet()
    net.add_place("guard", tokens=1)
    net.add_place("input", tokens=1)
    net.add_place("output")
    net.add_transition("t")
    net.add_arc("input", "t")
    net.add_arc("t", "output")
    net.add_inhibitor_arc("guard", "t")
    # guard has a token, so t is blocked even though input is ready.
    assert not net.is_enabled("t", net.initial_marking)


def test_inhibitor_arc_allows_firing_when_place_empty():
    net = PetriNet()
    net.add_place("guard")  # empty
    net.add_place("input", tokens=1)
    net.add_place("output")
    net.add_transition("t")
    net.add_arc("input", "t")
    net.add_arc("t", "output")
    net.add_inhibitor_arc("guard", "t")
    assert net.is_enabled("t", net.initial_marking)
    new = net.fire("t", net.initial_marking)
    # Firing did NOT consume from the inhibitor place — it never had a
    # token; it's a guard, not an input.
    assert "guard" not in new
    assert new["output"] == 1


def test_inhibitor_preset_lists_guard_places():
    net = PetriNet()
    net.add_place("a")
    net.add_place("guard1")
    net.add_place("guard2")
    net.add_place("b")
    net.add_transition("t")
    net.add_arc("a", "t")
    net.add_arc("t", "b")
    net.add_inhibitor_arc("guard1", "t")
    net.add_inhibitor_arc("guard2", "t")
    assert net.inhibitor_preset("t") == {"guard1", "guard2"}
    # The regular preset excludes inhibitors — they're a different kind
    # of input.
    assert net.preset("t") == {"a"}


def test_mutex_pattern_only_one_can_enter():
    """The mutex scenario. Both A and B are ready, the critical
    section is empty, so both transitions are initially enabled.
    After whichever fires first, the other is blocked until the
    critical section clears (which doesn't happen in this minimal
    fixture — that's a follow-up with leave-section transitions)."""
    net = _mutex_net()
    marking = dict(net.initial_marking)
    assert net.is_enabled("t_enter_a", marking)
    assert net.is_enabled("t_enter_b", marking)

    after_a = net.fire("t_enter_a", marking)
    # p_critical now has a token; both transitions are now inhibited.
    assert not net.is_enabled("t_enter_a", after_a)
    assert not net.is_enabled("t_enter_b", after_a)


def test_add_inhibitor_arc_rejects_unknown_endpoints():
    net = PetriNet()
    net.add_place("p")
    net.add_transition("t")
    with pytest.raises(ValueError, match="unknown place"):
        net.add_inhibitor_arc("ghost", "t")
    with pytest.raises(ValueError, match="unknown transition"):
        net.add_inhibitor_arc("p", "ghost_t")


# ---------------------------------------------------------------------------
# Phase 9 — transition durations
# ---------------------------------------------------------------------------


def test_default_transition_duration_is_one():
    net = PetriNet()
    net.add_transition("t_immediate")
    assert net.duration("t_immediate") == 1
    # And it's not stored in the sparse dict — only non-default
    # durations should occupy memory.
    assert net.transition_durations == {}


def test_transition_duration_stored_when_non_default():
    net = PetriNet()
    net.add_transition("t_slow", duration=5)
    assert net.duration("t_slow") == 5
    assert net.transition_durations["t_slow"] == 5


def test_transition_duration_rejects_non_positive_values():
    net = PetriNet()
    with pytest.raises(ValueError, match="positive integer"):
        net.add_transition("t", duration=0)
    with pytest.raises(ValueError, match="positive integer"):
        net.add_transition("t", duration=-3)


def test_default_transition_rate_is_one():
    net = PetriNet()
    net.add_transition("t_normal")
    assert net.rate("t_normal") == 1.0
    assert net.transition_rates == {}


def test_transition_rate_stored_when_non_default():
    net = PetriNet()
    net.add_transition("t_fast", rate=3.5)
    assert net.rate("t_fast") == pytest.approx(3.5)
    assert net.transition_rates["t_fast"] == pytest.approx(3.5)


def test_transition_rate_rejects_non_positive_values():
    net = PetriNet()
    with pytest.raises(ValueError, match="positive number"):
        net.add_transition("t", rate=0)
    with pytest.raises(ValueError, match="positive number"):
        net.add_transition("t", rate=-1.0)


# ---------------------------------------------------------------------------
# Phase 9 — coloured Petri nets
# ---------------------------------------------------------------------------


def _credit_net() -> PetriNet:
    """A credit-application net used to exercise CPN behaviour. Tokens
    at p_submitted carry the application amount; the approve / decline
    guards route on it."""
    net = PetriNet()
    net.add_place("p_submitted")
    net.add_place("p_approved")
    net.add_place("p_declined")
    net.add_transition(
        "t_approve",
        guard=lambda inputs: inputs["p_submitted"] >= 1000.0,
    )
    net.add_transition(
        "t_decline",
        guard=lambda inputs: inputs["p_submitted"] < 1000.0,
    )
    # Output arcs propagate the input amount through (closure over
    # the input dict — callable form of output_value).
    net.add_arc("p_submitted", "t_approve")
    net.add_arc(
        "t_approve",
        "p_approved",
        output_value=lambda inputs: inputs["p_submitted"],
    )
    net.add_arc("p_submitted", "t_decline")
    net.add_arc(
        "t_decline",
        "p_declined",
        output_value=lambda inputs: inputs["p_submitted"],
    )
    return net


def test_coloured_is_enabled_respects_guard():
    """The approve transition only fires when the input token's value
    is >= 1000; the decline transition only fires when it is < 1000."""
    net = _credit_net()
    high = {"p_submitted": [5000.0]}
    low = {"p_submitted": [250.0]}
    assert net.is_enabled_coloured("t_approve", high)
    assert not net.is_enabled_coloured("t_decline", high)
    assert not net.is_enabled_coloured("t_approve", low)
    assert net.is_enabled_coloured("t_decline", low)


def test_coloured_fire_routes_on_token_value():
    """Firing approve on a high-value token produces the same value
    at p_approved (the output arc's callable passed it through)."""
    net = _credit_net()
    new = net.fire_coloured("t_approve", {"p_submitted": [5000.0]})
    assert new == {"p_approved": [5000.0]}


def test_coloured_fire_propagates_low_amount_through_decline():
    net = _credit_net()
    new = net.fire_coloured("t_decline", {"p_submitted": [250.0]})
    assert new == {"p_declined": [250.0]}


def test_coloured_fire_consumes_FIFO_when_multiple_tokens_present():
    """When two tokens sit at the input, the first-in token's value is
    the one consumed (FIFO). The remaining token stays at the input."""
    net = _credit_net()
    new = net.fire_coloured(
        "t_approve",
        {"p_submitted": [3000.0, 7000.0]},
    )
    # First token consumed by approve; second still queued.
    assert new["p_approved"] == [3000.0]
    assert new["p_submitted"] == [7000.0]


def test_coloured_fire_raises_when_not_enabled():
    net = _credit_net()
    with pytest.raises(ValueError, match="not enabled"):
        net.fire_coloured("t_approve", {"p_submitted": [100.0]})


def test_constant_output_value_on_arc():
    """Arcs may have a constant output_value (rather than a callable)
    to produce tokens with a fixed value regardless of input."""
    net = PetriNet()
    net.add_place("p_in")
    net.add_place("p_out")
    net.add_transition("t")
    net.add_arc("p_in", "t")
    net.add_arc("t", "p_out", output_value=42.0)
    new = net.fire_coloured("t", {"p_in": [99.0]})
    assert new == {"p_out": [42.0]}


def test_output_value_default_is_one():
    """An arc with no output_value annotation produces tokens with
    the canonical default value 1.0."""
    net = PetriNet()
    net.add_place("p_in")
    net.add_place("p_out")
    net.add_transition("t")
    net.add_arc("p_in", "t")
    net.add_arc("t", "p_out")
    new = net.fire_coloured("t", {"p_in": [99.0]})
    assert new == {"p_out": [1.0]}


def test_output_value_rejected_on_place_to_transition_arc():
    """output_value only makes sense on transition -> place arcs; on
    place -> transition arcs it has no semantics."""
    net = PetriNet()
    net.add_place("p")
    net.add_transition("t")
    with pytest.raises(ValueError, match="output_value only applies"):
        net.add_arc("p", "t", output_value=1.0)


def test_inhibitor_arc_blocks_coloured_firing_too():
    """The inhibitor-arc semantics apply to the coloured token-game
    identically: a transition is blocked while its inhibitor place
    holds any tokens."""
    net = PetriNet()
    net.add_place("p_in")
    net.add_place("p_guard")
    net.add_place("p_out")
    net.add_transition("t")
    net.add_arc("p_in", "t")
    net.add_arc("t", "p_out")
    net.add_inhibitor_arc("p_guard", "t")
    blocked = {"p_in": [1.0], "p_guard": [0.0]}  # guard has a token
    assert not net.is_enabled_coloured("t", blocked)
    free = {"p_in": [1.0]}  # guard is empty
    assert net.is_enabled_coloured("t", free)


def test_validate_flags_invalid_rate_dict_entries():
    net = PetriNet()
    net.add_place("p", tokens=1)
    net.add_place("q")
    net.add_transition("t")
    net.add_arc("p", "t")
    net.add_arc("t", "q")
    net.transition_rates["ghost"] = 2.0
    net.transition_rates["t"] = 0.0
    issues = net.validate()
    assert any("unknown transition" in issue for issue in issues)
    assert any("non-positive rate" in issue for issue in issues)


def test_validate_flags_invalid_duration_dict_entries():
    net = PetriNet()
    net.add_place("p1", tokens=1)
    net.add_place("p2")
    net.add_transition("t")
    net.add_arc("p1", "t")
    net.add_arc("t", "p2")
    # Directly poke an invalid entry to simulate a malformed net
    # constructed by some lower-level loader.
    net.transition_durations["ghost"] = 4
    net.transition_durations["t"] = 0
    issues = net.validate()
    assert any("unknown transition" in issue for issue in issues)
    assert any("non-positive duration" in issue for issue in issues)
