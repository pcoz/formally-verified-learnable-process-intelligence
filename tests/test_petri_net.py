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
