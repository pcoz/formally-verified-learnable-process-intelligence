"""Tests for the PNML import / export (Phase 10 ecosystem
integration).

PNML is the ISO interchange format for Petri nets. These tests pin
the structural correctness of the import and the round-trip — a
hand-built net exported and re-imported should match the original
on places, transitions, flow relation, weights, initial marking,
and inhibitor arcs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from petri_net_nn import PetriNet, parse_pnml, to_pnml


FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Import — parse_pnml against a hand-written PNML 2009 P/T net
# ---------------------------------------------------------------------------


def test_parse_pnml_reads_producer_consumer_fixture():
    """The producer/consumer fixture has four places, two transitions,
    six arcs, and one initial marking of three tokens. Parsing should
    recover all of that."""
    net = parse_pnml(FIXTURES / "producer_consumer.pnml")
    assert net.places == {"buffer_slot", "buffer_item", "produced", "consumed"}
    assert net.transitions == {"t_produce", "t_consume"}
    assert len(net.flow) == 6
    assert net.initial_marking == {"buffer_slot": 3}
    assert net.validate() == []


def test_parse_pnml_captures_names_as_labels():
    net = parse_pnml(FIXTURES / "producer_consumer.pnml")
    assert net.place_labels["buffer_slot"] == "free buffer slot"
    assert net.transition_labels["t_produce"] == "produce one item"


def test_parse_pnml_token_game_runs_on_imported_net():
    """The imported net should support the standard token-game; the
    producer can fire three times (consuming the three slot tokens),
    then is blocked until the consumer frees a slot."""
    net = parse_pnml(FIXTURES / "producer_consumer.pnml")
    marking = dict(net.initial_marking)
    for _ in range(3):
        marking = net.fire("t_produce", marking)
    assert marking.get("buffer_item") == 3
    assert "buffer_slot" not in marking
    assert not net.is_enabled("t_produce", marking)
    assert net.is_enabled("t_consume", marking)


def test_parse_pnml_handles_arc_inscription_as_weight():
    """An <inscription>3</inscription> on an arc should round-trip
    as arc weight 3."""
    xml = """<?xml version="1.0"?>
        <pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
          <net id="n" type="http://www.pnml.org/version-2009/grammar/ptnet">
            <place id="p"><initialMarking><text>5</text></initialMarking></place>
            <transition id="t"/>
            <place id="q"/>
            <arc id="a1" source="p" target="t">
              <inscription><text>3</text></inscription>
            </arc>
            <arc id="a2" source="t" target="q"/>
          </net>
        </pnml>"""
    net = parse_pnml(xml)
    assert net.weight("p", "t") == 3
    assert net.weight("t", "q") == 1


def test_parse_pnml_handles_inhibitor_arctype_extension():
    """The de-facto inhibitor-arc extension uses an <arctype>
    inhibitor</arctype> element. Recognised on import and recorded
    in the inhibitor_arcs set."""
    xml = """<?xml version="1.0"?>
        <pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
          <net id="n" type="http://www.pnml.org/version-2009/grammar/ptnet">
            <place id="p_input"/>
            <place id="p_guard"/>
            <place id="p_output"/>
            <transition id="t"/>
            <arc id="a1" source="p_input" target="t"/>
            <arc id="a2" source="t" target="p_output"/>
            <arc id="a3" source="p_guard" target="t">
              <arctype><text>inhibitor</text></arctype>
            </arc>
          </net>
        </pnml>"""
    net = parse_pnml(xml)
    assert ("p_guard", "t") in net.inhibitor_arcs


def test_parse_pnml_handles_unnamespaced_documents():
    """Many tools emit PNML without the standard 2009 namespace.
    The parser should accept both with and without a namespace."""
    xml = """<?xml version="1.0"?>
        <pnml>
          <net id="n" type="http://www.pnml.org/version-2009/grammar/ptnet">
            <place id="p"><initialMarking><text>1</text></initialMarking></place>
            <transition id="t"/>
            <place id="q"/>
            <arc id="a1" source="p" target="t"/>
            <arc id="a2" source="t" target="q"/>
          </net>
        </pnml>"""
    net = parse_pnml(xml)
    assert net.places == {"p", "q"}
    assert net.initial_marking == {"p": 1}


def test_parse_pnml_rejects_non_pnml_root():
    with pytest.raises(ValueError, match="expected PNML root"):
        parse_pnml("<?xml version='1.0'?><notpnml/>")


def test_parse_pnml_rejects_empty_pnml():
    xml = """<?xml version="1.0"?>
        <pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml"/>"""
    with pytest.raises(ValueError, match="no <net>"):
        parse_pnml(xml)


def test_parse_pnml_flattens_multiple_pages():
    """PNML allows places / transitions / arcs to be split across
    pages. PETRA flattens them into a single net."""
    xml = """<?xml version="1.0"?>
        <pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
          <net id="n" type="http://www.pnml.org/version-2009/grammar/ptnet">
            <page id="page_a">
              <place id="p_a"/>
              <transition id="t_a"/>
              <arc id="a_1" source="p_a" target="t_a"/>
            </page>
            <page id="page_b">
              <place id="p_b"/>
              <arc id="a_2" source="t_a" target="p_b"/>
            </page>
          </net>
        </pnml>"""
    net = parse_pnml(xml)
    assert net.places == {"p_a", "p_b"}
    assert net.transitions == {"t_a"}
    assert net.flow == {("p_a", "t_a"), ("t_a", "p_b")}


# ---------------------------------------------------------------------------
# Export — to_pnml produces standards-compliant XML
# ---------------------------------------------------------------------------


def test_to_pnml_emits_valid_xml_with_expected_structure():
    net = PetriNet()
    net.add_place("p", tokens=1, label="start")
    net.add_place("q", label="end")
    net.add_transition("t", label="step")
    net.add_arc("p", "t")
    net.add_arc("t", "q")

    xml = to_pnml(net)
    assert "<pnml" in xml
    assert "<place" in xml
    assert "<transition" in xml
    assert "<arc" in xml
    assert "<initialMarking>" in xml
    # Sanity-check that the produced XML is well-formed.
    parse_pnml(xml)


def test_to_pnml_emits_arc_inscription_for_non_default_weight():
    net = PetriNet()
    net.add_place("p")
    net.add_place("q")
    net.add_transition("t")
    net.add_arc("p", "t", weight=4)
    net.add_arc("t", "q")
    xml = to_pnml(net)
    assert "<inscription>" in xml
    assert "<text>4</text>" in xml


def test_to_pnml_emits_arctype_inhibitor_for_inhibitor_arcs():
    net = PetriNet()
    net.add_place("p_in")
    net.add_place("p_guard")
    net.add_place("p_out")
    net.add_transition("t")
    net.add_arc("p_in", "t")
    net.add_arc("t", "p_out")
    net.add_inhibitor_arc("p_guard", "t")
    xml = to_pnml(net)
    assert "<arctype>" in xml
    assert "inhibitor" in xml


# ---------------------------------------------------------------------------
# Round-trip — export then import preserves the structural core
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_places_transitions_arcs_and_marking():
    original = PetriNet()
    original.add_place("p_a", tokens=2, label="alpha")
    original.add_place("p_b", label="beta")
    original.add_place("p_c")
    original.add_transition("t_x", label="ex")
    original.add_transition("t_y")
    original.add_arc("p_a", "t_x", weight=2)
    original.add_arc("t_x", "p_b")
    original.add_arc("p_b", "t_y")
    original.add_arc("t_y", "p_c")
    original.add_inhibitor_arc("p_c", "t_x")

    xml = to_pnml(original)
    roundtripped = parse_pnml(xml)

    assert roundtripped.places == original.places
    assert roundtripped.transitions == original.transitions
    assert roundtripped.flow == original.flow
    assert roundtripped.initial_marking == original.initial_marking
    assert roundtripped.weight("p_a", "t_x") == 2
    assert ("p_c", "t_x") in roundtripped.inhibitor_arcs
    assert roundtripped.place_labels["p_a"] == "alpha"
    assert roundtripped.transition_labels["t_x"] == "ex"
