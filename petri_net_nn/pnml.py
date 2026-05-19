"""PNML (Petri Net Markup Language) import and export.

PNML is the ISO/IEC 15909-2 interchange format for Petri nets. Adding
import / export here connects PETRA to the established Petri-net
tool ecosystem — CPN Tools, GreatSPN, TINA, Snoopy, ProM, and many
others all emit or consume PNML.

This module supports the **P/T net** subset of PNML 2009:

  * one or more ``<net>`` elements (the first net is parsed; others
    are ignored on import);
  * places, transitions, arcs nested under ``<page>`` elements (the
    parser flattens all pages into a single net, which is the
    standard interpretation when no page-level semantics are
    needed);
  * names (via ``<name><text>...</text></name>``);
  * initial markings (``<initialMarking><text>N</text></initialMarking>``);
  * arc inscriptions for arc weights
    (``<inscription><text>N</text></inscription>``);
  * inhibitor arcs via the common ``<arctype><text>inhibitor</text></arctype>``
    extension recognised by Snoopy, GreatSPN and others.

PETRA-specific extensions that have no standard PNML representation
— transition durations, firing rates, guards, arc output values —
are dropped on export and ignored on import. The structural net
(places, transitions, flow, weights, initial marking, inhibitor
arcs) round-trips cleanly.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO

from petri_net_nn.petri_net import PetriNet


# The 2009 standard uses this namespace; some tools emit PNML without
# any namespace at all. The parser handles both transparently.
PNML_NS = "http://www.pnml.org/version-2009/grammar/pnml"
NS_PREFIX = f"{{{PNML_NS}}}"

# Net-type URI for plain place/transition nets.
PT_NET_TYPE = "http://www.pnml.org/version-2009/grammar/ptnet"


def _local(tag: str) -> str:
    """Strip an XML namespace prefix from a tag if present, returning
    just the local name. Lets the parser handle both namespaced
    (standards-compliant) and unnamespaced (tool-emitted) PNML
    uniformly."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _text_of(element: ET.Element | None) -> str | None:
    """Read the ``<text>...</text>`` child of a PNML labelling element
    (the standard way PNML carries human-readable values). Returns
    ``None`` if the element or its text child is missing."""
    if element is None:
        return None
    for child in element:
        if _local(child.tag) == "text":
            return (child.text or "").strip()
    return (element.text or "").strip() or None


def _load_xml(source: str | Path | IO[str] | IO[bytes]) -> ET.Element:
    """Accept the same source forms as the other PETRA parsers: an
    XML string, a file path, or a file-like object."""
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return ET.fromstring(data)
    if isinstance(source, Path) or (
        isinstance(source, str) and not source.lstrip().startswith("<")
    ):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        return ET.parse(os.fspath(path)).getroot()
    return ET.fromstring(source)


def parse_pnml(source: str | Path | IO[str] | IO[bytes]) -> PetriNet:
    """Parse a PNML document into a PETRA ``PetriNet``.

    The first ``<net>`` element in the document is used; additional
    nets (PNML allows multiple) are ignored. All pages within that
    net are flattened — places, transitions, and arcs are collected
    regardless of which page they live on. Anything PETRA doesn't
    understand (graphics, tool-specific extensions, colour
    declarations) is silently dropped."""
    root = _load_xml(source)
    if _local(root.tag) != "pnml":
        raise ValueError(
            f"expected PNML root element <pnml>, got <{_local(root.tag)}>"
        )

    net_elem = next((c for c in root if _local(c.tag) == "net"), None)
    if net_elem is None:
        raise ValueError("no <net> element found inside <pnml>")

    net = PetriNet()

    # PNML allows places / transitions / arcs to appear either directly
    # under <net> or under nested <page> elements. We walk recursively
    # and pick up everything we recognise, regardless of nesting depth.
    def walk(element: ET.Element) -> None:
        for child in element:
            tag = _local(child.tag)
            if tag == "place":
                _read_place(child, net)
            elif tag == "transition":
                _read_transition(child, net)
            elif tag == "arc":
                _read_arc(child, net)
            elif tag == "page":
                walk(child)

    walk(net_elem)

    return net


def _read_place(elem: ET.Element, net: PetriNet) -> None:
    place_id = elem.attrib["id"]
    label = _text_of(_find(elem, "name"))
    initial = 0
    marking_elem = _find(elem, "initialMarking")
    if marking_elem is not None:
        text = _text_of(marking_elem)
        if text:
            try:
                initial = int(text)
            except ValueError:
                # Some tools write "Default,1" or similar; fall back
                # gracefully to zero rather than crashing.
                initial = 0
    net.add_place(place_id, label=label, tokens=initial)


def _read_transition(elem: ET.Element, net: PetriNet) -> None:
    transition_id = elem.attrib["id"]
    label = _text_of(_find(elem, "name"))
    net.add_transition(transition_id, label=label)


def _read_arc(elem: ET.Element, net: PetriNet) -> None:
    src = elem.attrib["source"]
    dst = elem.attrib["target"]

    # Arc weight comes from the optional <inscription><text>N</text>...
    weight = 1
    insc = _find(elem, "inscription")
    if insc is not None:
        text = _text_of(insc)
        if text:
            try:
                weight = int(text)
            except ValueError:
                weight = 1

    # Inhibitor arcs use an <arctype><text>inhibitor</text></arctype>
    # element in the common Snoopy / GreatSPN convention. PNML 2009
    # didn't standardise inhibitor arcs but this is the de-facto
    # extension PETRA recognises on import.
    arctype = _text_of(_find(elem, "arctype")) or _text_of(_find(elem, "type"))
    if arctype and arctype.strip().lower() == "inhibitor":
        # Inhibitor arcs only make sense place -> transition.
        if src in net.places and dst in net.transitions:
            net.add_inhibitor_arc(src, dst)
        else:
            raise ValueError(
                f"inhibitor arc {src!r} -> {dst!r} must run from a place "
                f"to a transition"
            )
        return

    net.add_arc(src, dst, weight=weight)


def _find(element: ET.Element, local_name: str) -> ET.Element | None:
    """Find the first direct child with the given local name,
    namespace-agnostic. ``element.find`` doesn't help when the
    document uses a namespace prefix we don't know in advance."""
    for child in element:
        if _local(child.tag) == local_name:
            return child
    return None


def to_pnml(net: PetriNet, *, net_id: str = "net1", name: str | None = None) -> str:
    """Serialise a PETRA ``PetriNet`` as a PNML 2009 P/T net document.

    Includes places, transitions, arcs, arc weights (as PNML
    inscriptions), initial markings, place / transition names, and
    inhibitor arcs (via the de-facto ``<arctype>inhibitor</arctype>``
    extension). PETRA-specific extensions without a standard PNML
    representation — transition durations, firing rates, guards,
    arc output values — are dropped. The export is a string of
    UTF-8 XML, suitable for writing to a ``.pnml`` file or piping
    into any PNML-aware tool."""
    pnml = ET.Element(f"{NS_PREFIX}pnml")
    net_elem = ET.SubElement(
        pnml, f"{NS_PREFIX}net", attrib={"id": net_id, "type": PT_NET_TYPE}
    )
    if name is not None:
        _write_label(net_elem, "name", name)

    # PNML conventionally wraps the contents in a single <page>;
    # some tools require it. Keep things on one page since PETRA
    # has no notion of pagination.
    page = ET.SubElement(net_elem, f"{NS_PREFIX}page", attrib={"id": "page1"})

    for place in sorted(net.places):
        place_elem = ET.SubElement(
            page, f"{NS_PREFIX}place", attrib={"id": place}
        )
        label = net.place_labels.get(place)
        if label is not None:
            _write_label(place_elem, "name", label)
        tokens = net.initial_marking.get(place, 0)
        if tokens:
            _write_label(place_elem, "initialMarking", str(tokens))

    for transition in sorted(net.transitions):
        t_elem = ET.SubElement(
            page, f"{NS_PREFIX}transition", attrib={"id": transition}
        )
        label = net.transition_labels.get(transition)
        if label is not None:
            _write_label(t_elem, "name", label)

    arc_counter = 0
    for src, dst in sorted(net.flow):
        arc_counter += 1
        arc_elem = ET.SubElement(
            page,
            f"{NS_PREFIX}arc",
            attrib={"id": f"arc_{arc_counter}", "source": src, "target": dst},
        )
        weight = net.weight(src, dst)
        if weight != 1:
            _write_label(arc_elem, "inscription", str(weight))

    for src, dst in sorted(net.inhibitor_arcs):
        arc_counter += 1
        arc_elem = ET.SubElement(
            page,
            f"{NS_PREFIX}arc",
            attrib={"id": f"arc_{arc_counter}", "source": src, "target": dst},
        )
        # The arctype element is the conventional way to tag an arc
        # as inhibitor. Plain PNML 2009 doesn't standardise this but
        # the Snoopy / GreatSPN / TINA ecosystem recognises it.
        _write_label(arc_elem, "arctype", "inhibitor")

    ET.register_namespace("", PNML_NS)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        pnml, encoding="unicode"
    )


def _write_label(parent: ET.Element, tag: str, value: str) -> None:
    """Add a PNML labelling element ``<tag><text>value</text></tag>``
    under ``parent``. This is the standard PNML pattern for any
    human-readable annotation — names, inscriptions, markings, types."""
    label_elem = ET.SubElement(parent, f"{NS_PREFIX}{tag}")
    text_elem = ET.SubElement(label_elem, f"{NS_PREFIX}text")
    text_elem.text = value
