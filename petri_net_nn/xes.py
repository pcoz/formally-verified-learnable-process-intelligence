"""IEEE XES (eXtensible Event Stream) log loader.

XES is the standard format for process execution logs and is mentioned
explicitly in §10 Step 3 of the architecture spec as the data source for
training. This scaffold implements a minimal reader covering the subset
sufficient to drive `train_on_traces` in `petri_net_nn.traces`:

  * one ``<log>`` containing zero or more ``<trace>`` elements
  * each ``<trace>`` carrying zero or more ``<event>`` elements
  * typed attributes (``<string>``, ``<int>``, ``<float>``, ``<boolean>``,
    ``<date>``) at the log, trace, or event level — all surfaced as
    string-valued entries in the corresponding ``attributes`` dict

Each event's name is the value of its ``concept:name`` attribute, which
is the XES convention for the activity (task) that fired. That string
is what matches against the transition labels produced by
``parse_bpmn``.

Out of scope: extensions, classifiers, globals, nested attributes
(lists / containers). They parse without error but are ignored.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO


XES_NS_CANDIDATES = (
    "http://www.xes-standard.org/",
    "http://code.deckfour.org/xes",
)
_TYPED_ATTRIBUTE_TAGS = {"string", "int", "float", "boolean", "date", "id"}


@dataclass
class XESEvent:
    name: str
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class XESTrace:
    attributes: dict[str, str] = field(default_factory=dict)
    events: list[XESEvent] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.attributes.get("concept:name", "")


def _local(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _collect_attributes(element: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for child in element:
        tag = _local(child.tag)
        if tag in _TYPED_ATTRIBUTE_TAGS:
            key = child.attrib.get("key")
            value = child.attrib.get("value", "")
            if key is not None:
                result[key] = value
    return result


def _load_xml(source: str | Path | IO[str] | IO[bytes]) -> ET.Element:
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


def parse_xes(source: str | Path | IO[str] | IO[bytes]) -> list[XESTrace]:
    """Parse an IEEE XES log into a list of XESTrace objects.

    The XES namespace is optional; both namespaced and non-namespaced
    documents parse identically.
    """
    root = _load_xml(source)
    if _local(root.tag) != "log":
        raise ValueError(
            f"expected XES root element <log>, got <{_local(root.tag)}>"
        )

    traces: list[XESTrace] = []
    for trace_el in root:
        if _local(trace_el.tag) != "trace":
            continue
        trace = XESTrace(attributes=_collect_attributes(trace_el))
        for event_el in trace_el:
            if _local(event_el.tag) != "event":
                continue
            attrs = _collect_attributes(event_el)
            event = XESEvent(name=attrs.get("concept:name", ""), attributes=attrs)
            trace.events.append(event)
        traces.append(trace)

    return traces
