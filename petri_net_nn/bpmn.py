"""BPMN 2.0 -> Petri net extractor.

Implements the structural translation in §3 of the architecture spec
(Aalst-style workflow-net mapping):

    BPMN sequenceFlow      -> place
    BPMN task              -> single transition
    BPMN startEvent        -> initial token on the outgoing flow's place
    BPMN endEvent          -> sink place (incoming flow's place)
    parallelGateway        -> single transition (AND-split / AND-join)
    exclusiveGateway       -> one transition per branch sharing a place
                              (XOR-split: shared input;
                               XOR-join : shared output)
    compensation boundary  -> see compensation translation below.
    compensate end event   -> see compensation translation below.
    error / timer /        -> alternative transition out of the
    signal / escalation /     attached task's input place, into the
    message boundary          boundary's outgoing sequenceFlow's place.
    intermediateThrowEvent -> pass-through transition (one input,
    / intermediateCatchEvent  one output, no event semantics).
    laneSet / lane         -> silently ignored (informational only).
    messageFlow            -> shared message place between the source
    (collaboration only)      task transition in one pool and the target
                              task transition in another, threading
                              token flow across pool boundaries.

Compensation has two interpretations depending on whether the process
contains a compensate end event:
  * without one — the boundary becomes an alternative T_fail
    transition out of the attached task's input place;
  * with one — the boundary is a passive marker, the task transition
    is extended with an output to a completion-marker place, and the
    throw event becomes a transition AND-joining its own input flow
    with that completion marker.

Cross-pool composition: a ``<collaboration>`` root with
``<participant processRef="...">`` children is supported. Each pool's
process is parsed independently into the same Petri net with the
participant's ID as a prefix on every place / transition ID, so the
pools' internal nodes don't collide. ``<messageFlow>`` elements then
introduce shared message places that connect the sending pool's task
transition to the receiving pool's task transition.

Out of scope for this scaffold: subprocesses (raise with a roadmap
pointer), non-interrupting boundary events, intermediate events with
event definitions, message flows to/from non-task nodes (events,
gateways).

The parser accepts a BPMN XML string, a path to a .bpmn or .xml file,
or any file-like object yielding XML text.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO

from petri_net_nn.petri_net import PetriNet


BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
NS = f"{{{BPMN_NS}}}"

TASK_TAGS = {
    "task",
    "userTask",
    "serviceTask",
    "scriptTask",
    "manualTask",
    "businessRuleTask",
    "sendTask",
    "receiveTask",
}

BOUNDARY_EVENT_DEFS = {
    "errorEventDefinition",
    "timerEventDefinition",
    "signalEventDefinition",
    "escalationEventDefinition",
    "messageEventDefinition",
}

INTERMEDIATE_EVENT_TAGS = {
    "intermediateThrowEvent",
    "intermediateCatchEvent",
}


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


def _local(tag: str) -> str:
    return tag.removeprefix(NS) if tag.startswith(NS) else tag


def parse_bpmn(source: str | Path | IO[str] | IO[bytes]) -> PetriNet:
    """Parse a BPMN 2.0 XML source into a Petri net.

    Dispatches on the root structure: a ``<collaboration>`` produces a
    composed multi-pool net (one per participant), while a plain
    ``<process>`` document produces a single-pool net (current
    behaviour). A ``ValueError`` is raised for unsupported BPMN
    constructs.
    """
    root = _load_xml(source)

    collab = root.find(f"{NS}collaboration")
    if collab is not None:
        return _parse_collaboration(root, collab)

    process = root.find(f"{NS}process")
    if process is None:
        raise ValueError("no <process> element found in BPMN document")

    net = PetriNet()
    _parse_process(process, prefix="", net=net)
    return net


def _parse_collaboration(root: ET.Element, collab: ET.Element) -> PetriNet:
    processes = {p.attrib["id"]: p for p in root.findall(f"{NS}process")}
    net = PetriNet()
    node_to_pool: dict[str, str] = {}

    for participant in collab.findall(f"{NS}participant"):
        pid = participant.attrib["id"]
        process_ref = participant.attrib.get("processRef")
        if not process_ref:
            raise ValueError(f"participant {pid!r} is missing processRef")
        if process_ref not in processes:
            raise ValueError(
                f"participant {pid!r} references unknown process "
                f"{process_ref!r}"
            )
        _parse_process(
            processes[process_ref],
            prefix=f"{pid}:",
            net=net,
            node_to_pool=node_to_pool,
        )

    for msg in collab.findall(f"{NS}messageFlow"):
        msg_id = msg.attrib["id"]
        src = msg.attrib["sourceRef"]
        tgt = msg.attrib["targetRef"]
        if src not in node_to_pool:
            raise ValueError(
                f"messageFlow {msg_id!r}: sourceRef {src!r} is not a known "
                f"node in any participant's process"
            )
        if tgt not in node_to_pool:
            raise ValueError(
                f"messageFlow {msg_id!r}: targetRef {tgt!r} is not a known "
                f"node in any participant's process"
            )
        src_pool = node_to_pool[src]
        tgt_pool = node_to_pool[tgt]
        if src_pool == tgt_pool:
            raise ValueError(
                f"messageFlow {msg_id!r}: source and target are in the "
                f"same pool — use sequenceFlow for intra-pool connections"
            )

        src_trans = f"{src_pool}t_{src}"
        tgt_trans = f"{tgt_pool}t_{tgt}"
        if src_trans not in net.transitions:
            raise ValueError(
                f"messageFlow {msg_id!r}: source {src!r} did not produce a "
                f"transition (must be a task or pass-through node)"
            )
        if tgt_trans not in net.transitions:
            raise ValueError(
                f"messageFlow {msg_id!r}: target {tgt!r} did not produce a "
                f"transition (must be a task or pass-through node)"
            )

        msg_place = f"msg_{msg_id}"
        net.add_place(
            msg_place, label=msg.attrib.get("name", "") or f"message: {msg_id}"
        )
        net.add_arc(src_trans, msg_place)
        net.add_arc(msg_place, tgt_trans)

    return net


def _parse_process(
    process: ET.Element,
    *,
    prefix: str,
    net: PetriNet,
    node_to_pool: dict[str, str] | None = None,
) -> None:
    """Translate one ``<process>`` element into ``net``, prefixing every
    place/transition ID with ``prefix``. When called for a participant
    inside a collaboration, ``node_to_pool`` is populated with
    ``{bpmn_node_id: prefix}`` so message flow translation can locate
    which pool each ``sourceRef`` / ``targetRef`` lives in."""
    nodes: dict[str, dict] = {}
    flows: list[dict] = []
    boundary_events: dict[str, dict] = {}
    associations: dict[str, str] = {}

    def P(flow_id: str) -> str:
        return f"{prefix}p_{flow_id}"

    def T(node_id: str) -> str:
        return f"{prefix}t_{node_id}"

    for child in process:
        tag = _local(child.tag)
        if tag == "sequenceFlow":
            flows.append(
                {
                    "id": child.attrib["id"],
                    "source": child.attrib["sourceRef"],
                    "target": child.attrib["targetRef"],
                    "name": child.attrib.get("name", ""),
                }
            )
        elif tag in TASK_TAGS:
            nodes[child.attrib["id"]] = {
                "kind": tag,
                "name": child.attrib.get("name", ""),
                "is_for_compensation": child.attrib.get("isForCompensation") == "true",
                "incoming": [],
                "outgoing": [],
            }
        elif tag in {
            "startEvent",
            "endEvent",
            "exclusiveGateway",
            "parallelGateway",
        }:
            kind = tag
            if tag == "endEvent" and any(
                _local(c.tag) == "compensateEventDefinition" for c in child
            ):
                kind = "compensateEndEvent"
            nodes[child.attrib["id"]] = {
                "kind": kind,
                "name": child.attrib.get("name", ""),
                "incoming": [],
                "outgoing": [],
            }
        elif tag in INTERMEDIATE_EVENT_TAGS:
            event_def = next(
                (
                    _local(c.tag)
                    for c in child
                    if _local(c.tag).endswith("EventDefinition")
                ),
                None,
            )
            if event_def is not None:
                raise ValueError(
                    f"intermediate event {child.attrib['id']!r} has event "
                    f"definition {event_def!r}; only plain pass-through "
                    f"intermediate events are supported in this scaffold"
                )
            nodes[child.attrib["id"]] = {
                "kind": "intermediateEvent",
                "name": child.attrib.get("name", ""),
                "incoming": [],
                "outgoing": [],
            }
        elif tag == "subProcess":
            raise ValueError(
                f"subProcess {child.attrib['id']!r} is not supported in this "
                f"scaffold; see docs/ROADMAP.md Phase 4 for the deferred inline "
                f"flattening approach"
            )
        elif tag == "boundaryEvent":
            if "attachedToRef" not in child.attrib:
                raise ValueError(
                    f"boundaryEvent {child.attrib['id']!r} is missing "
                    f"attachedToRef"
                )
            event_def = next(
                (
                    _local(c.tag)
                    for c in child
                    if _local(c.tag).endswith("EventDefinition")
                ),
                None,
            )
            if event_def is None:
                raise ValueError(
                    f"boundaryEvent {child.attrib['id']!r} has no event "
                    f"definition; plain boundary events are not supported"
                )
            cancel = child.attrib.get("cancelActivity", "true")
            if event_def == "compensateEventDefinition":
                boundary_events[child.attrib["id"]] = {
                    "attached_to": child.attrib["attachedToRef"],
                    "name": child.attrib.get("name", ""),
                }
            elif event_def in BOUNDARY_EVENT_DEFS:
                nodes[child.attrib["id"]] = {
                    "kind": "boundaryEvent",
                    "event_def": event_def,
                    "name": child.attrib.get("name", ""),
                    "attached_to": child.attrib["attachedToRef"],
                    "interrupting": cancel == "true",
                    "incoming": [],
                    "outgoing": [],
                }
            else:
                raise ValueError(
                    f"boundaryEvent {child.attrib['id']!r} has unsupported "
                    f"event definition {event_def!r}"
                )
        elif tag == "association":
            associations[child.attrib["sourceRef"]] = child.attrib["targetRef"]

    for f in flows:
        if f["source"] not in nodes:
            raise ValueError(
                f"sequenceFlow {f['id']!r} has unknown sourceRef {f['source']!r}"
            )
        if f["target"] not in nodes:
            raise ValueError(
                f"sequenceFlow {f['id']!r} has unknown targetRef {f['target']!r}"
            )
        nodes[f["source"]]["outgoing"].append(f["id"])
        nodes[f["target"]]["incoming"].append(f["id"])

    compensation_pairs: list[tuple[str, str, str]] = []
    compensation_handlers: set[str] = set()
    for bnd_id, bnd in boundary_events.items():
        task_id = bnd["attached_to"]
        if task_id not in nodes or nodes[task_id]["kind"] not in TASK_TAGS:
            raise ValueError(
                f"compensation boundaryEvent {bnd_id!r} is attached to "
                f"{task_id!r}, which is not a task"
            )
        handler_id = associations.get(bnd_id)
        if handler_id is None:
            raise ValueError(
                f"compensation boundaryEvent {bnd_id!r} has no <association> "
                f"linking it to a compensation handler"
            )
        if handler_id not in nodes or nodes[handler_id]["kind"] not in TASK_TAGS:
            raise ValueError(
                f"association from boundaryEvent {bnd_id!r} targets "
                f"{handler_id!r}, which is not a task"
            )
        if not nodes[handler_id].get("is_for_compensation"):
            raise ValueError(
                f"compensation handler {handler_id!r} must declare "
                f"isForCompensation=\"true\""
            )
        compensation_pairs.append((task_id, bnd_id, handler_id))
        compensation_handlers.add(handler_id)

    for f in flows:
        net.add_place(P(f["id"]), label=f["name"] or f["id"])

    non_interrupting_outputs: dict[str, list[str]] = {}
    for nid, node in nodes.items():
        if node["kind"] != "boundaryEvent":
            continue
        if node.get("interrupting", True):
            continue
        if node["incoming"]:
            raise ValueError(
                f"boundaryEvent {nid!r} must not have incoming sequenceFlows"
            )
        if len(node["outgoing"]) != 1:
            raise ValueError(
                f"boundaryEvent {nid!r} must have exactly one outgoing flow"
            )
        attached = nodes.get(node["attached_to"])
        if attached is None or attached["kind"] not in TASK_TAGS:
            raise ValueError(
                f"boundaryEvent {nid!r} must be attached to a task; "
                f"{node['attached_to']!r} is "
                f"{attached['kind'] if attached else 'unknown'!r}"
            )
        non_interrupting_outputs.setdefault(node["attached_to"], []).append(
            node["outgoing"][0]
        )

    for nid, node in nodes.items():
        if nid in compensation_handlers:
            continue

        kind = node["kind"]
        name = node["name"] or nid
        incoming = node["incoming"]
        outgoing = node["outgoing"]

        if kind == "startEvent":
            if len(outgoing) != 1 or incoming:
                raise ValueError(
                    f"startEvent {nid!r} must have exactly one outgoing flow "
                    f"and no incoming flow (got {len(incoming)} in, "
                    f"{len(outgoing)} out)"
                )
            place = P(outgoing[0])
            net.initial_marking[place] = net.initial_marking.get(place, 0) + 1

        elif kind == "endEvent":
            if len(incoming) != 1 or outgoing:
                raise ValueError(
                    f"endEvent {nid!r} must have exactly one incoming flow "
                    f"and no outgoing flow (got {len(incoming)} in, "
                    f"{len(outgoing)} out)"
                )

        elif kind == "compensateEndEvent":
            pass

        elif kind == "intermediateEvent":
            if not incoming or not outgoing:
                raise ValueError(
                    f"intermediate event {nid!r} must have at least one "
                    f"incoming and one outgoing flow"
                )
            tid = T(nid)
            net.add_transition(tid, label=name)
            for fid in incoming:
                net.add_arc(P(fid), tid)
            for fid in outgoing:
                net.add_arc(tid, P(fid))

        elif kind == "boundaryEvent":
            if not node.get("interrupting", True):
                continue
            if incoming:
                raise ValueError(
                    f"boundaryEvent {nid!r} must not have incoming "
                    f"sequenceFlows (it is triggered by its attached activity)"
                )
            if len(outgoing) != 1:
                raise ValueError(
                    f"boundaryEvent {nid!r} must have exactly one outgoing "
                    f"flow (the handler path); got {len(outgoing)}"
                )
            attached_id = node["attached_to"]
            if attached_id not in nodes:
                raise ValueError(
                    f"boundaryEvent {nid!r} is attached to unknown node "
                    f"{attached_id!r}"
                )
            attached = nodes[attached_id]
            if attached["kind"] not in TASK_TAGS:
                raise ValueError(
                    f"boundaryEvent {nid!r} must be attached to a task; "
                    f"{attached_id!r} has kind {attached['kind']!r}"
                )
            if len(attached["incoming"]) != 1:
                raise ValueError(
                    f"task {attached_id!r} with a boundary event must have "
                    f"exactly one incoming flow"
                )
            p_active = P(attached["incoming"][0])
            p_handler = P(outgoing[0])
            tid = T(nid)
            event_kind = node["event_def"].removesuffix("EventDefinition")
            net.add_transition(
                tid,
                label=(
                    f"{event_kind}: {attached['name'] or attached_id}"
                    if event_kind
                    else f"boundary: {attached['name'] or attached_id}"
                ),
            )
            net.add_arc(p_active, tid)
            net.add_arc(tid, p_handler)

        elif kind in TASK_TAGS:
            if not incoming or not outgoing:
                raise ValueError(
                    f"task {nid!r} must have at least one incoming and one "
                    f"outgoing flow"
                )
            tid = T(nid)
            net.add_transition(tid, label=name)
            for fid in incoming:
                net.add_arc(P(fid), tid)
            for fid in outgoing:
                net.add_arc(tid, P(fid))
            for boundary_fid in non_interrupting_outputs.get(nid, []):
                net.add_arc(tid, P(boundary_fid))

        elif kind == "parallelGateway":
            tid = T(nid)
            net.add_transition(tid, label=name)
            for fid in incoming:
                net.add_arc(P(fid), tid)
            for fid in outgoing:
                net.add_arc(tid, P(fid))

        elif kind == "exclusiveGateway":
            n_in, n_out = len(incoming), len(outgoing)
            if n_in == 0 or n_out == 0:
                raise ValueError(
                    f"exclusiveGateway {nid!r} must have at least one in and "
                    f"one out flow"
                )
            if n_in > 1 and n_out > 1:
                raise ValueError(
                    f"exclusiveGateway {nid!r} is both a split and a join "
                    f"({n_in} in, {n_out} out); split it into two gateways"
                )
            if n_in == 1:
                shared_in = P(incoming[0])
                for i, fid in enumerate(outgoing):
                    tid = f"{prefix}t_{nid}_{i}"
                    net.add_transition(tid, label=f"{name} -> {fid}")
                    net.add_arc(shared_in, tid)
                    net.add_arc(tid, P(fid))
            else:
                shared_out = P(outgoing[0])
                for i, fid in enumerate(incoming):
                    tid = f"{prefix}t_{nid}_{i}"
                    net.add_transition(tid, label=f"{fid} -> {name}")
                    net.add_arc(P(fid), tid)
                    net.add_arc(tid, shared_out)

    throw_event_ids = [
        nid for nid, n in nodes.items() if n["kind"] == "compensateEndEvent"
    ]
    if len(throw_event_ids) > 1:
        raise ValueError(
            f"this scaffold supports at most one compensate end event per "
            f"process; got {len(throw_event_ids)}"
        )
    throw_mode = bool(throw_event_ids)
    throw_event_id = throw_event_ids[0] if throw_mode else None

    if throw_mode and not compensation_pairs:
        raise ValueError(
            f"compensate end event {throw_event_id!r} has no compensation "
            f"boundary event in the process to fire compensation for"
        )
    if throw_mode and len(compensation_pairs) > 1:
        raise ValueError(
            f"this scaffold supports only one compensation pair when a "
            f"compensate end event is present; got {len(compensation_pairs)}"
        )

    for task_id, bnd_id, handler_id in compensation_pairs:
        task = nodes[task_id]
        handler = nodes[handler_id]

        if len(task["incoming"]) != 1:
            raise ValueError(
                f"task {task_id!r} has a compensation boundary event but "
                f"{len(task['incoming'])} incoming flows; the §5 saga pattern "
                f"requires exactly one input place"
            )
        if handler["incoming"]:
            raise ValueError(
                f"compensation handler {handler_id!r} must not have incoming "
                f"sequenceFlows; it is triggered only by its boundary event"
            )
        if len(handler["outgoing"]) != 1:
            raise ValueError(
                f"compensation handler {handler_id!r} must have exactly one "
                f"outgoing flow (the recovery state); got "
                f"{len(handler['outgoing'])}"
            )

        p_compensating = f"{prefix}p_compensating_{bnd_id}"
        p_initial = P(handler["outgoing"][0])

        net.add_place(
            p_compensating,
            label=boundary_events[bnd_id]["name"] or f"compensating ({bnd_id})",
        )

        t_compensate = T(handler_id)
        net.add_transition(
            t_compensate, label=handler["name"] or handler_id
        )
        net.add_arc(p_compensating, t_compensate)
        net.add_arc(t_compensate, p_initial)

        if throw_mode:
            assert throw_event_id is not None
            throw_event = nodes[throw_event_id]
            if len(throw_event["incoming"]) != 1:
                raise ValueError(
                    f"compensate end event {throw_event_id!r} must have "
                    f"exactly one incoming flow"
                )

            p_completed = f"{prefix}p_completed_{task_id}"
            net.add_place(p_completed, label=f"completed: {task_id}")
            net.add_arc(T(task_id), p_completed)

            t_throw = T(throw_event_id)
            if t_throw not in net.transitions:
                net.add_transition(
                    t_throw,
                    label=throw_event["name"] or throw_event_id,
                )
                net.add_arc(P(throw_event["incoming"][0]), t_throw)
            net.add_arc(p_completed, t_throw)
            net.add_arc(t_throw, p_compensating)
        else:
            p_active = P(task["incoming"][0])
            t_fail = T(bnd_id)
            net.add_transition(
                t_fail, label=f"fail: {task['name'] or task_id}"
            )
            net.add_arc(p_active, t_fail)
            net.add_arc(t_fail, p_compensating)

    if node_to_pool is not None:
        for nid in nodes:
            node_to_pool[nid] = prefix
