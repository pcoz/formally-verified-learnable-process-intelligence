"""Petri net data structure.

Implements the formal object from §2 of the architecture spec:

    N = (P, T, F, M_0)

where P is a finite set of places, T a finite set of transitions,
F ⊆ (P×T) ∪ (T×P) the flow relation, and M_0 the initial marking.

Token-game semantics (also §2) are provided so callers can simulate
execution of a parsed BPMN process before any neural compilation step.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PetriNet:
    places: set[str] = field(default_factory=set)
    transitions: set[str] = field(default_factory=set)
    flow: set[tuple[str, str]] = field(default_factory=set)
    initial_marking: dict[str, int] = field(default_factory=dict)
    place_labels: dict[str, str] = field(default_factory=dict)
    transition_labels: dict[str, str] = field(default_factory=dict)

    def add_place(self, pid: str, *, label: str | None = None, tokens: int = 0) -> None:
        self.places.add(pid)
        if label is not None:
            self.place_labels[pid] = label
        if tokens:
            self.initial_marking[pid] = self.initial_marking.get(pid, 0) + tokens

    def add_transition(self, tid: str, *, label: str | None = None) -> None:
        self.transitions.add(tid)
        if label is not None:
            self.transition_labels[tid] = label

    def add_arc(self, src: str, dst: str) -> None:
        if src in self.places and dst in self.transitions:
            pass
        elif src in self.transitions and dst in self.places:
            pass
        else:
            raise ValueError(
                f"arc {src!r} -> {dst!r} is not place->transition or "
                f"transition->place; one of the endpoints is unknown or both "
                f"are of the same kind"
            )
        self.flow.add((src, dst))

    def preset(self, node: str) -> set[str]:
        return {src for src, dst in self.flow if dst == node}

    def postset(self, node: str) -> set[str]:
        return {dst for src, dst in self.flow if src == node}

    def is_enabled(self, transition: str, marking: dict[str, int]) -> bool:
        if transition not in self.transitions:
            raise KeyError(transition)
        return all(marking.get(p, 0) >= 1 for p in self.preset(transition))

    def fire(self, transition: str, marking: dict[str, int]) -> dict[str, int]:
        if not self.is_enabled(transition, marking):
            raise ValueError(f"transition {transition!r} is not enabled under marking")
        new = dict(marking)
        for p in self.preset(transition):
            new[p] = new.get(p, 0) - 1
            if new[p] == 0:
                del new[p]
        for p in self.postset(transition):
            new[p] = new.get(p, 0) + 1
        return new

    def enabled_transitions(self, marking: dict[str, int]) -> set[str]:
        return {t for t in self.transitions if self.is_enabled(t, marking)}

    def validate(self) -> list[str]:
        """Return a list of structural issues. An empty list means the net
        is well-formed in the elementary sense: every arc connects known
        nodes, every transition has at least one input and one output,
        and the initial marking is supported on known places."""
        issues: list[str] = []

        for src, dst in self.flow:
            src_kind = (
                "place" if src in self.places
                else "transition" if src in self.transitions
                else None
            )
            dst_kind = (
                "place" if dst in self.places
                else "transition" if dst in self.transitions
                else None
            )
            if src_kind is None or dst_kind is None:
                issues.append(f"arc {src!r} -> {dst!r} references unknown node")
            elif src_kind == dst_kind:
                issues.append(f"arc {src!r} -> {dst!r} connects two {src_kind}s")

        for t in self.transitions:
            if not self.preset(t):
                issues.append(f"transition {t!r} has no input place")
            if not self.postset(t):
                issues.append(f"transition {t!r} has no output place")

        for p in self.initial_marking:
            if p not in self.places:
                issues.append(f"initial marking references unknown place {p!r}")

        return issues
