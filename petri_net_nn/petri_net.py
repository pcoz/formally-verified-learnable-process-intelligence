"""Petri net data structure.

A finite Petri net N = (P, T, F, M_0) plus arc multiplicities — the
classical extension where one firing of a transition can consume
or produce more than one token per connected place.

Arcs without an explicit weight have weight 1, so nets built before
multi-token markings existed behave identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PetriNet:
    # Core 4-tuple (P, T, F, M_0) of the classical Petri net.
    places: set[str] = field(default_factory=set)
    transitions: set[str] = field(default_factory=set)
    flow: set[tuple[str, str]] = field(default_factory=set)
    initial_marking: dict[str, int] = field(default_factory=dict)

    # Human labels — used by the parser, the rule extractor, the
    # explanation prose, and anything reading the net for display.
    place_labels: dict[str, str] = field(default_factory=dict)
    transition_labels: dict[str, str] = field(default_factory=dict)

    # Sparse arc-weight table. Only arcs whose weight is *not* 1 get an
    # entry here, so the common case stays cheap and existing nets
    # built before Phase 9 (multi-token markings) keep weight-1
    # semantics with no migration.
    arc_multiplicities: dict[tuple[str, str], int] = field(default_factory=dict)

    # Inhibitor arcs: (place, transition) pairs where the place must
    # be EMPTY for the transition to fire. Inhibitor arcs are *not*
    # part of the flow relation — firing the transition does not
    # consume tokens from these places, they're only guards.
    inhibitor_arcs: set[tuple[str, str]] = field(default_factory=set)

    # Per-transition firing duration, measured in time-unrolled steps.
    # A transition with duration D, fired at step n, contributes its
    # output to place updates D-1 steps later (so D=1 means immediate,
    # the original behaviour). Like arc_multiplicities, we store only
    # the entries that differ from the default of 1 to keep the
    # common case cheap. Durations are meaningful only in the
    # compiler's time-unrolled forward pass; the discrete token-game
    # treats every firing as atomic.
    transition_durations: dict[str, int] = field(default_factory=dict)

    # Per-transition firing rate (default 1.0). Multiplies the
    # pre-activation in the compiler, so a transition with rate λ
    # behaves as if it had sharpness λ * net_sharpness — high rate
    # = steeper firing curve (more eager to fire for given inputs),
    # low rate = shallower curve (more conservative). Lets the
    # modeller encode prior knowledge about transition propensity
    # without losing the trainable threshold. Like the other Phase 9
    # extensions, only non-default entries are stored.
    transition_rates: dict[str, float] = field(default_factory=dict)

    def add_place(self, pid: str, *, label: str | None = None, tokens: int = 0) -> None:
        self.places.add(pid)
        if label is not None:
            self.place_labels[pid] = label
        if tokens:
            self.initial_marking[pid] = self.initial_marking.get(pid, 0) + tokens

    def add_transition(
        self,
        tid: str,
        *,
        label: str | None = None,
        duration: int = 1,
        rate: float = 1.0,
    ) -> None:
        """Add a transition.

        ``duration`` is the number of time-unrolled steps the
        transition takes to produce its output once it has fired
        (default 1 = immediate). Durations only have effect in the
        compiler's time-unrolled forward pass.

        ``rate`` is a per-transition firing-rate multiplier applied
        to the pre-activation by the compiler. The default 1.0
        leaves behaviour unchanged. A rate of 3.0 makes this
        transition fire roughly three times as eagerly as its
        siblings for the same inputs; 0.3 makes it three times less
        eager. Lets the modeller encode prior knowledge about
        transition propensity (priority, stochastic rate, etc.)
        alongside the learnable weights and thresholds.
        """
        if duration < 1:
            raise ValueError(
                f"transition {tid!r}: duration must be a positive integer, "
                f"got {duration}"
            )
        if rate <= 0:
            raise ValueError(
                f"transition {tid!r}: rate must be a positive number, "
                f"got {rate}"
            )
        self.transitions.add(tid)
        if label is not None:
            self.transition_labels[tid] = label
        if duration != 1:
            self.transition_durations[tid] = duration
        if rate != 1.0:
            self.transition_rates[tid] = float(rate)

    def duration(self, transition: str) -> int:
        """The transition's firing duration in time-unrolled steps.
        Returns 1 (immediate) for transitions added without an
        explicit duration."""
        return self.transition_durations.get(transition, 1)

    def rate(self, transition: str) -> float:
        """The transition's firing-rate multiplier. Returns 1.0 for
        transitions added without an explicit rate."""
        return self.transition_rates.get(transition, 1.0)

    def add_arc(self, src: str, dst: str, *, weight: int = 1) -> None:
        if weight < 1:
            raise ValueError(
                f"arc {src!r} -> {dst!r}: weight must be a positive integer, "
                f"got {weight}"
            )
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
        if weight != 1:
            self.arc_multiplicities[(src, dst)] = weight

    def weight(self, src: str, dst: str) -> int:
        """How many tokens this arc moves per firing. 1 unless an
        explicit weight was supplied to ``add_arc``."""
        return self.arc_multiplicities.get((src, dst), 1)

    def add_inhibitor_arc(self, place: str, transition: str) -> None:
        """Add an inhibitor arc from ``place`` to ``transition``. The
        transition can only fire when the place is empty; the
        transition does not consume tokens from the inhibitor place.
        Used to model mutual exclusion, negative preconditions, and
        guard-against-already-running patterns."""
        if place not in self.places:
            raise ValueError(f"inhibitor arc references unknown place {place!r}")
        if transition not in self.transitions:
            raise ValueError(
                f"inhibitor arc references unknown transition {transition!r}"
            )
        self.inhibitor_arcs.add((place, transition))

    def inhibitor_preset(self, transition: str) -> set[str]:
        """Places that inhibit ``transition`` — must be empty for it
        to fire."""
        return {p for p, t in self.inhibitor_arcs if t == transition}

    def preset(self, node: str) -> set[str]:
        return {src for src, dst in self.flow if dst == node}

    def postset(self, node: str) -> set[str]:
        return {dst for src, dst in self.flow if src == node}

    def is_enabled(self, transition: str, marking: dict[str, int]) -> bool:
        if transition not in self.transitions:
            raise KeyError(transition)
        if not all(
            marking.get(p, 0) >= self.weight(p, transition)
            for p in self.preset(transition)
        ):
            return False
        return all(
            marking.get(p, 0) == 0
            for p in self.inhibitor_preset(transition)
        )

    def fire(self, transition: str, marking: dict[str, int]) -> dict[str, int]:
        if not self.is_enabled(transition, marking):
            raise ValueError(f"transition {transition!r} is not enabled under marking")
        new = dict(marking)
        for p in self.preset(transition):
            new[p] = new.get(p, 0) - self.weight(p, transition)
            if new[p] == 0:
                del new[p]
        for p in self.postset(transition):
            new[p] = new.get(p, 0) + self.weight(transition, p)
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

        for (src, dst), weight in self.arc_multiplicities.items():
            if (src, dst) not in self.flow:
                issues.append(
                    f"arc multiplicity for {src!r} -> {dst!r} but no such arc "
                    f"in the flow relation"
                )
            if weight < 1:
                issues.append(
                    f"arc {src!r} -> {dst!r} has non-positive weight {weight}"
                )

        for place, transition in self.inhibitor_arcs:
            if place not in self.places:
                issues.append(
                    f"inhibitor arc references unknown place {place!r}"
                )
            if transition not in self.transitions:
                issues.append(
                    f"inhibitor arc references unknown transition {transition!r}"
                )

        for transition, duration in self.transition_durations.items():
            if transition not in self.transitions:
                issues.append(
                    f"duration recorded for unknown transition {transition!r}"
                )
            if duration < 1:
                issues.append(
                    f"transition {transition!r} has non-positive duration "
                    f"{duration}"
                )

        for transition, rate in self.transition_rates.items():
            if transition not in self.transitions:
                issues.append(
                    f"rate recorded for unknown transition {transition!r}"
                )
            if rate <= 0:
                issues.append(
                    f"transition {transition!r} has non-positive rate {rate}"
                )

        return issues
