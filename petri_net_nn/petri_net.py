"""Petri net data structure.

A finite Petri net N = (P, T, F, M_0) plus a number of classical
extensions: arc multiplicities (multi-token markings), inhibitor
arcs, transition durations, firing rates, and coloured tokens
(per-token scalar values with optional transition guards reading
those values).

Every extension is sparse — entries are recorded only when they
deviate from the trivial default, so nets that don't use the
extension behave identically to the basic four-tuple definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

# A transition guard reads the values of the tokens that would be
# consumed (one value per input place, keyed by place id) and
# returns True when the transition is allowed to fire.
GuardFn = Callable[[dict[str, float]], bool]

# An arc's output value — either a constant scalar or a callable
# that produces the output value from the input-place values bound
# at the firing transition.
OutputValueSpec = Union[float, Callable[[dict[str, float]], float]]


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

    # Coloured-Petri-net extensions. transition_guards holds optional
    # predicates that read the values of the tokens a transition
    # would consume and decide whether it is allowed to fire — used
    # by fire_coloured / is_enabled_coloured. arc_output_values gives
    # output arcs an optional value spec (constant or callable),
    # determining the value carried by the tokens the transition
    # produces. Both maps are sparse: a missing entry means
    # "no guard" / "produce value 1.0".
    transition_guards: dict[str, GuardFn] = field(default_factory=dict)
    arc_output_values: dict[tuple[str, str], OutputValueSpec] = field(
        default_factory=dict
    )

    # Structural guard records (``{place, op, value}``) for the
    # subset of guards that were declared in the simple comparison
    # form rather than as opaque Python callables. The compiler reads
    # this dict to build differentiable, learnable soft guards: each
    # entry becomes an ``nn.Parameter`` threshold initialised at
    # ``value``, refined by training. Guards declared via raw
    # callables remain transparent to the compiler (they're stored in
    # ``transition_guards`` and used by the coloured token-game only).
    transition_structural_guards: dict[str, dict] = field(default_factory=dict)

    # Torch-friendly soft guards: user-supplied callables that take a
    # dict mapping each input place to a (batch_size,) value tensor
    # and return a (batch_size,) gate tensor in [0, 1]. The compiler
    # multiplies the transition's firing strength by this gate. Lets
    # the modeller express routing logic the ``{place, op, value}``
    # structural form can't — multi-input comparisons, compound
    # predicates, custom learnable sub-networks. Has no effect on
    # the discrete token-game; the matching bool-returning entry in
    # ``transition_guards`` is what serves that path. If a transition
    # appears in both this dict and ``transition_structural_guards``,
    # the torch callable wins (it's more expressive).
    transition_torch_guards: dict[str, Callable] = field(default_factory=dict)

    # Torch-friendly arc output-value transforms: callables taking the
    # bound input-place value tensors and returning the value tensor
    # the produced token carries. Symmetric to ``arc_output_values``
    # but lives in tensor-land, so the compiler can use it during
    # training. Float-returning callables in ``arc_output_values``
    # remain token-game-only.
    arc_torch_output_values: dict[tuple[str, str], Callable] = field(
        default_factory=dict
    )

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
        guard: GuardFn | None = None,
        structural_guard: dict | None = None,
        torch_guard: Callable | None = None,
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

        ``guard`` is an optional callable used by the coloured
        token-game (``is_enabled_coloured`` / ``fire_coloured``).
        It receives a dict mapping each input place to the value of
        the token that would be consumed there and returns True
        when the transition is allowed to fire. Guards have no
        effect on the count-based ``is_enabled`` / ``fire`` path.

        ``structural_guard`` is the matching declarative record for
        the simple comparison-form guard — ``{"place": ..., "op":
        ..., "value": ...}``. When present, the compiler picks it
        up and builds a differentiable soft guard with a learnable
        threshold parameter initialised at ``value``. The token-game
        path keeps using ``guard`` unchanged, so the two views stay
        in sync: the declarative record is the trainable face of the
        same rule the callable encodes.

        ``torch_guard`` is a user-supplied torch-friendly soft gate:
        a callable taking ``dict[place_id, Tensor(batch,)]`` of input
        values and returning a ``Tensor(batch,)`` gate in [0, 1].
        The compiler multiplies the transition's firing strength by
        this gate. Use it for routing logic the structural form
        can't express — multi-input comparisons, compound predicates,
        custom learnable sub-networks. If both ``structural_guard``
        and ``torch_guard`` are supplied, the torch callable wins in
        the compiler (it's strictly more expressive). The token-game
        is unaffected by ``torch_guard`` either way.
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
        if guard is not None:
            self.transition_guards[tid] = guard
        if structural_guard is not None:
            for key in ("place", "op", "value"):
                if key not in structural_guard:
                    raise ValueError(
                        f"transition {tid!r}: structural_guard must contain "
                        f"{key!r}; got {structural_guard}"
                    )
            self.transition_structural_guards[tid] = dict(structural_guard)
        if torch_guard is not None:
            if not callable(torch_guard):
                raise ValueError(
                    f"transition {tid!r}: torch_guard must be callable, "
                    f"got {type(torch_guard).__name__}"
                )
            self.transition_torch_guards[tid] = torch_guard

    def duration(self, transition: str) -> int:
        """The transition's firing duration in time-unrolled steps.
        Returns 1 (immediate) for transitions added without an
        explicit duration."""
        return self.transition_durations.get(transition, 1)

    def rate(self, transition: str) -> float:
        """The transition's firing-rate multiplier. Returns 1.0 for
        transitions added without an explicit rate."""
        return self.transition_rates.get(transition, 1.0)

    def add_arc(
        self,
        src: str,
        dst: str,
        *,
        weight: int = 1,
        output_value: OutputValueSpec | None = None,
        torch_output_value: Callable | None = None,
    ) -> None:
        """Add an arc.

        ``weight`` is the multiplicity — how many tokens this arc
        moves per firing (default 1).

        ``output_value`` only applies to arcs *from* transitions to
        places. It specifies the value of the token the firing
        produces — a constant float, or a callable receiving the
        input-place values bound at the transition and returning
        a float. If omitted, the produced token carries value 1.0.
        The float / callable forms are evaluated by the coloured
        token-game (``fire_coloured``); count-based firing ignores
        them, and the compiler honours only the constant form.

        ``torch_output_value`` is the torch-friendly counterpart:
        a callable taking ``dict[place_id, Tensor(batch,)]`` of bound
        input values and returning a ``Tensor(batch,)`` that the
        compiler uses as the value carried by the produced token in
        the per-place value channel. Only applies to transition →
        place arcs. Doesn't participate in the discrete token-game
        (that's what ``output_value`` is for); the two can coexist —
        the token-game uses ``output_value`` and the compiler uses
        ``torch_output_value`` when present.
        """
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
        if output_value is not None:
            if src not in self.transitions:
                raise ValueError(
                    f"arc {src!r} -> {dst!r}: output_value only applies to "
                    f"transition -> place arcs (this arc starts at a place)"
                )
            self.arc_output_values[(src, dst)] = output_value
        if torch_output_value is not None:
            if src not in self.transitions:
                raise ValueError(
                    f"arc {src!r} -> {dst!r}: torch_output_value only applies "
                    f"to transition -> place arcs (this arc starts at a place)"
                )
            if not callable(torch_output_value):
                raise ValueError(
                    f"arc {src!r} -> {dst!r}: torch_output_value must be "
                    f"callable, got {type(torch_output_value).__name__}"
                )
            self.arc_torch_output_values[(src, dst)] = torch_output_value

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

    # --- Coloured token-game ----------------------------------------------
    # The methods below operate on a *coloured* marking — a dict whose
    # values are lists of per-token floats (a token's "colour") rather
    # than just an integer count. Tokens are consumed FIFO from each
    # input place; their values are bound by input-place id and made
    # available to (a) the transition guard, which decides whether the
    # transition is allowed to fire, and (b) any arc output-value
    # callables on outputs, which compute the value the produced
    # token carries. The compiler's continuous relaxation is
    # independent of all this — coloured tokens live at the structural
    # token-game level for now.

    def is_enabled_coloured(
        self, transition: str, marking: dict[str, list[float]]
    ) -> bool:
        """Coloured-token enablement: a transition is enabled iff every
        input place holds at least ``weight`` tokens, every inhibitor
        place is empty, *and* the transition's guard (if any) returns
        True for the values of the tokens that would be consumed."""
        if transition not in self.transitions:
            raise KeyError(transition)
        # Token-count enablement, weighted as for the scalar token game.
        for p in self.preset(transition):
            if len(marking.get(p, [])) < self.weight(p, transition):
                return False
        # Inhibitor places must be empty for the transition to fire.
        for p in self.inhibitor_preset(transition):
            if len(marking.get(p, [])) > 0:
                return False
        # Bind input values (FIFO: the token at index 0 from each
        # input place is the one that would be consumed first).
        guard = self.transition_guards.get(transition)
        if guard is None:
            return True
        input_values = {p: marking[p][0] for p in self.preset(transition)}
        return bool(guard(input_values))

    def fire_coloured(
        self, transition: str, marking: dict[str, list[float]]
    ) -> dict[str, list[float]]:
        """Fire ``transition`` against a coloured marking. Consumes
        ``weight`` tokens FIFO from each input place; produces tokens
        whose values come from each output arc's ``output_value``
        spec (constant float, or a callable receiving the bound input
        values). Returns a new marking — the input is not mutated."""
        if not self.is_enabled_coloured(transition, marking):
            raise ValueError(f"transition {transition!r} is not enabled under marking")
        new: dict[str, list[float]] = {p: list(tokens) for p, tokens in marking.items()}

        # Bind the values that would be consumed — these may be needed
        # by output-arc callables and we want to capture them BEFORE
        # mutating the marking.
        bound_inputs = {p: new[p][0] for p in self.preset(transition)}

        for p in self.preset(transition):
            for _ in range(self.weight(p, transition)):
                new[p].pop(0)
            if not new[p]:
                del new[p]

        for p in self.postset(transition):
            spec = self.arc_output_values.get((transition, p), 1.0)
            value = float(spec(bound_inputs) if callable(spec) else spec)
            weight = self.weight(transition, p)
            new.setdefault(p, []).extend([value] * weight)
        return new

    def enabled_transitions_coloured(
        self, marking: dict[str, list[float]]
    ) -> set[str]:
        return {
            t for t in self.transitions if self.is_enabled_coloured(t, marking)
        }

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

        for transition in self.transition_guards:
            if transition not in self.transitions:
                issues.append(
                    f"guard recorded for unknown transition {transition!r}"
                )

        for transition, spec in self.transition_structural_guards.items():
            if transition not in self.transitions:
                issues.append(
                    f"structural guard recorded for unknown transition "
                    f"{transition!r}"
                )
                continue
            place = spec.get("place")
            if place not in self.places:
                issues.append(
                    f"structural guard on transition {transition!r} "
                    f"references unknown place {place!r}"
                )
            if place not in self.preset(transition):
                issues.append(
                    f"structural guard on transition {transition!r} "
                    f"reads place {place!r} which is not in its preset"
                )

        for transition in self.transition_torch_guards:
            if transition not in self.transitions:
                issues.append(
                    f"torch guard recorded for unknown transition "
                    f"{transition!r}"
                )

        for (src, dst) in self.arc_torch_output_values:
            if (src, dst) not in self.flow:
                issues.append(
                    f"torch output value recorded for {src!r} -> {dst!r} "
                    f"but no such arc in the flow relation"
                )
            elif src not in self.transitions:
                issues.append(
                    f"torch output value on arc {src!r} -> {dst!r}: only "
                    f"transition -> place arcs may carry output values"
                )

        for (src, dst) in self.arc_output_values:
            if (src, dst) not in self.flow:
                issues.append(
                    f"output value recorded for {src!r} -> {dst!r} but no "
                    f"such arc in the flow relation"
                )
            elif src not in self.transitions:
                issues.append(
                    f"output value on arc {src!r} -> {dst!r}: only "
                    f"transition -> place arcs may carry output values"
                )

        return issues
