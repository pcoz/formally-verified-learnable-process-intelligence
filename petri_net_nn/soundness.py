"""Aalst soundness verification and deadlock localisation.

A net is **sound** in van der Aalst's sense iff:

1. **Option to complete.** From every reachable marking, the
   intended final marking is reachable. Nothing the net does is
   a one-way trip to a stuck state.

2. **Proper completion.** When tokens reach the sink, no tokens
   linger elsewhere. The final marking is the *only* marking
   with the expected sink token count.

3. **No dead transitions.** Every transition is enabled in at
   least one reachable marking. A transition that can never fire
   is a structural bug — either the modeller forgot a flow, or
   the transition itself is dead code.

PETRA's existing :func:`petri_net_nn.PetriNet.validate` catches
*well-formedness* (every arc references real nodes, every
transition has inputs and outputs, etc.). That's structural
sanity, not behavioural soundness. This module closes the gap:
behavioural soundness, computed by reachability-graph traversal.

The companion :func:`find_deadlocks` returns the *non-final*
markings the net can reach but can't progress from — a more
actionable, location-pinned form of the same information that
the soundness check's *option-to-complete* failure already
implies.

These checks are deliberately decoupled from training. They
analyse the *structure* of the net, not anything the trained
network has learned. Run them before training to catch
modeller-side bugs; run them after a refactoring to confirm the
new variant is still sound; bake them into CI to enforce
soundness on every scenario.

Scope: bounded Petri nets only — the reachability graph must
terminate. ``max_states`` (default 10,000) is the standard guard.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from petri_net_nn.bisimulation import Marking, _to_marking, reachability_graph
from petri_net_nn.petri_net import PetriNet


@dataclass(frozen=True)
class SoundnessReport:
    """Outcome of an Aalst soundness check.

    The three lists pin which condition failed and where. A net is
    sound iff all three are empty — checked via :attr:`is_sound`.

    Attributes
    ----------
    final_marking
        The marking the soundness check used as the intended
        completion state. Echoed back so the caller can confirm
        the heuristic picked what they expected.
    incomplete_markings
        Reachable markings from which the final marking cannot be
        reached. Each one is a one-way trip into a dead end —
        the *option-to-complete* property fails here.
    lingering_token_markings
        Reachable markings that have at least the final marking's
        token count at the sink but also carry tokens elsewhere.
        These break *proper completion* — the sink "fired" but
        work continues in parallel branches, suggesting an
        unsynchronised AND-split or a missed join.
    dead_transitions
        Transitions that are never enabled in any reachable
        marking. Either the structure has a missing input arc,
        the transition's preset is unreachable, or the transition
        itself is unnecessary.
    """

    final_marking: Marking
    incomplete_markings: list[Marking] = field(default_factory=list)
    lingering_token_markings: list[Marking] = field(default_factory=list)
    dead_transitions: list[str] = field(default_factory=list)

    @property
    def is_sound(self) -> bool:
        """``True`` iff none of the three soundness conditions
        failed. A sound net is one that, from every reachable
        marking, can complete cleanly with no lingering tokens
        and no dead transitions."""
        return (
            not self.incomplete_markings
            and not self.lingering_token_markings
            and not self.dead_transitions
        )

    def summary(self) -> str:
        """One-line human-readable digest of the report. Useful
        for log lines and assertion messages."""
        if self.is_sound:
            return "sound"
        pieces: list[str] = []
        if self.incomplete_markings:
            pieces.append(
                f"{len(self.incomplete_markings)} marking(s) cannot complete"
            )
        if self.lingering_token_markings:
            pieces.append(
                f"{len(self.lingering_token_markings)} marking(s) "
                f"have lingering tokens at completion"
            )
        if self.dead_transitions:
            pieces.append(
                f"{len(self.dead_transitions)} dead transition(s): "
                f"{', '.join(sorted(self.dead_transitions))}"
            )
        return "; ".join(pieces)


def _default_final_marking(net: PetriNet) -> Marking:
    """Pick the intended final marking heuristically from the net's
    sink places. A *sink* is a place with no outgoing arcs — the
    natural terminus of a workflow-net's control flow.

    Returns the marking that puts exactly one token on each sink
    place and zero elsewhere. Raises ``ValueError`` if no sinks
    exist; the caller should then supply ``final_marking``
    explicitly.
    """
    sinks = [p for p in net.places if not net.postset(p)]
    if not sinks:
        raise ValueError(
            "net has no sink places (places with no outgoing arcs); "
            "soundness needs an explicit final_marking argument when "
            "the natural sink is ambiguous"
        )
    return frozenset((p, 1) for p in sinks)


def _states_that_can_reach(
    target: Marking,
    states: frozenset[Marking],
    transitions: frozenset,
) -> set[Marking]:
    """Backward BFS over the LTS: the set of states from which
    ``target`` is reachable in zero or more transitions. Used by
    the option-to-complete check — every reachable state should be
    in this set.

    We invert the transition relation once (linear in |edges|)
    and then BFS from ``target`` over the inverted edges. That's
    O(|states| + |edges|), the same cost as a forward BFS."""
    if target not in states:
        # Target isn't even in the reachable state space — every
        # state fails to reach it.
        return set()
    backwards: dict[Marking, set[Marking]] = {s: set() for s in states}
    for src, _label, dst in transitions:
        backwards[dst].add(src)
    visited: set[Marking] = {target}
    queue: deque[Marking] = deque([target])
    while queue:
        s = queue.popleft()
        for pred in backwards[s]:
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)
    return visited


def check_soundness(
    net: PetriNet,
    *,
    final_marking: Marking | None = None,
    max_states: int = 10_000,
) -> SoundnessReport:
    """Run Aalst soundness on ``net``.

    Parameters
    ----------
    net
        The PetriNet to check. Must be bounded — the reachability
        graph has to terminate.
    final_marking
        The marking the net is meant to reach at completion. When
        omitted, the heuristic in :func:`_default_final_marking`
        picks one token at each sink place (no-outgoing-arcs).
        Pass an explicit marking when the sink is ambiguous or
        when the intended completion state has multiple sink
        tokens (rare).
    max_states
        Reachability-graph size cap; raises ``ValueError`` if the
        net is unbounded enough to exceed it.

    Returns
    -------
    SoundnessReport
        The pinned outcome — three lists (incomplete markings,
        lingering-token markings, dead transitions) plus a
        ``is_sound`` flag.
    """
    if final_marking is None:
        final_marking = _default_final_marking(net)

    # The LTS gives us reachable markings plus their labelled
    # transitions; we reuse it for all three checks.
    lts = reachability_graph(net, max_states=max_states)

    # Check 1: every reachable marking must be able to reach the
    # final marking. Compute the backward-reachable set from
    # final_marking; anything outside it is an "incomplete" marking.
    can_complete = _states_that_can_reach(
        final_marking, lts.states, lts.transitions
    )
    incomplete = sorted(
        (s for s in lts.states if s not in can_complete),
        key=lambda m: sorted(m),
    )

    # Check 2: proper completion. The sink token counts in
    # ``final_marking`` give us the expected completion-state
    # signature. Any reachable marking whose sink counts match-or-
    # exceed the final's sink counts must equal final_marking
    # exactly — otherwise the sink "fired" but other places still
    # hold tokens (a lingering AND branch, a missed synchronisation).
    final_dict = dict(final_marking)
    sink_places = set(final_dict.keys())
    lingering: list[Marking] = []
    for marking in lts.states:
        if marking == final_marking:
            continue
        m_dict = dict(marking)
        sinks_complete = all(
            m_dict.get(p, 0) >= n for p, n in final_dict.items()
        )
        if sinks_complete:
            # The sink has reached completion but the marking
            # disagrees with the final marking — either extra tokens
            # at the sink or tokens at non-sink places.
            lingering.append(marking)
    lingering.sort(key=lambda m: sorted(m))

    # Check 3: no dead transitions. A transition is dead iff it
    # never appears as an enabled choice in the reachability graph.
    # We could iterate the transitions field of the LTS, but those
    # carry LABELS rather than transition IDs — and PETRA allows
    # multiple transitions to share a label, so collisions are
    # possible. Instead, re-scan the reachable markings and check
    # each transition's enablement directly. O(|states| * |T|).
    enabled_somewhere: set[str] = set()
    for marking in lts.states:
        m_dict = dict(marking)
        for t in net.transitions:
            if t in enabled_somewhere:
                continue
            if net.is_enabled(t, m_dict):
                enabled_somewhere.add(t)
    dead = sorted(net.transitions - enabled_somewhere)

    return SoundnessReport(
        final_marking=final_marking,
        incomplete_markings=incomplete,
        lingering_token_markings=lingering,
        dead_transitions=dead,
    )


def find_deadlocks(
    net: PetriNet,
    *,
    final_marking: Marking | None = None,
    max_states: int = 10_000,
) -> list[Marking]:
    """Find non-final markings the net can reach but can't progress
    from.

    A deadlock is a marking with **no enabled transitions** that
    isn't the intended final marking. From a deadlock the net is
    stuck — no further firings are possible. Each deadlock is a
    specific token configuration you can hand a developer or
    process owner: "here's what the system was holding when it
    froze."

    This overlaps with the option-to-complete failure that
    :func:`check_soundness` reports, but the two are not identical.
    Soundness asks *"can every reachable state reach the final
    state?"*; deadlock localisation asks *"which states have
    nowhere to go?"*. Soundness reports a superset (any state
    that can only reach a deadlock fails option-to-complete too),
    but the deadlock list is the *root cause* pinned to specific
    markings.

    Parameters
    ----------
    net
        The PetriNet to inspect. Must be bounded.
    final_marking
        Excluded from the result — the final marking is *intended*
        to have no successors, so it's not a deadlock. Same
        default as :func:`check_soundness`.
    max_states
        Reachability-graph size cap.

    Returns
    -------
    list[Marking]
        Markings (sorted, for deterministic output) with no
        enabled transitions and not equal to the final marking.
        Empty list = no deadlocks.
    """
    if final_marking is None:
        final_marking = _default_final_marking(net)

    lts = reachability_graph(net, max_states=max_states)

    deadlocks: list[Marking] = []
    for marking in lts.states:
        if marking == final_marking:
            continue
        m_dict = dict(marking)
        # A state with at least one enabled transition is not a
        # deadlock; bail out as soon as we find one to keep the
        # scan O(|states| * |T|) worst-case but typically faster.
        any_enabled = any(
            net.is_enabled(t, m_dict) for t in net.transitions
        )
        if not any_enabled:
            deadlocks.append(marking)
    deadlocks.sort(key=lambda m: sorted(m))
    return deadlocks
