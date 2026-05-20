"""Strong and weak bisimulation for bounded Petri nets.

Implements §7.3 of the architecture spec — a structural check that
two Petri nets exhibit the same observable behaviour from their
initial markings, where "observable" means the labels on their
transitions. The strong checker (``are_bisimilar``) matches each
transition exactly; the weak checker (``are_weakly_bisimilar``)
collapses transitions marked ``silent=True`` so refactorings that
add or remove internal steps (logging, no-op gates, internal
handoffs) don't break the equivalence.

Algorithm: reachability-graph + partition refinement. Each net's
labelled transition system is computed by BFS over markings from
M_0; the two LTSs are combined and refined into bisimulation
equivalence classes; the nets are bisimilar iff their initial
markings end up in the same class.

For weak bisimulation, the LTS is **saturated** first: silent
transitions are relabelled to a τ sentinel, the τ-edges are
closed reflexive-transitively (every state gains a τ self-loop;
τ-paths become direct τ-edges), and each visible-action edge is
extended with τ-prefix and τ-suffix detours into all τ-reachable
neighbours. Strong bisimulation over the saturated LTS gives weak
bisimulation over the original.

Scope: bounded Petri nets only. The BFS terminates exactly when
the reachable state space is finite, which is the case for every
1-bounded workflow net produced by ``parse_bpmn`` today. A
``max_states`` cap raises clearly if a caller feeds in something
unbounded.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from petri_net_nn.petri_net import PetriNet


Marking = frozenset[tuple[str, int]]
LtsTransition = tuple[Marking, str, Marking]

# Sentinel used to relabel silent transitions in the LTS for weak
# bisimulation. The leading-underscore form keeps it from colliding
# with any plausible user-supplied transition label.
TAU = "__τ__"


@dataclass(frozen=True)
class LTS:
    """Labelled transition system view of a Petri net's reachable
    markings. The ``initial`` field is the canonical frozenset
    representation of M_0; ``states`` contains every reachable marking;
    ``transitions`` records every (from, label, to) edge in the
    reachability graph."""

    initial: Marking
    states: frozenset[Marking]
    transitions: frozenset[LtsTransition]


def _to_marking(d: dict[str, int]) -> Marking:
    return frozenset((p, c) for p, c in d.items() if c > 0)


def reachability_graph(net: PetriNet, *, max_states: int = 10_000) -> LTS:
    """Compute the reachable LTS of ``net`` from its initial marking.

    Each LTS state is a frozenset of (place, token-count) pairs;
    this canonical form lets bisimulation refinement use markings
    as dict keys. ``max_states`` guards against accidentally fed
    unbounded nets — it raises ``ValueError`` if the frontier grows
    past the cap.

    Silent transitions appear with their declared label, exactly
    as visible transitions do; the strong checker treats them as
    ordinary edges. Use ``_reachability_graph_with_tau`` (the weak
    checker calls it) to relabel silent transitions to ``TAU``.
    """
    return _lts(net, silent_as_tau=False, max_states=max_states)


def _lts(net: PetriNet, *, silent_as_tau: bool, max_states: int) -> LTS:
    """Internal reachability-graph builder shared by the strong and
    weak checkers. With ``silent_as_tau=True``, transitions in
    ``net.silent_transitions`` produce edges labelled ``TAU`` in
    the LTS instead of their declared label — the relabelling the
    weak-bisimulation saturation step expects."""
    initial = _to_marking(net.initial_marking)
    states: set[Marking] = {initial}
    transitions: set[LtsTransition] = set()
    queue: list[Marking] = [initial]

    while queue:
        m = queue.pop()
        m_dict = dict(m)
        for t in net.transitions:
            if not net.is_enabled(t, m_dict):
                continue
            m2_dict = net.fire(t, m_dict)
            m2 = _to_marking(m2_dict)
            if silent_as_tau and t in net.silent_transitions:
                label = TAU
            else:
                label = net.transition_labels.get(t, t)
            transitions.add((m, label, m2))
            if m2 not in states:
                states.add(m2)
                if len(states) > max_states:
                    raise ValueError(
                        f"reachability graph exceeded {max_states} states; "
                        f"the net may be unbounded"
                    )
                queue.append(m2)

    return LTS(
        initial=initial,
        states=frozenset(states),
        transitions=frozenset(transitions),
    )


def _refine_partition(
    states: frozenset, transitions: frozenset
) -> list[frozenset]:
    """Partition-refinement bisimulation over a labelled transition
    system. Returns the list of equivalence classes (largest
    bisimulation)."""
    if not states:
        return []

    successors: dict[object, dict[str, set]] = defaultdict(
        lambda: defaultdict(set)
    )
    for src, label, dst in transitions:
        successors[src][label].add(dst)

    state_class: dict[object, int] = {s: 0 for s in states}
    blocks: dict[int, set] = {0: set(states)}

    while True:
        new_blocks: dict[int, set] = {}
        new_state_class: dict[object, int] = {}
        next_block_id = 0
        any_split = False

        for block in blocks.values():
            buckets: dict[frozenset, set] = defaultdict(set)
            for s in block:
                signature = frozenset(
                    (label, state_class[dst])
                    for label, dsts in successors[s].items()
                    for dst in dsts
                )
                buckets[signature].add(s)
            if len(buckets) > 1:
                any_split = True
            for bucket in buckets.values():
                new_blocks[next_block_id] = bucket
                for s in bucket:
                    new_state_class[s] = next_block_id
                next_block_id += 1

        blocks = new_blocks
        state_class = new_state_class
        if not any_split:
            break

    return [frozenset(b) for b in blocks.values()]


def _tau_closure(
    states: frozenset, transitions: frozenset
) -> dict[object, set]:
    """Reflexive transitive closure of the τ-edge relation.

    Returns a dict mapping each state to the set of states reachable
    from it via zero or more τ-labelled transitions (the empty path
    means the state is always in its own closure).
    """
    tau_succ: dict[object, set] = defaultdict(set)
    for src, label, dst in transitions:
        if label == TAU:
            tau_succ[src].add(dst)

    closure: dict[object, set] = {s: {s} for s in states}
    # Iterative fixed-point. For each state, expand its closure by
    # the closures of its τ-successors until nothing grows.
    changed = True
    while changed:
        changed = False
        for s in states:
            new_reach = set(closure[s])
            for via in list(closure[s]):
                new_reach |= tau_succ[via]
            if new_reach != closure[s]:
                closure[s] = new_reach
                changed = True
    return closure


def _saturate(lts: LTS) -> LTS:
    """Saturate an LTS for weak bisimulation.

    For weak bisimulation, the matching condition is on **weak**
    transitions ``⇒^a`` defined as ``⇒^τ → →^a → ⇒^τ`` (visible
    actions can be flanked by arbitrary τ-paths on either side).
    Saturating the LTS turns weak transitions into direct edges,
    after which strong bisimulation over the saturated LTS is
    exactly weak bisimulation over the original.

    The saturation adds:

    * a τ self-loop on every state (the reflexive part of ``⇒^τ``);
    * a τ-edge from every state to every state in its τ-closure;
    * for each original visible-action edge ``s →^a s'``, weak
      edges ``s'' →^a s'''`` for every ``s''`` in the τ-closure
      preimage of ``s`` and every ``s'''`` in the τ-closure of
      ``s'``.
    """
    closure = _tau_closure(lts.states, lts.transitions)

    saturated: set[LtsTransition] = set()

    # Every state has a τ self-loop (reflexivity of ⇒^τ); plus a
    # τ-edge for every other state in its τ-closure.
    for s in lts.states:
        for s_prime in closure[s]:
            saturated.add((s, TAU, s_prime))

    # For visible-action edges, add the τ-closure-flanked weak
    # transitions. Pre-compute the τ-closure preimage so we can
    # find "states that τ-reach the source of this visible edge".
    tau_preimage: dict[object, set] = defaultdict(set)
    for s, reach in closure.items():
        for r in reach:
            tau_preimage[r].add(s)

    for src, label, dst in lts.transitions:
        if label == TAU:
            continue
        for pre in tau_preimage[src]:
            for post in closure[dst]:
                saturated.add((pre, label, post))

    return LTS(
        initial=lts.initial,
        states=lts.states,
        transitions=frozenset(saturated),
    )


def bisimulation_equivalence_classes(net: PetriNet) -> list[frozenset[Marking]]:
    """Return the partition of ``net``'s reachable markings into
    bisimulation equivalence classes — markings from which the net
    exhibits the same labelled future behaviour."""
    lts = reachability_graph(net)
    return _refine_partition(lts.states, lts.transitions)


def are_bisimilar(net1: PetriNet, net2: PetriNet) -> bool:
    """Strong-bisimilarity check between the initial markings of two
    Petri nets.

    The two nets are bisimilar iff their initial markings fall into
    the same equivalence class of the combined LTS — that is, every
    labelled successor of one can be matched exactly by a labelled
    successor of the other, recursively. "Exactly" includes silent
    transitions, which behave like any other labelled edge here; use
    :func:`are_weakly_bisimilar` if you want silent transitions
    collapsed.
    """
    # Two separate LTSs that we'll combine. Tagging the states keeps
    # net1's and net2's markings disjoint in the combined LTS, so
    # that an accidental marking-collision between the two nets
    # doesn't pre-merge their state spaces.
    lts1 = reachability_graph(net1)
    lts2 = reachability_graph(net2)
    return _initials_in_same_class(lts1, lts2, saturate=False)


def are_weakly_bisimilar(net1: PetriNet, net2: PetriNet) -> bool:
    """Weak-bisimilarity check between the initial markings of two
    Petri nets.

    Weak bisimulation differs from strong bisimulation in one
    important way: transitions marked ``silent=True`` are treated as
    invisible internal steps that can be inserted or removed without
    changing the observable behaviour. A refactoring that adds an
    internal logging transition, a no-op routing gate, or any other
    structural artefact whose label nobody cares about will be
    *strongly* bisimilar-rejected but *weakly* bisimilar-accepted —
    which is what makes the cost-ranked refactoring story actually
    work for real-world redesigns.

    Internally the LTS for each net is built with silent transitions
    relabelled to the :data:`TAU` sentinel; both LTSs are then
    saturated (see :func:`_saturate` for the algorithm); strong
    bisimulation over the saturated combined LTS is weak bisimulation
    over the originals.
    """
    # Build each net's LTS with silent transitions collapsed to τ.
    # The strong-bisimulation public API does not relabel — silent
    # is purely a weak-bisimulation concept.
    lts1 = _lts(net1, silent_as_tau=True, max_states=10_000)
    lts2 = _lts(net2, silent_as_tau=True, max_states=10_000)
    return _initials_in_same_class(lts1, lts2, saturate=True)


def _initials_in_same_class(
    lts1: LTS,
    lts2: LTS,
    *,
    saturate: bool,
) -> bool:
    """Combine two LTSs (tagged to keep their states disjoint),
    optionally saturate the result for weak bisimulation, refine into
    bisimulation equivalence classes, and report whether the two
    initial markings end up in the same class.

    This is the shared spine of both :func:`are_bisimilar` and
    :func:`are_weakly_bisimilar` — they differ only in whether
    they pre-relabel silent transitions to τ (the LTS construction
    step) and whether they saturate before refinement (this step).
    """
    # Tag each net's states so we can tell them apart in the
    # combined LTS even if they happen to share a marking value
    # (which is common for two structurally-similar nets).
    tag1 = ("__net1__",)
    tag2 = ("__net2__",)

    def tag(state: Marking, t: tuple) -> tuple:
        return (t, state)

    tagged_states = (
        {tag(s, tag1) for s in lts1.states}
        | {tag(s, tag2) for s in lts2.states}
    )
    tagged_transitions = frozenset(
        (tag(src, tag1), label, tag(dst, tag1))
        for src, label, dst in lts1.transitions
    ) | frozenset(
        (tag(src, tag2), label, tag(dst, tag2))
        for src, label, dst in lts2.transitions
    )

    if saturate:
        # Saturating the combined LTS — rather than each LTS
        # individually — lets τ-paths within either net build up
        # weak transitions correctly. Saturating per-net first
        # would still work because we never cross from one net's
        # tagged states to the other's, but doing it once on the
        # combined LTS is simpler and equivalent.
        combined = LTS(
            initial=tag(lts1.initial, tag1),  # arbitrary; unused
            states=frozenset(tagged_states),
            transitions=tagged_transitions,
        )
        saturated = _saturate(combined)
        partition = _refine_partition(
            saturated.states, saturated.transitions
        )
    else:
        partition = _refine_partition(
            frozenset(tagged_states), tagged_transitions
        )

    initial1 = tag(lts1.initial, tag1)
    initial2 = tag(lts2.initial, tag2)
    for block in partition:
        if initial1 in block:
            return initial2 in block
    # Defensive — every state lives in exactly one block of the
    # final partition, so this fall-through is unreachable given a
    # well-formed LTS. Returning False on the impossible path keeps
    # the type-checker happy.
    return False


def weak_bisimulation_equivalence_classes(
    net: PetriNet,
) -> list[frozenset[Marking]]:
    """The partition of ``net``'s reachable markings into **weak**
    bisimulation equivalence classes.

    Same idea as :func:`bisimulation_equivalence_classes`, but two
    markings are in the same class if they exhibit the same
    visible-action behaviour up to silent τ-detours. Useful for
    asking "how many distinguishable observable states does this
    net have?", which collapses transient internal states into
    the equivalence class of whatever they τ-reach.
    """
    lts = _lts(net, silent_as_tau=True, max_states=10_000)
    saturated = _saturate(lts)
    return _refine_partition(saturated.states, saturated.transitions)
