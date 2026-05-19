"""Strong bisimulation for bounded Petri nets.

Implements §7.3 of the architecture spec: a structural check that two
Petri nets exhibit the same observable behaviour from their initial
markings, where "observable" means the labels on their transitions.
The bisimulation check is the structural half of §7.3 — the
behavioural half (two bisimilar subnets converge to the same trained
function) is demonstrated in `test_bisimulation.py`.

Algorithm: reachability-graph + partition refinement. Each net's
labelled transition system is computed by BFS over markings from M_0;
the two LTSs are combined and refined into bisimulation equivalence
classes; the nets are bisimilar iff their initial markings end up in
the same class.

Scope: bounded Petri nets only. The BFS terminates exactly when the
reachable state space is finite, which is the case for every 1-bounded
workflow net produced by `parse_bpmn` today. A `max_states` cap raises
clearly if a caller feeds in something unbounded.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from petri_net_nn.petri_net import PetriNet


Marking = frozenset[tuple[str, int]]
LtsTransition = tuple[Marking, str, Marking]


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

    Each LTS state is a frozenset of (place, token-count) pairs; this
    canonical form lets bisimulation refinement use markings as dict
    keys. ``max_states`` guards against accidentally fed unbounded
    nets — it raises ``ValueError`` if the frontier grows past the
    cap."""
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


def bisimulation_equivalence_classes(net: PetriNet) -> list[frozenset[Marking]]:
    """Return the partition of ``net``'s reachable markings into
    bisimulation equivalence classes — markings from which the net
    exhibits the same labelled future behaviour."""
    lts = reachability_graph(net)
    return _refine_partition(lts.states, lts.transitions)


def are_bisimilar(net1: PetriNet, net2: PetriNet) -> bool:
    """Strong-bisimilarity check between the initial markings of two
    Petri nets. The two nets are bisimilar iff their initial markings
    fall into the same equivalence class of the combined LTS."""
    lts1 = reachability_graph(net1)
    lts2 = reachability_graph(net2)

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

    partition = _refine_partition(frozenset(tagged_states), tagged_transitions)

    initial1 = tag(lts1.initial, tag1)
    initial2 = tag(lts2.initial, tag2)
    for block in partition:
        if initial1 in block:
            return initial2 in block
    return False
