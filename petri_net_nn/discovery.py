"""Inductive process mining — discover a sound Petri net from
event logs.

Implementation of the basic Inductive Miner (Leemans, Fahland,
van der Aalst, 2013, *"Discovering Block-Structured Process
Models from Event Logs - A Constructive Approach"*). Given a
list of traces (each a sequence of activity names from the
``event.name`` field), recursively find structural cuts in the
directly-follows graph until a process tree emerges, then
translate the process tree into a Petri net.

The output net is **sound by construction** — option to
complete, proper completion, and no dead transitions all hold
for any output of this algorithm. That guarantee composes
cleanly with the rest of PETRA's pipeline:
:func:`discover_and_train` chains discovery, a soundness check,
compilation, and training in one call.

Algorithm sketch
----------------

1. From the trace set, build the **directly-follows graph**
   (DFG) — nodes are activities, an edge ``(a, b)`` means
   ``a`` was directly followed by ``b`` in some trace.

2. Try the four cut shapes in priority order:

   - **Exclusive choice** (``×``): activities partition into
     disjoint groups that never co-occur in a single trace.
     Detected as connected components of the undirected DFG.

   - **Sequence** (``→``): activities partition into ordered
     groups ``Σ₁, …, Σₙ`` such that for each pair ``i<j``,
     every activity in ``Σᵢ`` reaches every activity in
     ``Σⱼ`` via DFG paths and none in ``Σⱼ`` reaches any in
     ``Σᵢ``. Detected via strongly-connected components and a
     total topological ordering.

   - **Parallel** (``∧``): activities partition into groups
     such that every cross-group pair ``(a, b)`` has both
     ``a→b`` and ``b→a`` in the DFG, and every group contains
     at least one trace-start activity and one trace-end
     activity.

   - **Loop** (``↻``): designate one group as the body and the
     rest as redo parts. The body alone contains all start
     and end activities of the log; the redos can only be
     entered after the body ends and exited by re-entering
     the body.

3. If a cut is found, **split the log** accordingly and
   recurse on each sub-log.

4. **Base cases:**

   - Empty log → ``Tau`` (silent transition).
   - Single activity log → ``Activity`` leaf.

5. **Fallback:** when no cut applies, return a *flower model*
   — an XOR of all activities inside a self-loop. This is a
   degenerate output that always accepts the log; it appears
   when the log is too unstructured for the cut detectors to
   handle.

Limitations of this first delivery
----------------------------------

- **Basic IM only.** Not IMf (infrequent-noise filtering) or
  IMi (incremental). Noisy logs may collapse into the
  flower-model fallback.
- **Loop cut handles body+redo pairs.** Nested loops work via
  recursion, but esoteric loop topologies may degrade.
- **Timestamps are ignored** — only activity sequences matter
  for the structural discovery. PETRA's later training step
  can read attributes via the standard ``train_on_traces``
  path.
- **Event attributes are not used** for discovery.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Union

from petri_net_nn.petri_net import PetriNet
from petri_net_nn.xes import XESTrace


# ---------------------------------------------------------------------------
# Process tree.
#
# Five node types — frozen dataclasses so the tree is hashable
# and the type-tag is the class itself. The recursive miner
# returns one of these; the translator below walks the tree to
# emit a PetriNet.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tau:
    """Silent / τ transition — produced for empty sub-logs and
    for the τ branches of loop and exclusive cuts that allow
    skipping."""

    def __repr__(self) -> str:
        return "τ"


@dataclass(frozen=True)
class Activity:
    """Leaf node carrying a single activity name from the log.
    Becomes a single labelled transition in the emitted net."""

    name: str

    def __repr__(self) -> str:
        return repr(self.name)


@dataclass(frozen=True)
class Sequence:
    """``→`` — children execute one after the other, in order."""

    children: tuple["ProcessTree", ...]

    def __repr__(self) -> str:
        return f"→({', '.join(repr(c) for c in self.children)})"


@dataclass(frozen=True)
class ExclusiveChoice:
    """``×`` — exactly one of the children executes."""

    children: tuple["ProcessTree", ...]

    def __repr__(self) -> str:
        return f"×({', '.join(repr(c) for c in self.children)})"


@dataclass(frozen=True)
class Parallel:
    """``∧`` — all children execute concurrently and synchronise
    at the join."""

    children: tuple["ProcessTree", ...]

    def __repr__(self) -> str:
        return f"∧({', '.join(repr(c) for c in self.children)})"


@dataclass(frozen=True)
class Loop:
    """``↻`` — ``children[0]`` is the body; ``children[1:]`` are
    redo parts. The body executes; from its end, either continue
    (loop exits) or do one of the redos and re-enter the body.

    A loop with no explicit redo is equivalent to ``Loop(body,
    Tau())`` — every body iteration can be followed by either
    another body execution or termination."""

    children: tuple["ProcessTree", ...]

    def __repr__(self) -> str:
        return f"↻({', '.join(repr(c) for c in self.children)})"


ProcessTree = Union[Tau, Activity, Sequence, ExclusiveChoice, Parallel, Loop]


# ---------------------------------------------------------------------------
# Directly-follows graph and derived relations.
#
# A log here is a list of "activity sequences" — each trace is
# a tuple of activity-name strings. The DFG is the multiset of
# pairs (a, b) such that a is directly followed by b in some
# trace; we store the set of edges (presence, not frequency)
# because the basic IM doesn't use frequencies.
# ---------------------------------------------------------------------------


Trace = tuple[str, ...]
Log = list[Trace]


def _build_dfg(log: Log) -> set[tuple[str, str]]:
    """Edges of the directly-follows graph for ``log`` — the set
    of pairs ``(a, b)`` such that ``a`` is followed directly by
    ``b`` in at least one trace."""
    edges: set[tuple[str, str]] = set()
    for trace in log:
        for a, b in zip(trace, trace[1:]):
            edges.add((a, b))
    return edges


def _activities(log: Log) -> set[str]:
    """All distinct activity names that appear in the log."""
    out: set[str] = set()
    for trace in log:
        out.update(trace)
    return out


def _start_activities(log: Log) -> set[str]:
    """Activities that begin at least one (non-empty) trace."""
    return {t[0] for t in log if t}


def _end_activities(log: Log) -> set[str]:
    """Activities that end at least one (non-empty) trace."""
    return {t[-1] for t in log if t}


def _reachability(
    activities: set[str], edges: set[tuple[str, str]]
) -> dict[str, set[str]]:
    """Transitive closure of the DFG. Returns a dict mapping each
    activity to the set of activities reachable from it (excluding
    itself unless there's a self-cycle path back). Used by the
    sequence and loop cut detectors."""
    successors: dict[str, set[str]] = {a: set() for a in activities}
    for src, dst in edges:
        if src in successors and dst in activities:
            successors[src].add(dst)

    closure: dict[str, set[str]] = {a: set() for a in activities}
    for start in activities:
        frontier = {start}
        seen: set[str] = set()
        while frontier:
            node = frontier.pop()
            for nxt in successors[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.add(nxt)
        closure[start] = seen
    return closure


# ---------------------------------------------------------------------------
# Cut detectors and log splitters.
#
# Each detector returns either a list of activity groups (the
# cut) or None (no cut found). The corresponding splitter then
# projects the log according to the cut semantics.
# ---------------------------------------------------------------------------


def _exclusive_choice_cut(
    activities: set[str], edges: set[tuple[str, str]]
) -> list[set[str]] | None:
    """Connected components of the undirected DFG. If there's
    more than one, the activities partition exclusively — each
    trace lives entirely in one component."""
    if len(activities) < 2:
        return None
    # Undirected adjacency.
    adj: dict[str, set[str]] = {a: set() for a in activities}
    for src, dst in edges:
        if src in adj and dst in adj:
            adj[src].add(dst)
            adj[dst].add(src)
    # BFS components.
    components: list[set[str]] = []
    unvisited = set(activities)
    while unvisited:
        start = next(iter(unvisited))
        component: set[str] = set()
        frontier = {start}
        while frontier:
            node = frontier.pop()
            if node in component:
                continue
            component.add(node)
            frontier.update(adj[node] - component)
        components.append(component)
        unvisited -= component
    if len(components) < 2:
        return None
    # Deterministic ordering — sort by the lexicographically-
    # smallest member so the test outputs are stable.
    components.sort(key=lambda c: min(c))
    return components


def _split_exclusive(
    log: Log, groups: list[set[str]]
) -> list[Log]:
    """For each group, the traces that lie entirely within it."""
    sub_logs: list[Log] = [[] for _ in groups]
    for trace in log:
        for i, group in enumerate(groups):
            if all(activity in group for activity in trace):
                sub_logs[i].append(trace)
                break
    return sub_logs


def _sequence_cut(
    activities: set[str], edges: set[tuple[str, str]]
) -> list[set[str]] | None:
    """Find a totally-ordered partition Σ₁, …, Σₙ such that
    each Σᵢ reaches every later Σⱼ but no later Σⱼ reaches any
    earlier Σᵢ.

    Implementation: compute pairwise reachability, collapse
    activities into mutual-reachability equivalence classes
    (SCCs of the DFG), build the condensation DAG, and run
    Kahn's algorithm on it. A sequence cut requires the
    condensation to be a *linear* chain — at every step of
    Kahn's, exactly one source class must be available. If
    Kahn's finds more than one ready class at any step, the
    SCCs form a partial order with branches, not a sequence,
    and we return None (the parallel cut will catch the
    independent-branches case)."""
    if len(activities) < 2:
        return None
    reach = _reachability(activities, edges)

    # Equivalence: a ≡ b iff a reaches b and b reaches a (or
    # a == b). Each equivalence class is an SCC.
    equiv_class: dict[str, frozenset[str]] = {}
    for a in activities:
        if a in equiv_class:
            continue
        members = {a}
        for b in activities:
            if b != a and b in reach[a] and a in reach[b]:
                members.add(b)
        frozen = frozenset(members)
        for m in members:
            equiv_class[m] = frozen

    classes = list({equiv_class[a] for a in activities})
    if len(classes) < 2:
        return None

    # Build the DAG of classes. [c1] → [c2] iff there exist
    # a ∈ c1, b ∈ c2 (c1 ≠ c2) with a reaches b. Because of how
    # equivalence is defined, this gives a strict DAG (no cycles).
    class_adj: dict[frozenset[str], set[frozenset[str]]] = {
        c: set() for c in classes
    }
    in_degree: dict[frozenset[str], int] = {c: 0 for c in classes}
    for c1 in classes:
        for c2 in classes:
            if c1 == c2:
                continue
            if any(b in reach[a] for a in c1 for b in c2):
                if c2 not in class_adj[c1]:
                    class_adj[c1].add(c2)
                    in_degree[c2] += 1

    # Level-based antichain decomposition of the SCC DAG.
    # Each SCC's level is 1 + max(level of predecessors); SCCs
    # at the same level form a sequence group together.
    #
    # The "every activity in Σᵢ reaches every activity in Σᵢ₊₁"
    # constraint is then explicitly verified below — the level
    # decomposition gives the right *structure* but not all
    # level-decomposed partitions are valid sequence cuts
    # (e.g. a → b → d and a → c → e has b and c at the same
    # level but b doesn't reach e). We reject those cases here.
    #
    # The pure-uniqueness-of-sources approach was simpler but
    # rejected valid sequence cuts where the middle group is
    # a convergent set of independent branches — e.g. log
    # [(a,b,d), (a,c,d)] has the SCC DAG a → {b,c} → d which
    # decomposes correctly as [{a}, {b,c}, {d}].
    pred_class_adj: dict[frozenset[str], set[frozenset[str]]] = {
        c: set() for c in classes
    }
    for c1, succs in class_adj.items():
        for c2 in succs:
            pred_class_adj[c2].add(c1)

    level: dict[frozenset[str], int] = {}
    remaining = set(classes)
    while remaining:
        # Sources first: classes whose predecessors are all
        # already levelled (i.e. no predecessors among
        # ``remaining``).
        sources = {
            c for c in remaining
            if not (pred_class_adj[c] & remaining)
        }
        if not sources:
            # Cycle in the class graph — shouldn't happen
            # since equivalence classes are SCCs, but guards
            # against bugs in the equivalence computation.
            return None
        for c in sources:
            if not pred_class_adj[c]:
                level[c] = 0
            else:
                level[c] = 1 + max(level[p] for p in pred_class_adj[c])
        remaining -= sources

    max_level = max(level.values())
    if max_level == 0:
        # All classes at level 0 — only happens when there's a
        # single class, but we already returned None for that
        # case at the top. Defence-in-depth.
        return None

    # Build the per-level groups in order.
    groups_by_level: list[set[str]] = [set() for _ in range(max_level + 1)]
    for c, lvl in level.items():
        groups_by_level[lvl].update(c)

    # Validity check: every activity in Σᵢ must reach every
    # activity in Σᵢ₊₁ in the DFG closure. Without this, the
    # level decomposition can group activities that don't
    # actually sequence cleanly.
    for i in range(len(groups_by_level) - 1):
        for a in groups_by_level[i]:
            for b in groups_by_level[i + 1]:
                if b not in reach[a]:
                    return None

    return groups_by_level


def _split_sequence(
    log: Log, groups: list[set[str]]
) -> list[Log]:
    """Project each trace onto each group in order — keep only
    the events whose activity is in the group, preserving the
    original ordering."""
    sub_logs: list[Log] = [[] for _ in groups]
    for trace in log:
        for i, group in enumerate(groups):
            projected = tuple(a for a in trace if a in group)
            sub_logs[i].append(projected)
    return sub_logs


def _parallel_cut(
    activities: set[str],
    edges: set[tuple[str, str]],
    log: Log,
) -> list[set[str]] | None:
    """Activities a and b are "parallel" iff both (a→b) and
    (b→a) appear in the DFG. The non-parallel graph (edges
    where this *fails*) has connected components that must
    stay together; if the non-parallel graph has multiple
    components AND every component contains at least one start
    and one end activity, that's a valid parallel cut."""
    if len(activities) < 2:
        return None
    edge_set = edges

    def parallel(a: str, b: str) -> bool:
        return (a, b) in edge_set and (b, a) in edge_set

    # Edges of the non-parallel graph: pairs that are NOT
    # parallel. Connected components of this graph are the
    # groups that cannot be separated by a parallel cut.
    non_parallel_adj: dict[str, set[str]] = {a: set() for a in activities}
    for a in activities:
        for b in activities:
            if a < b and not parallel(a, b):
                non_parallel_adj[a].add(b)
                non_parallel_adj[b].add(a)

    # Connected components of the non-parallel graph.
    components: list[set[str]] = []
    unvisited = set(activities)
    while unvisited:
        start = next(iter(unvisited))
        component: set[str] = set()
        frontier = {start}
        while frontier:
            node = frontier.pop()
            if node in component:
                continue
            component.add(node)
            frontier.update(non_parallel_adj[node] - component)
        components.append(component)
        unvisited -= component

    if len(components) < 2:
        return None

    # Validity check: each component must contain at least one
    # start activity and at least one end activity of the log
    # (otherwise the parallel composition wouldn't faithfully
    # represent the start/end behaviour).
    starts = _start_activities(log)
    ends = _end_activities(log)
    for component in components:
        if not (component & starts):
            return None
        if not (component & ends):
            return None

    components.sort(key=lambda c: min(c))
    return components


def _split_parallel(
    log: Log, groups: list[set[str]]
) -> list[Log]:
    """Same as sequence: project each trace onto each group."""
    return _split_sequence(log, groups)


def _loop_cut(
    activities: set[str],
    edges: set[tuple[str, str]],
    log: Log,
) -> list[set[str]] | None:
    """Detect a loop cut: a body Σ₁ containing the start and
    end activities of the log, plus one or more redo parts that
    only connect to the body via "from end-of-log activity →
    start-of-redo" and "from end-of-redo → start-of-log
    activity" edges.

    The body is **minimal** — exactly ``start_activities ∪
    end_activities``. This is the right call for the basic IM:
    a larger body candidate (e.g. ``forward∩backward`` from the
    start/end sets) collapses obvious loop patterns like
    ``(a, b, a, b, a)`` into a single block with no redos, so
    the algorithm falls through to the flower model. Pinning
    the body at start∪end and pushing every other activity into
    a redo component preserves loop structure when it exists.
    The validity check on the cross-edges (below) rejects
    candidates that aren't actually loops, so this minimalism
    doesn't produce false positives.
    """
    starts = _start_activities(log)
    ends = _end_activities(log)
    if not (starts and ends):
        return None
    if len(activities) < 2:
        return None
    edge_set = edges

    # Minimal body: the activities that begin or end any trace.
    # Activities outside this set are candidate redo activities.
    body = starts | ends
    if body == activities:
        # No activities outside start/end — nothing to put in a
        # redo. Not a loop cut. (For self-looping single
        # activities like trace `(a, a, a)`, the base case in
        # `_mine` wraps in `Loop(a, Tau)` directly.)
        return None

    redo_activities = activities - body

    # Find connected components of the redo activities in the
    # DFG restricted to them.
    redo_adj: dict[str, set[str]] = {a: set() for a in redo_activities}
    for src, dst in edge_set:
        if src in redo_activities and dst in redo_activities:
            redo_adj[src].add(dst)
            redo_adj[dst].add(src)
    redo_components: list[set[str]] = []
    unvisited = set(redo_activities)
    while unvisited:
        start = next(iter(unvisited))
        component: set[str] = set()
        frontier = {start}
        while frontier:
            node = frontier.pop()
            if node in component:
                continue
            component.add(node)
            frontier.update(redo_adj[node] - component)
        redo_components.append(component)
        unvisited -= component

    # Validate each redo component: must connect to the body
    # only via "end-of-body → start-of-redo" and "end-of-redo
    # → start-of-body" edges.
    for redo in redo_components:
        # Edges from body to this redo must originate from
        # end-of-body activities, and land on start-of-this-redo.
        for src, dst in edge_set:
            if src in body and dst in redo:
                if src not in ends:
                    return None
            if src in redo and dst in body:
                if dst not in starts:
                    return None

    redo_components.sort(key=lambda c: min(c))
    return [body, *redo_components]


def _split_loop(
    log: Log, groups: list[set[str]]
) -> list[Log]:
    """Split each trace into alternating body / redo segments
    and accumulate per-group sub-logs.

    Each trace starts in the body. When the activity exits the
    body, we collect the redo segment and continue. When the
    activity re-enters the body, we close the redo segment and
    start a new body segment."""
    body = groups[0]
    redo_groups = groups[1:]
    sub_logs: list[Log] = [[] for _ in groups]

    for trace in log:
        current_body: list[str] = []
        current_redo: list[str] | None = None
        current_redo_idx: int = -1
        for activity in trace:
            if activity in body:
                # Entering or staying in the body.
                if current_redo is not None:
                    # Finish the redo segment.
                    sub_logs[current_redo_idx + 1].append(
                        tuple(current_redo)
                    )
                    current_redo = None
                    current_redo_idx = -1
                current_body.append(activity)
            else:
                # In a redo. Figure out which one.
                found_idx = -1
                for i, redo in enumerate(redo_groups):
                    if activity in redo:
                        found_idx = i
                        break
                if found_idx == -1:
                    # Shouldn't happen if the cut was valid;
                    # silently drop.
                    continue
                if current_redo is None or current_redo_idx != found_idx:
                    # New redo segment — close the body segment
                    # first if we have one.
                    if current_body:
                        sub_logs[0].append(tuple(current_body))
                        current_body = []
                    if current_redo is not None:
                        sub_logs[current_redo_idx + 1].append(
                            tuple(current_redo)
                        )
                    current_redo = []
                    current_redo_idx = found_idx
                current_redo.append(activity)

        # End of trace — flush whatever's open.
        if current_redo is not None:
            sub_logs[current_redo_idx + 1].append(tuple(current_redo))
        elif current_body:
            sub_logs[0].append(tuple(current_body))

    return sub_logs


# ---------------------------------------------------------------------------
# The recursive miner.
# ---------------------------------------------------------------------------


def _mine(log: Log) -> ProcessTree:
    """Discover a process tree for ``log`` by recursive cuts.

    Pruning: if the log is empty (or only contains empty
    traces) we return ``Tau``. If there's exactly one activity
    in the log we return an ``Activity`` leaf (with optional
    Tau-loop wrapping if the activity repeats)."""
    activities = _activities(log)

    # Base case 1: empty log or only empty traces.
    if not activities:
        return Tau()

    # Base case 2: single activity.
    if len(activities) == 1:
        only = next(iter(activities))
        # If the activity appears multiple times in some trace,
        # it's a self-loop — wrap in a Loop with τ-redo so the
        # net can replay the activity. Otherwise it's a single
        # firing.
        max_repeats = max(t.count(only) for t in log) if log else 1
        if max_repeats > 1:
            return Loop(children=(Activity(name=only), Tau()))
        return Activity(name=only)

    edges = _build_dfg(log)

    # Try cuts in priority order.
    groups = _exclusive_choice_cut(activities, edges)
    if groups is not None:
        sub_logs = _split_exclusive(log, groups)
        return ExclusiveChoice(
            children=tuple(_mine(sub_log) for sub_log in sub_logs)
        )

    groups = _sequence_cut(activities, edges)
    if groups is not None:
        sub_logs = _split_sequence(log, groups)
        return Sequence(
            children=tuple(_mine(sub_log) for sub_log in sub_logs)
        )

    groups = _parallel_cut(activities, edges, log)
    if groups is not None:
        sub_logs = _split_parallel(log, groups)
        return Parallel(
            children=tuple(_mine(sub_log) for sub_log in sub_logs)
        )

    groups = _loop_cut(activities, edges, log)
    if groups is not None:
        sub_logs = _split_loop(log, groups)
        return Loop(
            children=tuple(_mine(sub_log) for sub_log in sub_logs)
        )

    # Fallback: flower model — an XOR of all activities in a
    # self-loop. The log is too unstructured for any cut to
    # apply.
    return Loop(
        children=(
            ExclusiveChoice(
                children=tuple(
                    Activity(name=a) for a in sorted(activities)
                )
            ),
            Tau(),
        )
    )


# ---------------------------------------------------------------------------
# Process tree → Petri net translator.
#
# The classical block-structured construction (Aalst 1998).
# Each subtree returns a net with a single source and single
# sink place; combining via fresh transitions stitches them
# together.
# ---------------------------------------------------------------------------


class _Translator:
    """Stateful counter so we can mint fresh ids for places and
    transitions during the recursive walk without thinking
    about collisions."""

    def __init__(self) -> None:
        self.net = PetriNet()
        self._next_place = 0
        self._next_transition = 0

    def fresh_place(self) -> str:
        pid = f"p_{self._next_place}"
        self._next_place += 1
        self.net.add_place(pid)
        return pid

    def fresh_transition(self, label: str | None = None, *, silent: bool = False) -> str:
        tid = f"t_{self._next_transition}"
        self._next_transition += 1
        self.net.add_transition(tid, label=label, silent=silent)
        return tid

    def translate(self, tree: ProcessTree) -> tuple[str, str]:
        """Return the (source_place, sink_place) of the net
        fragment representing ``tree``. The fragment is added
        in-place to ``self.net``."""
        if isinstance(tree, Tau):
            src = self.fresh_place()
            sink = self.fresh_place()
            tid = self.fresh_transition(label="τ", silent=True)
            self.net.add_arc(src, tid)
            self.net.add_arc(tid, sink)
            return src, sink

        if isinstance(tree, Activity):
            src = self.fresh_place()
            sink = self.fresh_place()
            tid = self.fresh_transition(label=tree.name)
            self.net.add_arc(src, tid)
            self.net.add_arc(tid, sink)
            return src, sink

        if isinstance(tree, Sequence):
            # Chain: source of first ←→ sink of last.
            src, sink = self.translate(tree.children[0])
            for child in tree.children[1:]:
                child_src, child_sink = self.translate(child)
                # Connect previous-sink to child-source via a
                # silent transition.
                tid = self.fresh_transition(label="τ", silent=True)
                self.net.add_arc(sink, tid)
                self.net.add_arc(tid, child_src)
                sink = child_sink
            return src, sink

        if isinstance(tree, ExclusiveChoice):
            src = self.fresh_place()
            sink = self.fresh_place()
            for child in tree.children:
                child_src, child_sink = self.translate(child)
                # Entry silent transition: src → child_src.
                t_in = self.fresh_transition(label="τ", silent=True)
                self.net.add_arc(src, t_in)
                self.net.add_arc(t_in, child_src)
                # Exit silent transition: child_sink → sink.
                t_out = self.fresh_transition(label="τ", silent=True)
                self.net.add_arc(child_sink, t_out)
                self.net.add_arc(t_out, sink)
            return src, sink

        if isinstance(tree, Parallel):
            src = self.fresh_place()
            sink = self.fresh_place()
            # AND-split: one transition with N output places.
            t_split = self.fresh_transition(label="τ", silent=True)
            self.net.add_arc(src, t_split)
            child_sinks: list[str] = []
            for child in tree.children:
                child_src, child_sink = self.translate(child)
                self.net.add_arc(t_split, child_src)
                child_sinks.append(child_sink)
            # AND-join: one transition with N input places.
            t_join = self.fresh_transition(label="τ", silent=True)
            for child_sink in child_sinks:
                self.net.add_arc(child_sink, t_join)
            self.net.add_arc(t_join, sink)
            return src, sink

        if isinstance(tree, Loop):
            # Loop(body, redo_0, redo_1, ...). One iteration:
            # execute body. From body's sink: either exit to
            # the loop's sink, OR execute one redo and re-enter
            # body. The construction uses two "hub" places:
            # entry (= body's source) and exit-or-redo
            # (= body's sink, also the redo's source via
            # entry transitions).
            body, *redos = tree.children
            src = self.fresh_place()  # loop entry
            sink = self.fresh_place()  # loop exit
            body_src, body_sink = self.translate(body)
            # Entry into the body: src → body_src.
            t_enter = self.fresh_transition(label="τ", silent=True)
            self.net.add_arc(src, t_enter)
            self.net.add_arc(t_enter, body_src)
            # Exit out of the body: body_sink → sink.
            t_exit = self.fresh_transition(label="τ", silent=True)
            self.net.add_arc(body_sink, t_exit)
            self.net.add_arc(t_exit, sink)
            # Each redo: body_sink → redo_src; redo_sink →
            # body_src (loop back).
            for redo in redos:
                redo_src, redo_sink = self.translate(redo)
                t_back_in = self.fresh_transition(label="τ", silent=True)
                self.net.add_arc(body_sink, t_back_in)
                self.net.add_arc(t_back_in, redo_src)
                t_back_out = self.fresh_transition(label="τ", silent=True)
                self.net.add_arc(redo_sink, t_back_out)
                self.net.add_arc(t_back_out, body_src)
            return src, sink

        raise TypeError(
            f"unknown process-tree node type: {type(tree).__name__}"
        )


def _process_tree_to_petri_net(tree: ProcessTree) -> PetriNet:
    """Translate a process tree to a Petri net with one
    initial token at the source place. The net's initial
    marking is set so the net is ready to fire from M_0."""
    translator = _Translator()
    src, _sink = translator.translate(tree)
    # Initial marking: one token at the overall source.
    translator.net.initial_marking[src] = 1
    return translator.net


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def discover_inductive(
    traces: "list[XESTrace] | list[Trace]",
) -> PetriNet:
    """Discover a Petri net from a trace set via the basic
    Inductive Miner.

    Parameters
    ----------
    traces
        Either a list of :class:`XESTrace` objects (the
        natural shape after :func:`parse_xes` / ``parse_csv``
        / ``parse_json``) or a raw list of activity-name
        tuples. XESTrace inputs are projected to their event
        names; everything else about the traces (attributes,
        timestamps) is deliberately ignored — discovery is
        purely structural.

    Returns
    -------
    PetriNet
        Sound by construction (every Petri net the basic IM
        emits is option-to-complete, properly-terminating, and
        free of dead transitions; verify with
        :func:`check_soundness` if you want explicit assurance).
        The initial marking places one token at the source.
        Transition labels are the activity names from the log;
        silent (τ) transitions are flagged via
        :attr:`PetriNet.silent_transitions` so weak bisimulation
        treats them correctly.

    Notes
    -----
    Activity names that are pure Python identifiers map cleanly
    onto place / transition IDs. Names with spaces or special
    characters are stored verbatim on the transition labels
    but never appear in IDs (the translator mints synthetic
    ``t_N`` / ``p_N`` IDs).
    """
    log: Log = []
    for trace in traces:
        if isinstance(trace, XESTrace):
            log.append(tuple(event.name for event in trace.events))
        else:
            log.append(tuple(trace))
    tree = _mine(log)
    return _process_tree_to_petri_net(tree)


def discover_and_train(
    traces: list[XESTrace],
    *,
    attribute_to_marking,
    steps: int = 500,
    lr: float = 0.1,
    sharpness: float = 1.0,
    seed: int | None = None,
) -> "tuple[PetriNet, object, list[float]]":
    """End-to-end discover → verify → compile → train pipeline.

    1. Mine a Petri net from ``traces`` via
       :func:`discover_inductive`.
    2. Verify the mined net is sound via
       :func:`petri_net_nn.soundness.check_soundness`. By
       construction the basic IM always yields sound nets;
       this check guards against future changes to the
       miner or to the soundness definition.
    3. Compile the net into a :class:`PetriNetModule`.
    4. Train on the same ``traces`` via
       :func:`petri_net_nn.traces.train_on_traces`.

    Returns ``(net, module, losses)`` — the discovered net,
    the trained module, and the per-step loss trajectory.
    The caller can then extract rules, score anomalies, run
    bisimulation against another variant, etc., using the
    standard PETRA pipeline.
    """
    # Local imports keep the module's load-time dependency
    # footprint small — discovery doesn't itself need torch.
    import torch

    from petri_net_nn.compiler import PetriNetModule
    from petri_net_nn.soundness import check_soundness
    from petri_net_nn.traces import train_on_traces

    net = discover_inductive(traces)
    report = check_soundness(net)
    if not report.is_sound:
        # Should be impossible for the basic IM, but if it
        # ever happens it's a bug we want surfaced immediately
        # rather than masked.
        raise RuntimeError(
            f"discover_and_train: the mined net failed soundness — "
            f"{report.summary()}. This is a bug in the miner; "
            f"please report it with the input trace set."
        )
    if seed is not None:
        torch.manual_seed(seed)
    module = PetriNetModule(net, sharpness=sharpness)
    losses = train_on_traces(
        module,
        traces,
        attribute_to_marking=attribute_to_marking,
        steps=steps,
        lr=lr,
    )
    return net, module, losses
