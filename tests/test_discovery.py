"""Tests for the basic Inductive Miner.

The load-bearing assertions follow the risk analysis posted
mid-implementation:

  * **Each canonical cut shape** — sequence, exclusive choice,
    parallel, loop — gets a positive test that asserts both
    the produced tree's shape AND the two invariants:
      1. ``check_soundness(net).is_sound`` (the miner promises
         sound nets by construction);
      2. *replay invariant* — every input trace must be
         replayable on the output net, modulo τ collapse.
        Implemented by a BFS over (marking, trace-position)
        pairs that fires τ transitions freely and visible
        transitions only when they match the next expected
        event.

  * **Loop regression cases** — the patterns that triggered the
    initial body-computation bug ((a,b,a,b,a) and
    (a,b,c,a,b,c,a)) are pinned as regression tests so any
    future drift surfaces immediately.

  * **Compound case** — sequence containing an exclusive
    choice — to verify the recursion composes.

  * **Tree-shape sanity tests** — exercising the underlying
    process tree builder directly (without the Petri-net
    translation step) to catch bugs in the cut detectors
    independently of the translator.

The replay invariant is the single most useful test for
catching subtle mining bugs: any wrong split / wrong tree /
wrong translation makes some input trace un-replayable, which
the BFS detects.
"""
from __future__ import annotations

import pytest

from petri_net_nn import (
    Activity,
    ExclusiveChoice,
    Loop,
    Parallel,
    Sequence,
    Tau,
    check_soundness,
    discover_inductive,
)
from petri_net_nn.discovery import _mine
from petri_net_nn.petri_net import PetriNet


# ---------------------------------------------------------------------------
# Replay invariant — the core test helper.
# ---------------------------------------------------------------------------


def _can_replay(net: PetriNet, trace: tuple[str, ...]) -> bool:
    """Return True iff a Petri-net firing sequence consumes
    ``trace`` exactly and lands the marking at the canonical
    final marking (one token at each sink place).

    BFS over (marking, position) pairs:
      * silent (τ) transitions fire without consuming from the
        trace;
      * visible transitions fire only when their label matches
        the next expected event;
      * the search succeeds when the trace is fully consumed
        AND the marking matches the expected final marking.

    Capped at 50,000 explored states; well above what the test
    nets need."""
    sinks = {p for p in net.places if not net.postset(p)}
    if not sinks:
        # No sinks — can't define a final marking. Bail.
        return False
    final_marking = {p: 1 for p in sinks}

    initial = dict(net.initial_marking)
    silent = net.silent_transitions

    def marking_key(m: dict[str, int]) -> tuple:
        # Canonical, hashable form for the visited set.
        return tuple(sorted((p, c) for p, c in m.items() if c > 0))

    initial_key = marking_key(initial)
    visited: set[tuple[tuple, int]] = {(initial_key, 0)}
    queue: list[tuple[dict[str, int], int]] = [(initial, 0)]

    def matches_final(m: dict[str, int]) -> bool:
        if any(m.get(p, 0) != final_marking[p] for p in final_marking):
            return False
        for p, c in m.items():
            if p not in final_marking and c != 0:
                return False
        return True

    explored = 0
    while queue:
        explored += 1
        if explored > 50_000:
            # Safety cap — the test nets should never hit this.
            raise RuntimeError(
                "replay search exceeded 50000 states; check the "
                "test net for unbounded loops or pathological "
                "shape"
            )
        m, pos = queue.pop()
        if pos == len(trace) and matches_final(m):
            return True
        for t in net.transitions:
            if not net.is_enabled(t, m):
                continue
            new_m = net.fire(t, m)
            if t in silent:
                new_pos = pos
            else:
                label = net.transition_labels.get(t, t)
                if pos >= len(trace) or label != trace[pos]:
                    continue
                new_pos = pos + 1
            key = (marking_key(new_m), new_pos)
            if key not in visited:
                visited.add(key)
                queue.append((new_m, new_pos))
    return False


def _assert_invariants(net: PetriNet, log: list[tuple[str, ...]]) -> None:
    """Shared post-conditions every miner test asserts:
    soundness + replay invariant for every input trace."""
    report = check_soundness(net)
    assert report.is_sound, (
        f"miner produced an unsound net: {report.summary()}"
    )
    for trace in log:
        assert _can_replay(net, trace), (
            f"trace {trace!r} cannot be replayed on the mined net"
        )


# ---------------------------------------------------------------------------
# Tree-shape tests — exercise the cut detectors via _mine without
# the Petri-net translator in between.
# ---------------------------------------------------------------------------


def test_mine_sequence():
    """[(a, b, c), (a, b, c)] → Sequence(a, b, c)."""
    log = [("a", "b", "c"), ("a", "b", "c")]
    tree = _mine(log)
    assert isinstance(tree, Sequence)
    names = [child.name for child in tree.children if isinstance(child, Activity)]
    assert names == ["a", "b", "c"]


def test_mine_exclusive_choice():
    """[(b,), (c,), (d,)] should produce ExclusiveChoice(b, c, d)
    — three traces, each a single distinct activity, no shared
    structure."""
    log = [("b",), ("c",), ("d",)]
    tree = _mine(log)
    assert isinstance(tree, ExclusiveChoice)
    branch_names = sorted(
        c.name for c in tree.children if isinstance(c, Activity)
    )
    assert branch_names == ["b", "c", "d"]


def test_mine_parallel():
    """[(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)] —
    all six permutations of (a,b,c), full interleaving — must
    produce Parallel(a, b, c)."""
    log = [
        ("a", "b", "c"), ("a", "c", "b"),
        ("b", "a", "c"), ("b", "c", "a"),
        ("c", "a", "b"), ("c", "b", "a"),
    ]
    tree = _mine(log)
    assert isinstance(tree, Parallel)
    names = sorted(c.name for c in tree.children if isinstance(c, Activity))
    assert names == ["a", "b", "c"]


def test_mine_loop_simple():
    """[(a,), (a,b,a), (a,b,a,b,a)] — body=a, redo=b. The
    minimal-body fix (start∪end = {a}) is what makes this
    test pass; the original forward∩backward criterion picked
    body={a,b} and fell through to the flower."""
    log = [("a",), ("a", "b", "a"), ("a", "b", "a", "b", "a")]
    tree = _mine(log)
    assert isinstance(tree, Loop), (
        f"expected Loop(...), got {type(tree).__name__}"
    )
    # The body should be Activity('a'); the redo should reduce
    # to Activity('b') (with possibly a τ wrapper if the
    # single-activity sub-log triggered the self-loop base case;
    # not the case here since b appears only once per redo
    # segment).
    body = tree.children[0]
    assert isinstance(body, Activity) and body.name == "a"


def test_mine_loop_with_sequence_redo():
    """[(a,b,c,a,b,c,a)] — body=a, redo activities={b,c} with
    a sequence b→c structure. Same regression case as the
    simple loop; the redo sub-tree should resolve to a
    Sequence."""
    log = [("a", "b", "c", "a", "b", "c", "a")]
    tree = _mine(log)
    assert isinstance(tree, Loop)
    body = tree.children[0]
    assert isinstance(body, Activity) and body.name == "a"


def test_mine_sequence_with_exclusive():
    """[(a,b,d), (a,c,d)] — a then (b or c) then d.
    Sequence(a, XOR(b, c), d). Verifies that recursion composes
    sequence and exclusive cuts at different levels."""
    log = [("a", "b", "d"), ("a", "c", "d")]
    tree = _mine(log)
    assert isinstance(tree, Sequence)
    assert len(tree.children) == 3
    first, middle, last = tree.children
    assert isinstance(first, Activity) and first.name == "a"
    assert isinstance(last, Activity) and last.name == "d"
    assert isinstance(middle, ExclusiveChoice)
    branch_names = sorted(
        c.name for c in middle.children if isinstance(c, Activity)
    )
    assert branch_names == ["b", "c"]


def test_mine_single_activity_no_repetition():
    """[(a,)] — base case: single activity, no repetition."""
    tree = _mine([("a",)])
    assert isinstance(tree, Activity) and tree.name == "a"


def test_mine_single_activity_with_repetition():
    """[(a, a, a)] — single activity repeated. Wraps in
    Loop(Activity('a'), Tau)."""
    tree = _mine([("a", "a", "a")])
    assert isinstance(tree, Loop)
    body = tree.children[0]
    assert isinstance(body, Activity) and body.name == "a"
    # The redo should be a Tau.
    assert any(isinstance(c, Tau) for c in tree.children[1:])


def test_mine_empty_log():
    """Empty log → Tau."""
    assert isinstance(_mine([]), Tau)


def test_mine_only_empty_traces():
    """[(), ()] → Tau — there are no activities at all."""
    assert isinstance(_mine([(), ()]), Tau)


# ---------------------------------------------------------------------------
# discover_inductive end-to-end — soundness + replay across all four cut
# shapes plus the compound case and the regression cases.
# ---------------------------------------------------------------------------


def _to_log(traces: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Identity — alias for clarity at the call site."""
    return traces


def test_discover_sequence_replays_and_is_sound():
    log = _to_log([("a", "b", "c"), ("a", "b", "c")])
    net = discover_inductive(log)
    _assert_invariants(net, log)


def test_discover_exclusive_replays_and_is_sound():
    log = _to_log([("a", "b"), ("a", "c"), ("a", "d")])
    net = discover_inductive(log)
    _assert_invariants(net, log)


def test_discover_parallel_replays_and_is_sound():
    log = _to_log([
        ("a", "b", "c"), ("a", "c", "b"),
        ("b", "a", "c"), ("b", "c", "a"),
        ("c", "a", "b"), ("c", "b", "a"),
    ])
    net = discover_inductive(log)
    _assert_invariants(net, log)


def test_discover_loop_replays_and_is_sound():
    """Regression: this exact log triggered the original
    forward∩backward body-computation bug."""
    log = _to_log([
        ("a",),
        ("a", "b", "a"),
        ("a", "b", "a", "b", "a"),
    ])
    net = discover_inductive(log)
    _assert_invariants(net, log)


def test_discover_loop_with_seq_redo_replays_and_is_sound():
    """Regression: another body-computation case where the
    minimal body (start∪end={a}) is essential."""
    log = _to_log([("a", "b", "c", "a", "b", "c", "a")])
    net = discover_inductive(log)
    _assert_invariants(net, log)


def test_discover_compound_sequence_with_xor_replays_and_is_sound():
    log = _to_log([("a", "b", "d"), ("a", "c", "d")])
    net = discover_inductive(log)
    _assert_invariants(net, log)


def test_discover_single_event_traces_replay_and_are_sound():
    """Edge case: every trace is one event long."""
    log = _to_log([("a",), ("b",), ("c",)])
    net = discover_inductive(log)
    _assert_invariants(net, log)


def test_discover_handles_xes_traces_natively():
    """The public API accepts XESTrace lists too — the
    discover step projects each to its event-name tuple."""
    from petri_net_nn import XESEvent, XESTrace

    xes_traces = [
        XESTrace(
            attributes={},
            events=[
                XESEvent(name="a"),
                XESEvent(name="b"),
                XESEvent(name="c"),
            ],
        ),
        XESTrace(
            attributes={},
            events=[
                XESEvent(name="a"),
                XESEvent(name="b"),
                XESEvent(name="c"),
            ],
        ),
    ]
    net = discover_inductive(xes_traces)
    # Soundness is what matters here; the tree shape is the
    # same as the tuple-input case above.
    assert check_soundness(net).is_sound


# ---------------------------------------------------------------------------
# discover_and_train end-to-end — discovery + training pipeline.
# ---------------------------------------------------------------------------


def test_discover_and_train_runs_end_to_end():
    """The convenience pipeline: discover → check_soundness →
    compile → train. Smoke test only — the underlying pieces
    are covered by their own tests."""
    from petri_net_nn import XESEvent, XESTrace, discover_and_train

    traces = [
        XESTrace(
            attributes={"x": "1.0"},
            events=[XESEvent(name="a"), XESEvent(name="b")],
        )
        for _ in range(10)
    ]

    def to_marking(trace):
        return {}

    net, module, losses = discover_and_train(
        traces,
        attribute_to_marking=to_marking,
        steps=20,
        lr=0.1,
        seed=0,
    )
    # Discovered net is sound by construction; the convenience
    # function also verifies this internally, so reaching here
    # means it passed.
    assert check_soundness(net).is_sound
    # Training produced a non-empty loss trajectory.
    assert len(losses) == 20
