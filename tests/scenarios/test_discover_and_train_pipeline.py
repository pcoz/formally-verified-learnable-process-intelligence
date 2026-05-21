"""End-to-end test for the native log-to-net discovery scenario.

Demonstrates that PETRA can ingest a log alone — no structural
model supplied — discover a sound Petri net via the basic
Inductive Miner, and train weights inside the mined structure.

Coverage:

* Scenario loads through ``load_scenario`` with
  ``net.source = "discover"``, producing a non-empty Petri net
  whose initial marking sits at the canonical ``p_0`` source
  place.
* The mined net is sound by construction (option to complete,
  proper completion, no dead transitions).
* Every input trace replays on the mined net up to ``τ`` collapse
  (the load-bearing invariant the Inductive Miner promises).
* Training through ``ScenarioContext.compile`` + ``train_on_traces``
  reduces the loss measurably from initial to final.
* The convenience ``discover_and_train`` one-call API produces
  the same shape of result for users who only have a log.
"""
from __future__ import annotations

from pathlib import Path

import torch

from petri_net_nn import (
    XESTrace,
    check_soundness,
    discover_and_train,
    load_scenario,
    train_on_traces,
)


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "discover_and_train_pipeline"
    / "scenario.toml"
)


# The Inductive Miner's translator always emits ``p_0`` as the
# canonical source place; the scenario.toml feeds a unit
# activation there so the trainer can propagate signal through
# the full mined topology.
def _source_marking(trace: XESTrace) -> dict[str, float]:
    return {"p_0": 1.0}


def _can_replay(net, trace: tuple[str, ...]) -> bool:
    """Return True iff the trace's events can be fired in order
    on ``net`` from the initial marking, with silent ``τ``
    transitions allowed to interleave freely. BFS over
    (marking, position) pairs."""
    from collections import deque

    sinks = {p for p in net.places if not net.postset(p)}
    if not sinks:
        return False
    final_marking = {p: 1 for p in sinks}

    initial = dict(net.initial_marking)
    start = (frozenset(initial.items()), 0)
    visited: set = {start}
    queue: deque = deque([(initial, 0)])

    while queue:
        marking, pos = queue.popleft()
        if pos == len(trace) and all(
            marking.get(p, 0) == n for p, n in final_marking.items()
        ):
            return True
        for t in net.transitions:
            if not net.is_enabled(t, marking):
                continue
            label = net.transition_labels.get(t, t)
            is_silent = t in net.silent_transitions
            if is_silent:
                new_marking = net.fire(t, marking)
                state = (frozenset(new_marking.items()), pos)
                if state not in visited:
                    visited.add(state)
                    queue.append((new_marking, pos))
            elif pos < len(trace) and label == trace[pos]:
                new_marking = net.fire(t, marking)
                state = (frozenset(new_marking.items()), pos + 1)
                if state not in visited:
                    visited.add(state)
                    queue.append((new_marking, pos + 1))
    return False


def test_scenario_loads_and_discovers_a_net():
    """``load_scenario`` with ``net.source = "discover"`` returns
    a ScenarioContext whose net was mined from the inline traces."""
    ctx = load_scenario(SCENARIO)
    assert len(ctx.traces) == 4
    # The Inductive Miner produces a net with structural places
    # plus one transition per visible activity in the log; both
    # counts must be positive.
    assert len(ctx.net.places) > 0
    assert len(ctx.net.transitions) > 0
    # The mined net's initial marking is the canonical ``{p_0: 1}``.
    assert ctx.net.initial_marking == {"p_0": 1}


def test_discovered_net_carries_all_visible_activities():
    """Every activity name in the log appears as a transition
    label in the mined net (modulo silent ``τ`` transitions
    minted for structural routing)."""
    ctx = load_scenario(SCENARIO)
    visible_labels = {
        label
        for tid, label in ctx.net.transition_labels.items()
        if tid not in ctx.net.silent_transitions
    }
    expected = {
        "request",
        "verify_id",
        "credit_check",
        "review",
        "approve",
        "decline",
        "close",
    }
    assert expected <= visible_labels


def test_discovered_net_is_sound_by_construction():
    """The basic Inductive Miner promises soundness; the
    scenario test enforces it explicitly so any future
    regression in the miner trips this assertion."""
    ctx = load_scenario(SCENARIO)
    report = check_soundness(ctx.net)
    assert report.is_sound, report.summary()


def test_every_input_trace_replays_on_the_mined_net():
    """Replay invariant: every input trace must be firable in
    order on the mined net, with silent transitions allowed to
    interleave. The Inductive Miner is correct iff this holds
    for every trace it was given."""
    ctx = load_scenario(SCENARIO)
    for trace in ctx.traces:
        events = tuple(e.name for e in trace.events)
        assert _can_replay(ctx.net, events), (
            f"trace {events} did not replay on the mined net"
        )


def test_training_reduces_loss_through_the_full_adapter_path():
    """End-to-end through the adapter: load the scenario, compile
    the discovered net, train on the same traces, and confirm the
    loss falls. This is the headline shape — a user who has only
    a log can land on a trained model with one ``load_scenario``
    call plus a standard ``train_on_traces`` invocation."""
    torch.manual_seed(0)
    ctx = load_scenario(SCENARIO)
    module = ctx.compile()
    losses = train_on_traces(
        module,
        ctx.traces,
        attribute_to_marking=_source_marking,
        steps=ctx.training.steps,
        lr=ctx.training.lr,
    )
    assert losses[-1] < losses[0]


def test_one_call_discover_and_train_api_matches():
    """``discover_and_train`` bundles discovery + soundness check
    + compile + train into a single call, for users who don't
    want to thread the steps themselves. Same falling-loss
    guarantee as the adapter-driven path above."""
    torch.manual_seed(0)
    ctx = load_scenario(SCENARIO)
    net, module, losses = discover_and_train(
        ctx.traces,
        attribute_to_marking=_source_marking,
        steps=ctx.training.steps,
        lr=ctx.training.lr,
        seed=0,
    )
    # The convenience API runs its own soundness check internally;
    # the test asserts the returned net is sound to pin that the
    # check actually fires.
    assert check_soundness(net).is_sound
    assert losses[-1] < losses[0]
