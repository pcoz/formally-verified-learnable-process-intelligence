"""Tests for order-preserving discovery and order-aware replay training.

The load-bearing test is the *ordered dependency* log: the outcome
(EXIT_A vs EXIT_B) is determined by the ORDER of two middle
activities, never by which activities occurred. The occurrence-vector
path (Inductive Miner + ``train_on_traces``) sees identical activity
sets for both variants and therefore cannot do better than 50/50; the
new path (``discover_transition_system`` + ``train_by_replay``) must
recover the dependency.
"""
from __future__ import annotations

import math

import pytest
import torch

from petri_net_nn import (
    PetriNetModule,
    XESEvent,
    XESTrace,
    check_soundness,
    discover_inductive,
    discover_transition_system,
    enabled_choices,
    extract_next_step_rules,
    extract_xor_rule,
    find_xor_groups,
    next_step_distribution,
    replay_anomaly_score,
    replay_log_likelihood,
    replay_trace,
    train_by_replay,
    train_on_traces,
)


def _trace(events, **attrs):
    return XESTrace(attributes=dict(attrs), events=[XESEvent(e) for e in events])


def _ordered_log(n=40):
    # Order decides the outcome: X then Y -> EXIT_A; Y then X -> EXIT_B.
    return ([_trace(["START", "X", "Y", "EXIT_A"])] * n
            + [_trace(["START", "Y", "X", "EXIT_B"])] * n)


def _consecutive_log(n=30):
    # One FALL then EXIT is as common as FALL FALL ...; but after TWO
    # consecutive FALLs the story always exits.
    return ([_trace(["ENTER", "FALL", "EXIT"])] * n
            + [_trace(["ENTER", "FALL", "UP", "FALL", "UP", "EXIT"])] * n
            + [_trace(["ENTER", "UP", "FALL", "FALL", "EXIT"])] * n)


# ---------------------------------------------------------------------------
# discover_transition_system
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_transition_system_is_sound_and_replays_every_trace(horizon):
    log = _ordered_log(3) + _consecutive_log(3)
    net = discover_transition_system(log, horizon=horizon)
    assert check_soundness(net).is_sound
    for i, t in enumerate(log):
        steps, fits = replay_trace(net, [e.name for e in t.events], trace_index=i)
        assert fits and all(s.fired is not None for s in steps)


def test_transition_system_states_are_k_tails():
    net = discover_transition_system(_consecutive_log(1), horizon=2)
    labels = set(net.place_labels.values())
    assert {"start", "end", "ENTER", "ENTER > FALL", "FALL > FALL", "UP > FALL"} <= labels
    # one visible transition per (state, activity) pair; every transition
    # is a state-machine step (one input, one output)
    assert all(len(net.preset(t)) == 1 and len(net.postset(t)) == 1 for t in net.transitions)


def test_transition_system_rejects_bad_input():
    with pytest.raises(ValueError):
        discover_transition_system([], horizon=1)
    with pytest.raises(ValueError):
        discover_transition_system([("a",)], horizon=0)


def test_transition_system_accepts_raw_tuples():
    net = discover_transition_system([("a", "b"), ("a", "c")], horizon=1)
    assert check_soundness(net).is_sound
    assert sorted(net.transition_labels[t] for t in net.transitions
                  if t not in net.silent_transitions) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# replay mechanics on an Inductive-Miner net (silent transitions)
# ---------------------------------------------------------------------------


def test_enabled_choices_sees_through_silent_transitions():
    log = [("a", "b", "d"), ("a", "c", "d")]
    net = discover_inductive(log)
    m = dict(net.initial_marking)
    first = enabled_choices(net, m)
    assert {net.transition_labels[t] for t in first} == {"a"}
    m = net.fire(next(iter(first)), first[next(iter(first))])
    assert {net.transition_labels[t] for t in enabled_choices(net, m)} == {"b", "c"}


def test_replay_reports_unfit_event():
    net = discover_transition_system([("a", "b")], horizon=1)
    steps, fits = replay_trace(net, ["a", "z"])
    assert not fits and steps[-1].fired is None and steps[-1].event == "z"


def test_train_by_replay_works_on_inductive_miner_net():
    log = [_trace(["a", "b", "d"])] * 30 + [_trace(["a", "c", "d"])] * 10
    net = discover_inductive(log)
    torch.manual_seed(0)
    module = PetriNetModule(net)
    losses = train_by_replay(module, log, steps=200, lr=0.1)
    assert losses[-1] < losses[0]
    b = next(t for t in net.transitions if net.transition_labels.get(t) == "b")
    m = net.fire(next(t for t in net.transitions if net.transition_labels.get(t) == "a"),
                 dict(net.initial_marking))
    assert next_step_distribution(module, m)[b] == pytest.approx(0.75, abs=0.05)


# ---------------------------------------------------------------------------
# the ordered dependency: old path cannot, new path does
# ---------------------------------------------------------------------------


def _label_prob(module, marking, label):
    dist = next_step_distribution(module, marking)
    return sum(p for t, p in dist.items() if module.net.transition_labels.get(t) == label)


def test_occurrence_training_cannot_learn_order_dependency():
    log = _ordered_log()
    net = discover_inductive(log)
    torch.manual_seed(0)
    module = PetriNetModule(net)
    train_on_traces(module, log, attribute_to_marking=lambda t: {}, steps=300, lr=0.1)
    with torch.no_grad():
        out = module(batch_size=1)
    pa = next(out[t].item() for t in net.transitions if net.transition_labels.get(t) == "EXIT_A")
    pb = next(out[t].item() for t in net.transitions if net.transition_labels.get(t) == "EXIT_B")
    # identical for both orders: the best it can do is the 50/50 marginal
    assert abs(pa - 0.5) < 0.1 and abs(pb - 0.5) < 0.1


def test_replay_training_recovers_order_dependency():
    log = _ordered_log()
    net = discover_transition_system(log, horizon=2)
    torch.manual_seed(0)
    module = PetriNetModule(net, num_steps=1)   # cyclic-safe; replay never unrolls
    train_by_replay(module, log, steps=300, lr=0.1)
    place = {v: k for k, v in net.place_labels.items()}
    assert _label_prob(module, {place["X > Y"]: 1}, "EXIT_A") > 0.95
    assert _label_prob(module, {place["Y > X"]: 1}, "EXIT_B") > 0.95
    ev = replay_log_likelihood(module, log)
    assert ev["accuracy"] > 0.7 and ev["unmatched_events"] == 0
    # the START->X/Y choice is a genuine coin flip; everything else is determined
    assert ev["mean_log_likelihood"] > math.log(0.5) / 3


def test_consecutive_state_rule_is_readable():
    log = _consecutive_log()
    net = discover_transition_system(log, horizon=2)
    torch.manual_seed(0)
    module = PetriNetModule(net, num_steps=1)
    train_by_replay(module, log, steps=300, lr=0.1)
    rules = {r.place_label: r for r in extract_next_step_rules(module, min_choices=1)}
    ff = rules["FALL > FALL"]
    assert ff.probabilities[0][0] == "EXIT" and ff.probabilities[0][1] > 0.95
    assert "after 'FALL > FALL': next → EXIT" in ff.description()
    # after a single FALL that followed ENTER the story is genuinely undecided
    ef = dict(rules["ENTER > FALL"].probabilities)
    assert 0.3 < ef["EXIT"] < 0.7


def test_attribute_conditioned_routing_gives_xor_crossover():
    # after START, B when the case attribute is high, C when it is low
    log = ([_trace(["START", "B"], v=x) for x in (0.6, 0.7, 0.8, 0.9, 1.0)] * 8
           + [_trace(["START", "C"], v=x) for x in (0.0, 0.1, 0.2, 0.3, 0.4)] * 8)
    net = discover_transition_system(log, horizon=1)
    torch.manual_seed(0)
    module = PetriNetModule(net, num_steps=1)
    a2m = lambda t: {p: float(t.attributes["v"]) for p in net.places}
    train_by_replay(module, log, attribute_to_marking=a2m, steps=500, lr=0.1)
    place, (ta, tb) = next((p, ts) for p, ts in find_xor_groups(net)
                           if net.place_labels[p] == "START")
    rule = extract_xor_rule(module, place, ta, tb)
    assert net.transition_labels[rule.transition_above] == "B"
    assert 0.35 < rule.crossover < 0.65


def test_replay_anomaly_pins_the_surprising_step():
    log = _consecutive_log()
    net = discover_transition_system(log, horizon=2)
    torch.manual_seed(0)
    module = PetriNetModule(net, num_steps=1)
    train_by_replay(module, log, steps=300, lr=0.1)
    odd = _trace(["ENTER", "UP", "FALL", "FALL", "UP"])   # never continues after FALL > FALL
    scores = replay_anomaly_score(module, odd)
    worst = max(scores, key=scores.get)
    assert worst == "<unfit:UP>"
    normal = replay_anomaly_score(module, log[-1])
    assert sum(normal.values()) < sum(scores.values())
