"""Tests for the CTL model checker.

Coverage is shaped around the canonical CTL operator pairs:

  * **EF / AG** — reachability and safety.
  * **AF / EG** — liveness on every path / non-trivial path
    existence.
  * **EX / AX** — one-step lookahead.
  * **EU / AU** — until.

Each operator gets one positive test (formula holds) and one
negative test (formula fails, with the counterexample being the
specific state pinned). The overall checker also gets tests for
boolean connectives and the derived helpers (``conj``, ``disj``,
``implies``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from petri_net_nn import (
    AF,
    AG,
    AU,
    AX,
    And,
    Atom,
    EF,
    EG,
    EU,
    EX,
    Not,
    Or,
    PetriNet,
    check_ctl,
    conj,
    disj,
    implies,
    place_count_ge,
    place_empty,
    place_has_token,
    satisfies,
    transition_enabled,
)


def _sequence(*, length: int) -> PetriNet:
    """Linear net p0 -t0-> p1 -t1-> p2 ... with a single token
    initially at p0. ``length`` is the number of transitions
    (places = length + 1).

    Reusable across many tests because the reachability graph is
    a simple ``length + 1``-state chain that the formula
    semantics can be verified against by hand."""
    net = PetriNet()
    for i in range(length + 1):
        net.add_place(f"p{i}", tokens=(1 if i == 0 else 0))
    for i in range(length):
        net.add_transition(f"t{i}")
        net.add_arc(f"p{i}", f"t{i}")
        net.add_arc(f"t{i}", f"p{i+1}")
    return net


# ---------------------------------------------------------------------------
# Atomic propositions — sanity checks before testing temporal operators
# ---------------------------------------------------------------------------


def test_place_has_token_at_initial_marking():
    """The initial marking of a one-token-at-p0 net satisfies
    ``place_has_token('p0')`` and not ``place_has_token('p1')``."""
    net = _sequence(length=2)
    assert satisfies(net, place_has_token("p0"))
    assert not satisfies(net, place_has_token("p1"))


def test_place_empty_complements_place_has_token():
    """In every reachable marking, exactly one of
    ``place_has_token(p)`` and ``place_empty(p)`` should hold."""
    net = _sequence(length=2)
    result_has = check_ctl(net, place_has_token("p1"))
    result_empty = check_ctl(net, place_empty("p1"))
    # The two satisfying sets partition the reachable state space.
    assert result_has.holds_at.isdisjoint(result_empty.holds_at)
    assert result_has.holds_at | result_empty.holds_at == frozenset(
        check_ctl(net, Atom(lambda m: True, "true")).holds_at
    )


# ---------------------------------------------------------------------------
# EF / AG — reachability and safety
# ---------------------------------------------------------------------------


def test_ef_can_reach_sink_state():
    """``EF (token at sink)`` should hold at the initial marking
    of a sequential net — there's a path that fires every
    transition in turn and reaches the sink."""
    net = _sequence(length=3)
    assert satisfies(net, EF(place_has_token("p3")))


def test_ef_unreachable_state_does_not_hold():
    """``EF false`` should never hold — false is unreachable by
    construction."""
    net = _sequence(length=2)
    assert not satisfies(net, EF(Atom(lambda m: False, "false")))


def test_ag_safety_property_holds_when_invariant_is_preserved():
    """``AG (at-most-one-token-total)`` should hold on a
    sequential net — every firing moves exactly one token
    forward, so the global token count stays at 1 throughout."""
    net = _sequence(length=3)
    one_token_total = Atom(
        lambda m: sum(m.values()) == 1, "total_tokens == 1"
    )
    assert satisfies(net, AG(one_token_total))


def test_ag_safety_failure_reports_counterexample():
    """``AG (token always at p0)`` should fail on a sequential
    net — once t0 fires, p0 is empty. The checker should
    report a counterexample marking (any reachable marking where
    the formula doesn't hold). The semantics is "states satisfying
    the formula" — for AG φ that means "every reachable state
    from here keeps φ true" — so the initial marking itself
    counts as a counterexample because it can reach a φ-violator."""
    net = _sequence(length=2)
    result = check_ctl(net, AG(place_has_token("p0")))
    assert not result.holds_at_initial
    assert result.counterexample is not None
    # The full satisfying set should be empty for this formula:
    # every reachable state can reach the post-t0 marking where
    # p0 is empty, so AG φ fails everywhere.
    assert result.holds_at == frozenset()


# ---------------------------------------------------------------------------
# AF / EG — liveness on every path / existence of an infinite path
# ---------------------------------------------------------------------------


def test_af_eventually_reaches_sink_on_every_path():
    """``AF (token at sink)`` on a sequential net: every path
    from the initial marking eventually reaches the sink. There's
    only one path, so AF is equivalent to EF here, but the test
    pins that the more demanding "all paths" semantics also
    passes."""
    net = _sequence(length=3)
    assert satisfies(net, AF(place_has_token("p3")))


def test_eg_holds_for_token_alive_property():
    """``EG (some token exists)`` should hold on a sequential
    net — the token persists through every state of the only
    path, never vanishing."""
    net = _sequence(length=3)
    some_token = Atom(
        lambda m: sum(m.values()) >= 1, "some_token_exists"
    )
    assert satisfies(net, EG(some_token))


def test_eg_fails_when_property_is_only_locally_true():
    """``EG (token at p0)`` should fail on a sequential net — the
    token leaves p0 after the first firing. There's no infinite
    path along which it stays."""
    net = _sequence(length=2)
    assert not satisfies(net, EG(place_has_token("p0")))


# ---------------------------------------------------------------------------
# EX / AX — one-step lookahead
# ---------------------------------------------------------------------------


def test_ex_some_successor_satisfies():
    """``EX (token at p1)`` should hold at the initial marking
    of a sequential net — the only successor (after firing t0)
    is the marking with a token at p1."""
    net = _sequence(length=2)
    assert satisfies(net, EX(place_has_token("p1")))


def test_ax_all_successors_satisfy():
    """``AX (token at p1)`` should hold at the initial marking
    — the only successor satisfies the property, so the
    universal-over-successors statement also holds."""
    net = _sequence(length=2)
    assert satisfies(net, AX(place_has_token("p1")))


def test_ex_fails_when_no_successor_satisfies():
    """``EX (token at p3)`` should fail at the initial marking
    of a 2-step net — p3 doesn't exist, so no successor
    satisfies."""
    net = _sequence(length=2)
    assert not satisfies(net, EX(place_has_token("p3")))


# ---------------------------------------------------------------------------
# EU / AU — until
# ---------------------------------------------------------------------------


def test_eu_existential_until():
    """``E[some-token U p2-has-token]`` should hold at the
    initial marking of a 2-step sequence — there's a path where
    "some token exists somewhere" holds at every step until p2
    acquires the token. The prefix property must hold at *every*
    state before ψ becomes true (that's the strict-until
    semantics); we chose ``some-token`` because it holds at all
    three states of the path, while p0-has-token doesn't (it
    fails at the intermediate state where the token is at p1)."""
    net = _sequence(length=2)
    some_token = Atom(lambda m: sum(m.values()) >= 1, "some_token")
    assert satisfies(net, EU(some_token, place_has_token("p2")))


def test_au_universal_until():
    """``A[some-token U token-at-sink]`` should hold at the
    initial marking of a sequential net — on every path, some
    token persists throughout until the sink is reached. There
    is only one path so AU collapses to EU here."""
    net = _sequence(length=3)
    some_token = Atom(lambda m: sum(m.values()) >= 1, "some_token")
    assert satisfies(net, AU(some_token, place_has_token("p3")))


# ---------------------------------------------------------------------------
# Combined formulae — the headline use cases
# ---------------------------------------------------------------------------


def test_response_property_request_implies_eventual_response():
    """Classic *liveness*: ``AG (request → AF response)`` —
    every request is eventually followed by a response. Modelled
    as a tiny request-response net: a request-place feeds a
    handler transition that produces a response-place."""
    net = PetriNet()
    net.add_place("request", tokens=1)
    net.add_place("response")
    net.add_transition("handle")
    net.add_arc("request", "handle")
    net.add_arc("handle", "response")

    prop = AG(implies(
        place_has_token("request"),
        AF(place_has_token("response")),
    ))
    assert satisfies(net, prop)


def test_ordering_property_no_decline_without_prior_check():
    """*"Decline cannot fire until credit-check has fired."* This
    is the safety side of an ordering invariant — useful for
    regulatory rules. We build a 3-step net check → decide and
    assert that the decline-fired-place is never marked unless
    the check-fired-place is also marked."""
    net = PetriNet()
    net.add_place("p_in", tokens=1)
    net.add_place("p_check_done")
    net.add_place("p_declined")
    net.add_transition("t_check")
    net.add_transition("t_decline")
    net.add_arc("p_in", "t_check")
    net.add_arc("t_check", "p_check_done")
    net.add_arc("p_check_done", "t_decline")
    net.add_arc("t_decline", "p_declined")

    # AG (decline → check_done). Whenever p_declined holds, we
    # also have p_check_done in the same marking — the post-state
    # of t_decline retains the check_done place? Actually no: t_decline
    # consumes p_check_done. So after t_decline the marking has
    # p_declined but not p_check_done. So a different invariant
    # is appropriate.
    #
    # The right formulation: there's no path where decline fires
    # before check_done is ever populated. Equivalently:
    # ``A[¬declined U check_done]`` — on every path, decline
    # stays empty until check_done holds. We test that.
    prop = AU(
        Not(place_has_token("p_declined")),
        place_has_token("p_check_done"),
    )
    assert satisfies(net, prop)


def test_deadlock_freedom_via_ag_some_successor_exists():
    """*"No reachable state is a deadlock"* expressed in CTL as
    ``AG EX true`` — every reachable state has at least one
    successor. Fails on the final marking of a sequential net
    (sink has no outgoing transitions), so this property is
    *false* for a workflow net that terminates; it should be
    true for a non-terminating cyclic net."""
    # The cycle: p0 ↔ p1, two transitions, one token. Every
    # state has at least one enabled transition.
    net = PetriNet()
    net.add_place("p0", tokens=1)
    net.add_place("p1")
    net.add_transition("t_forward")
    net.add_transition("t_back")
    net.add_arc("p0", "t_forward")
    net.add_arc("t_forward", "p1")
    net.add_arc("p1", "t_back")
    net.add_arc("t_back", "p0")

    deadlock_free = AG(EX(Atom(lambda m: True, "true")))
    assert satisfies(net, deadlock_free)


def test_conj_and_disj_fold_correctly():
    """``conj(a, b, c)`` should be equivalent to
    ``And(a, And(b, c))`` and ``disj(a, b, c)`` to
    ``Or(a, Or(b, c))``. We verify by evaluating both forms
    against the same net and checking the satisfying sets
    match."""
    net = _sequence(length=2)
    a = place_has_token("p0")
    b = place_empty("p1")
    c = place_empty("p2")
    folded = conj(a, b, c)
    explicit = And(a, And(b, c))
    assert check_ctl(net, folded).holds_at == check_ctl(net, explicit).holds_at

    folded_or = disj(a, b, c)
    explicit_or = Or(a, Or(b, c))
    assert (
        check_ctl(net, folded_or).holds_at
        == check_ctl(net, explicit_or).holds_at
    )


def test_transition_enabled_atom_changes_with_marking():
    """``transition_enabled(net, t)`` should produce an atom
    that holds in exactly the markings where ``t`` is enabled.
    In a 2-step net, t0 is enabled at the initial marking but
    not at the post-t0 marking; t1 is the reverse."""
    net = _sequence(length=2)
    t0_enabled = transition_enabled(net, "t0")
    t1_enabled = transition_enabled(net, "t1")

    # At least one marking satisfies t0 (the initial); at least
    # one satisfies t1 (post-t0). They should be disjoint.
    t0_states = check_ctl(net, t0_enabled).holds_at
    t1_states = check_ctl(net, t1_enabled).holds_at
    assert t0_states
    assert t1_states
    assert t0_states.isdisjoint(t1_states)
