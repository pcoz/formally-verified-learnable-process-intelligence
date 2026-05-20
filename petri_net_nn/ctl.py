"""Computation Tree Logic (CTL) model checking for bounded Petri nets.

CTL extends propositional logic with temporal operators that
quantify over the **tree of possible futures** rooted at a state.
Two kinds of operators compose:

  * **Path quantifiers** — ``A`` (for *all* future paths) and
    ``E`` (there *exists* a future path).
  * **Temporal operators** — ``X`` (next state), ``F`` (some
    state in the future), ``G`` (every state in the future),
    ``U`` (until).

Combined, you get pairs like ``AG``, ``EF``, ``AF``, ``EG``, plus
the binary ``A[φ U ψ]`` and ``E[φ U ψ]``. The pairs answer
questions like:

  * ``AG (request → AF response)`` — every request is *eventually*
    followed by a response on every path. The classic *liveness*
    property.
  * ``AG ¬deadlock`` — on every path, every reachable state has
    a successor. *Safety from deadlock.*
  * ``EF goal`` — *some path* eventually reaches the goal.
    *Reachability.*
  * ``EG running`` — *some path* keeps the system running forever.
    Useful for checking non-trivial control loops.

For a bounded Petri net, CTL model checking is *decidable* —
the reachability graph is finite, and the standard fixed-point
algorithms terminate. PETRA's checker:

1. Builds the reachability graph from the net (reuses
   :func:`petri_net_nn.bisimulation.reachability_graph`).
2. Recursively evaluates the formula bottom-up, computing for
   each sub-formula the **set of states satisfying it**.
3. For temporal operators, the satisfying set is a fixed point
   — least fixed point for ``EF`` and ``EU`` (eventually-true
   semantics), greatest fixed point for ``EG`` (always-true
   semantics). The other operators are derived from these three
   primitives plus boolean connectives.

The AST is built from six **primitive** nodes — ``Atom``, ``Not``,
``And``, ``Or``, ``EX``, ``EU``, ``EG`` — plus **derived**
constructors (``AX``, ``AG``, ``AF``, ``EF``, ``AU``) that expand
to combinations of the primitives via the standard CTL
equivalences. This keeps the model-checking algorithm small (only
six primitive cases) while giving users the full CTL vocabulary.

Scope: bounded Petri nets only. The reachability graph must be
finite; ``max_states`` (default 10,000) guards against unbounded
inputs.

Atomic propositions: predicates over the current marking. The
:func:`place_has_token`, :func:`place_empty`, :func:`place_count_eq`,
:func:`place_count_ge`, :func:`transition_enabled` helpers cover the
common cases. For anything else, build an ``Atom`` directly with
a custom predicate ``(marking_dict) -> bool``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

from petri_net_nn.bisimulation import Marking, reachability_graph
from petri_net_nn.petri_net import PetriNet


# A predicate over a marking — given the marking as a dict[place,
# count], return True iff the proposition holds in that state.
# This is the most general atomic-proposition shape; the helpers
# below build common predicates from it.
MarkingPredicate = Callable[[dict[str, int]], bool]


# ---------------------------------------------------------------------------
# CTL formula AST.
#
# Frozen dataclasses with deliberately *no* derivation from a common base
# class — we use a Formula type alias to capture the union and rely on
# isinstance checks in the model checker. Mirrors the algebraic-data-
# type style typical of formal-methods libraries.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    """Atomic proposition. ``predicate`` is evaluated against every
    reachable marking; ``label`` is a human-readable name carried
    only for display and debugging."""

    predicate: MarkingPredicate
    label: str = "atom"


@dataclass(frozen=True)
class Not:
    """Logical negation."""

    arg: "Formula"


@dataclass(frozen=True)
class And:
    """Logical conjunction. Stored as a binary node; chain by
    nesting (``And(a, And(b, c))``) or use the :func:`conj`
    helper to fold over a sequence."""

    left: "Formula"
    right: "Formula"


@dataclass(frozen=True)
class Or:
    """Logical disjunction. Same convention as :class:`And`."""

    left: "Formula"
    right: "Formula"


@dataclass(frozen=True)
class EX:
    """``EX φ`` — there *exists* a successor state in which φ holds.
    The next-state existential."""

    arg: "Formula"


@dataclass(frozen=True)
class EU:
    """``E[φ U ψ]`` — there *exists* a path along which φ holds at
    every step until a step where ψ holds. The existential until."""

    left: "Formula"
    right: "Formula"


@dataclass(frozen=True)
class EG:
    """``EG φ`` — there *exists* a path along which φ holds at
    *every* state forever. The existential globally."""

    arg: "Formula"


Formula = Union[Atom, Not, And, Or, EX, EU, EG]


# ---------------------------------------------------------------------------
# Derived constructors. Each one expands to a combination of the
# six primitives using the standard CTL equivalences:
#
#   AX φ   ≡  ¬EX ¬φ            (for all successors, φ)
#   EF φ   ≡  E[true U φ]       (eventually on some path, φ)
#   AF φ   ≡  ¬EG ¬φ            (eventually on every path, φ)
#   AG φ   ≡  ¬EF ¬φ            (always on every path, φ)
#   A[φ U ψ] ≡ ¬E[¬ψ U (¬φ ∧ ¬ψ)] ∧ ¬EG ¬ψ
# ---------------------------------------------------------------------------


def _true() -> Atom:
    """Tautology — every marking satisfies it. Used internally
    to build ``EF`` and to give callers a starting point for
    boolean combinations."""
    return Atom(lambda m: True, "true")


def AX(arg: Formula) -> Formula:
    """``AX φ`` — for *all* successors, φ holds. Expands to
    ``¬EX ¬φ``: nothing exists in the next-state set that
    violates φ."""
    return Not(EX(Not(arg)))


def EF(arg: Formula) -> Formula:
    """``EF φ`` — *eventually* on some path, φ holds. Expands to
    ``E[true U φ]``: there is a path along which the trivial
    proposition true holds until φ holds (the *true* condition
    is a non-restrictive prefix)."""
    return EU(_true(), arg)


def AF(arg: Formula) -> Formula:
    """``AF φ`` — *eventually* on *every* path, φ holds. Expands
    to ``¬EG ¬φ``: there's no path that keeps φ false forever."""
    return Not(EG(Not(arg)))


def AG(arg: Formula) -> Formula:
    """``AG φ`` — *always* on every path, φ holds. Expands to
    ``¬EF ¬φ``: no reachable state violates φ. The classic
    safety operator."""
    return Not(EF(Not(arg)))


def AU(left: Formula, right: Formula) -> Formula:
    """``A[φ U ψ]`` — on *every* path, φ holds at every step
    until ψ holds. Expands to the standard duality:
    ``¬E[¬ψ U (¬φ ∧ ¬ψ)] ∧ ¬EG ¬ψ``. The left conjunct rules
    out any path with a ψ-violation reached without satisfying
    ¬φ-then-¬ψ first; the right conjunct rules out paths where
    ψ never holds."""
    return And(
        Not(EU(Not(right), And(Not(left), Not(right)))),
        Not(EG(Not(right))),
    )


def conj(*args: Formula) -> Formula:
    """Fold a sequence of formulae into a right-associated
    conjunction. Convenience for users — ``conj(a, b, c)`` is
    nicer to read than ``And(a, And(b, c))``."""
    if not args:
        return _true()
    result = args[-1]
    for f in reversed(args[:-1]):
        result = And(f, result)
    return result


def disj(*args: Formula) -> Formula:
    """Fold a sequence of formulae into a right-associated
    disjunction. Mirror of :func:`conj`."""
    if not args:
        # Empty disjunction is "false" — no atom satisfies it.
        return Atom(lambda m: False, "false")
    result = args[-1]
    for f in reversed(args[:-1]):
        result = Or(f, result)
    return result


def implies(antecedent: Formula, consequent: Formula) -> Formula:
    """Material implication ``φ → ψ`` desugared to ``¬φ ∨ ψ``.
    Reads cleaner than the equivalent ``Or(Not(...), ...)`` at
    the call site, which matters for compound safety / liveness
    formulae like ``AG (request → AF response)``."""
    return Or(Not(antecedent), consequent)


# ---------------------------------------------------------------------------
# Atomic-proposition helpers. These cover the common cases —
# build an ``Atom`` directly with a custom callable when none of
# them fit.
# ---------------------------------------------------------------------------


def place_has_token(place: str) -> Atom:
    """Atom that holds in every marking with at least one token
    at ``place``. The most common atomic proposition for workflow
    nets — *"is there work in this state right now?"*."""
    return Atom(lambda m: m.get(place, 0) >= 1, f"has_token({place})")


def place_empty(place: str) -> Atom:
    """Atom that holds in every marking where ``place`` has zero
    tokens. Dual of :func:`place_has_token`. Useful for
    pre-condition checks — *"the lock is free"*, *"the buffer is
    empty"*."""
    return Atom(lambda m: m.get(place, 0) == 0, f"empty({place})")


def place_count_eq(place: str, n: int) -> Atom:
    """Atom that holds when ``place`` has *exactly* ``n`` tokens.
    Useful for batching / quorum predicates — *"the buffer holds
    exactly six items"*."""
    return Atom(
        lambda m: m.get(place, 0) == n, f"count({place}) == {n}"
    )


def place_count_ge(place: str, n: int) -> Atom:
    """Atom that holds when ``place`` has at least ``n`` tokens.
    Useful for batching thresholds — *"there are enough items to
    process a batch"*."""
    return Atom(
        lambda m: m.get(place, 0) >= n, f"count({place}) >= {n}"
    )


def transition_enabled(net: PetriNet, transition: str) -> Atom:
    """Atom that holds when ``transition`` is enabled — its
    preset has enough tokens *and* its inhibitor preset is empty
    *and* (if applicable) its guard accepts. Defers to
    :meth:`PetriNet.is_enabled` so the predicate stays consistent
    with everything else the token-game does."""
    return Atom(
        lambda m: net.is_enabled(transition, m),
        f"enabled({transition})",
    )


# ---------------------------------------------------------------------------
# Model checker. The core function evaluates a formula bottom-up,
# computing the set of states satisfying each sub-formula.
# Returns a CTLResult bundling the satisfying set, whether the
# initial marking is in it, and (for failed AG / AF properties)
# a sample counterexample marking that violates the formula.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CTLResult:
    """Outcome of a CTL model-checking query.

    Attributes
    ----------
    holds_at
        The full set of reachable markings at which the formula
        is satisfied.
    holds_at_initial
        Whether the net's initial marking is in ``holds_at`` —
        the standard yes/no answer for the property.
    counterexample
        A specific reachable marking that *violates* the formula,
        or ``None`` if the formula holds at every reachable
        marking. Useful for `AG φ` failures: the counterexample
        is one of the states where φ fails to hold.
    """

    holds_at: frozenset[Marking]
    holds_at_initial: bool
    counterexample: Marking | None = None


def check_ctl(
    net: PetriNet,
    formula: Formula,
    *,
    max_states: int = 10_000,
) -> CTLResult:
    """Model-check ``formula`` against ``net``.

    Builds the reachability graph and recursively evaluates the
    formula bottom-up. Returns a :class:`CTLResult` with the full
    satisfying set, the initial-marking answer, and a sample
    counterexample marking if the formula fails to hold at every
    reachable marking.
    """
    # Build the LTS once; both atomic-proposition evaluation and
    # the temporal-operator fixed points read from the same graph.
    lts = reachability_graph(net, max_states=max_states)

    # Successor map (forward) — used by EX and the EU fixed point.
    # We add an implicit self-loop at every deadlock state (a state
    # with no outgoing transitions in the LTS). This is the standard
    # Kripke-structure convention for CTL semantics: requiring every
    # state to have at least one successor avoids surprises at
    # terminating states. Without it, EG φ would be vacuously false
    # on every terminating workflow net, and AX φ would be vacuously
    # true — both unintuitive for process analysts who think of the
    # final marking as a *stable* end state, not an unobservable one.
    successors: dict[Marking, set[Marking]] = {s: set() for s in lts.states}
    for src, _label, dst in lts.transitions:
        successors[src].add(dst)
    for s in lts.states:
        if not successors[s]:
            successors[s].add(s)

    holds_at = _evaluate(formula, lts.states, successors)
    initial_holds = lts.initial in holds_at

    # Counterexample: any reachable marking where the formula
    # fails. None if the formula holds everywhere — that's the
    # cleaner "no counterexamples" signal.
    counterexample: Marking | None = None
    if len(holds_at) < len(lts.states):
        # Deterministic pick — sort the failing markings so the
        # counterexample is stable across runs.
        failing = sorted(lts.states - holds_at, key=lambda m: sorted(m))
        counterexample = failing[0]

    return CTLResult(
        holds_at=frozenset(holds_at),
        holds_at_initial=initial_holds,
        counterexample=counterexample,
    )


def satisfies(
    net: PetriNet,
    formula: Formula,
    *,
    max_states: int = 10_000,
) -> bool:
    """Convenience wrapper: does the net's initial marking
    satisfy ``formula``? Equivalent to
    ``check_ctl(net, formula).holds_at_initial`` but reads better
    at call sites where only the boolean answer matters."""
    return check_ctl(net, formula, max_states=max_states).holds_at_initial


def _evaluate(
    formula: Formula,
    states: frozenset[Marking],
    successors: dict[Marking, set[Marking]],
) -> set[Marking]:
    """Recursive bottom-up evaluation. Returns the set of states
    satisfying ``formula``. Each operator's case is the standard
    CTL semantics; the three temporal primitives (EX, EU, EG)
    do their own fixed-point or pre-image computation."""
    if isinstance(formula, Atom):
        # Atomic proposition: evaluate the predicate against each
        # marking, materialised as a dict for the predicate's sake.
        return {s for s in states if formula.predicate(dict(s))}

    if isinstance(formula, Not):
        sub = _evaluate(formula.arg, states, successors)
        return set(states) - sub

    if isinstance(formula, And):
        left = _evaluate(formula.left, states, successors)
        right = _evaluate(formula.right, states, successors)
        return left & right

    if isinstance(formula, Or):
        left = _evaluate(formula.left, states, successors)
        right = _evaluate(formula.right, states, successors)
        return left | right

    if isinstance(formula, EX):
        # EX φ — states with at least one successor in the
        # satisfying set of φ. One-step pre-image.
        sub = _evaluate(formula.arg, states, successors)
        return {s for s in states if successors[s] & sub}

    if isinstance(formula, EU):
        # E[φ U ψ] — least fixed point of
        #   ψ ∨ (φ ∧ EX(E[φ U ψ]))
        # i.e. states satisfying ψ, or satisfying φ and having a
        # successor in the current iterate.
        phi = _evaluate(formula.left, states, successors)
        psi = _evaluate(formula.right, states, successors)
        current = set(psi)
        while True:
            new = set(current)
            # States in φ with at least one successor already in
            # current — they should also be in the next iterate.
            for s in phi - current:
                if successors[s] & current:
                    new.add(s)
            if new == current:
                return current
            current = new

    if isinstance(formula, EG):
        # EG φ — greatest fixed point of
        #   φ ∧ EX(EG φ)
        # i.e. states in φ that have at least one successor also
        # in the iterate. Start with all φ-states and shrink until
        # stable. The remaining states form an infinite path
        # within φ — either a cycle or a chain into one.
        phi = _evaluate(formula.arg, states, successors)
        current = set(phi)
        while True:
            new = {s for s in current if successors[s] & current}
            if new == current:
                return current
            current = new

    raise TypeError(f"unknown CTL formula node: {type(formula).__name__}")
