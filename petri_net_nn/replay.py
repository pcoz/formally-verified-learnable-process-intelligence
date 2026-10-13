"""Order-aware training and scoring by token replay.

:func:`petri_net_nn.traces.train_on_traces` supervises a module with
each trace's *occurrence vector* — which transitions fired at all.
That signal is order-free: ``(A, B, P)`` and ``(B, A, P)`` produce the
same target, so a dependency such as "after two FALL days the story
exits" cannot be learned from it, however the net is shaped.

This module adds the sequential counterpart. Each trace is
**replayed** on the net with the discrete token game. At every event
the module sees the current marking and the set of visible
transitions that are *enabled* there (after firing any silent τ
transitions needed to enable them), and is trained to maximise the
log-likelihood of the transition that actually fired next::

    P(t | marking, case) = softmax over enabled t' of  logit(t')
    logit(t) = rate(t) · sharpness · (Σ_{p ∈ •t} w(p,t) · a(p) − θ_t)

``w`` and ``θ`` are the module's own arc weights and transition
thresholds — the same parameters :func:`train_on_traces` fits and that
:func:`~petri_net_nn.interpretability.extract_xor_rule` reads — so the
whole interpretability toolkit applies unchanged. ``a(p)`` is the
token count at ``p`` multiplied by a per-place case value from the
existing ``attribute_to_marking`` mechanism (default 1.0): a marked
place "carries" the case attribute, so routing at that place becomes a
logistic function of the attribute and :func:`extract_xor_rule` /
:func:`extract_xor_partition` read the crossover directly.

The replay path is discrete and parameter-independent, so it is
compiled once into flat tensors; training is then a single vectorised
softmax cross-entropy per optimisation step and handles cyclic nets
with no time unrolling.

Pairs naturally with
:func:`petri_net_nn.discovery.discover_transition_system`, whose
state-machine nets make every place a named state (``"FALL > FALL"``)
and every routing decision an ordered dependency.

Termination is not itself a choice here: logs whose ending matters
should carry an explicit terminal event (e.g. ``EXIT``).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable

import torch

from petri_net_nn.compiler import PetriNetModule
from petri_net_nn.petri_net import PetriNet
from petri_net_nn.xes import XESTrace

AttributeToMarking = Callable[[XESTrace], dict[str, float]]
Marking = dict[str, int]


def _key(m: Marking) -> tuple:
    return tuple(sorted((p, c) for p, c in m.items() if c > 0))


def _closure(net: PetriNet, marking: Marking, max_markings: int) -> list[Marking]:
    """Markings reachable from ``marking`` by firing silent transitions
    only (BFS order, ``marking`` first), capped at ``max_markings``."""
    silent = sorted(net.silent_transitions)
    seen = {_key(marking)}
    out = [marking]
    queue = deque([marking])
    while queue and len(out) < max_markings:
        m = queue.popleft()
        for t in silent:
            if net.is_enabled(t, m):
                n = net.fire(t, m)
                k = _key(n)
                if k not in seen:
                    seen.add(k)
                    out.append(n)
                    queue.append(n)
    return out


def enabled_choices(
    net: PetriNet, marking: Marking, *, max_markings: int = 64
) -> dict[str, Marking]:
    """Visible transitions that can fire next from ``marking``, each
    mapped to the (τ-closure) marking in which it is enabled — the
    first such marking in BFS order, i.e. via the fewest τ firings."""
    choices: dict[str, Marking] = {}
    visible = sorted(t for t in net.transitions if t not in net.silent_transitions)
    for m in _closure(net, marking, max_markings):
        for t in visible:
            if t not in choices and net.is_enabled(t, m):
                choices[t] = m
    return choices


@dataclass(frozen=True)
class ReplayStep:
    """One decision in a replayed trace: under ``marking`` the
    ``candidates`` were enabled and ``fired`` matched the event."""

    trace_index: int
    position: int
    event: str
    candidates: tuple[str, ...]
    contexts: tuple[tuple[tuple[str, int], ...], ...]
    fired: str | None  # None when the event matched no enabled transition


def replay_trace(
    net: PetriNet, events: list[str], *, trace_index: int = 0, max_markings: int = 64
) -> tuple[list[ReplayStep], bool]:
    """Token-replay ``events`` on ``net`` from its initial marking.

    Returns ``(steps, fits)``. Replay stops at the first event no
    enabled transition can produce; that step is recorded with
    ``fired=None`` and ``fits`` is False."""
    marking: Marking = dict(net.initial_marking)
    steps: list[ReplayStep] = []
    for pos, ev in enumerate(events):
        choices = enabled_choices(net, marking, max_markings=max_markings)
        cands = tuple(sorted(choices))
        ctx = tuple(
            tuple(sorted((p, choices[t].get(p, 0)) for p in net.preset(t))) for t in cands
        )
        match = next(
            (t for t in cands if net.transition_labels.get(t, t) == ev), None
        )
        steps.append(ReplayStep(trace_index, pos, ev, cands, ctx, match))
        if match is None:
            return steps, False
        marking = net.fire(match, choices[match])
    return steps, True


def _place_values(trace: XESTrace, attribute_to_marking: AttributeToMarking | None) -> dict[str, float]:
    return dict(attribute_to_marking(trace)) if attribute_to_marking else {}


class _Compiled:
    """Flat tensor form of a replayed log, for vectorised training."""

    def __init__(self, module: PetriNetModule, traces, attribute_to_marking, max_markings):
        net = module.net
        self.arc_keys = list(module._arc_key.values())
        arc_index = {edge: i for i, edge in enumerate(module._arc_key)}
        self.theta_keys = list(module._threshold_key.values())
        theta_index = {t: i for i, t in enumerate(module._threshold_key)}
        entry_dec, entry_theta, entry_scale, target_entry = [], [], [], []
        arc_entry, arc_param, arc_act = [], [], []
        self.n_fitting = 0
        self.n_unmatched = 0
        dec = 0
        for ti, trace in enumerate(traces):
            values = _place_values(trace, attribute_to_marking)
            steps, fits = replay_trace(
                net, [e.name for e in trace.events], trace_index=ti, max_markings=max_markings
            )
            self.n_fitting += fits
            for st in steps:
                if st.fired is None:
                    self.n_unmatched += 1
                    continue
                for t, ctx in zip(st.candidates, st.contexts):
                    e = len(entry_dec)
                    entry_dec.append(dec)
                    entry_theta.append(theta_index[t])
                    entry_scale.append(net.rate(t) * module.sharpness)
                    if t == st.fired:
                        target_entry.append(e)
                    for p, tokens in ctx:
                        arc_entry.append(e)
                        arc_param.append(arc_index[(p, t)])
                        arc_act.append(tokens * float(values.get(p, 1.0)))
                dec += 1
        self.n_decisions = dec
        self.entry_dec = torch.tensor(entry_dec, dtype=torch.long)
        self.entry_theta = torch.tensor(entry_theta, dtype=torch.long)
        self.entry_scale = torch.tensor(entry_scale, dtype=torch.float32)
        self.target_entry = torch.tensor(target_entry, dtype=torch.long)
        self.arc_entry = torch.tensor(arc_entry, dtype=torch.long)
        self.arc_param = torch.tensor(arc_param, dtype=torch.long)
        self.arc_act = torch.tensor(arc_act, dtype=torch.float32)

    def log_probs(self, module: PetriNetModule, *, with_argmax: bool = False):
        """Log-probability of the fired transition at every decision
        (and, with ``with_argmax``, whether it was the most likely one)."""
        w = torch.stack([module.arc_weights[k] for k in self.arc_keys])
        theta = torch.stack([module.transition_thresholds[k] for k in self.theta_keys])
        n_e = self.entry_dec.shape[0]
        weighted = torch.zeros(n_e).index_add(0, self.arc_entry, w[self.arc_param] * self.arc_act)
        logits = self.entry_scale * (weighted - theta[self.entry_theta])
        mx = torch.full((self.n_decisions,), -1e30).scatter_reduce(
            0, self.entry_dec, logits.detach(), reduce="amax"
        )
        sumexp = torch.zeros(self.n_decisions).index_add(
            0, self.entry_dec, torch.exp(logits - mx[self.entry_dec])
        )
        lse = mx + torch.log(sumexp)
        lp = logits[self.target_entry] - lse
        if with_argmax:
            return lp, logits[self.target_entry].detach() >= mx[: self.n_decisions] - 1e-9
        return lp


def train_by_replay(
    module: PetriNetModule,
    traces: list[XESTrace],
    *,
    attribute_to_marking: AttributeToMarking | None = None,
    steps: int = 300,
    lr: float = 0.1,
    weight_decay: float = 0.0,
    max_markings: int = 64,
) -> list[float]:
    """Fit ``module`` by maximum likelihood of each trace's *ordered*
    firing sequence under token replay (see module docstring).

    ``attribute_to_marking`` returns per-place case values; a marked
    place's activation is its token count times that value (places not
    in the dict use 1.0). Events that no enabled transition can produce
    end that trace's replay (they are counted, not trained on).
    ``weight_decay`` is passed to Adam (L2 regularisation of the arc
    weights and thresholds). Returns the per-step mean negative
    log-likelihood."""
    compiled = _Compiled(module, traces, attribute_to_marking, max_markings)
    if compiled.n_decisions == 0:
        raise ValueError("no replayable decisions in the trace set")
    opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    module.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = -compiled.log_probs(module).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def next_step_distribution(
    module: PetriNetModule,
    marking: Marking,
    *,
    place_values: dict[str, float] | None = None,
    max_markings: int = 64,
) -> dict[str, float]:
    """``{transition: probability}`` over the visible transitions enabled
    (after τ-closure) in ``marking``, under the trained module."""
    net = module.net
    choices = enabled_choices(net, marking, max_markings=max_markings)
    if not choices:
        return {}
    values = place_values or {}
    logits = []
    with torch.no_grad():
        for t, m in sorted(choices.items()):
            s = sum(
                module.arc_weights[module._arc_key[(p, t)]].item()
                * m.get(p, 0) * float(values.get(p, 1.0))
                for p in net.preset(t)
            )
            theta = module.transition_thresholds[module._threshold_key[t]].item()
            logits.append(net.rate(t) * module.sharpness * (s - theta))
    mx = max(logits)
    z = [math.exp(x - mx) for x in logits]
    return {t: v / sum(z) for t, v in zip(sorted(choices), z)}


def replay_surprisal(
    module: PetriNetModule,
    trace: XESTrace,
    *,
    attribute_to_marking: AttributeToMarking | None = None,
    unseen_logp: float = math.log(1e-3),
) -> list[tuple[int, str, str | None, float]]:
    """Per-event ``(position, event, transition, -log P)`` under replay.

    The anomaly signal is pinned to the named transition that fired;
    an event the net cannot produce scores ``-unseen_logp`` with
    transition ``None`` and ends the replay."""
    net = module.net
    values = _place_values(trace, attribute_to_marking)
    marking: Marking = dict(net.initial_marking)
    out = []
    for pos, e in enumerate(trace.events):
        dist = next_step_distribution(module, marking, place_values=values)
        match = next((t for t in sorted(dist) if net.transition_labels.get(t, t) == e.name), None)
        if match is None:
            out.append((pos, e.name, None, -unseen_logp))
            break
        out.append((pos, e.name, match, -math.log(max(dist[match], 1e-12))))
        marking = net.fire(match, enabled_choices(net, marking)[match])
    return out


def replay_anomaly_score(
    module: PetriNetModule,
    trace: XESTrace,
    *,
    attribute_to_marking: AttributeToMarking | None = None,
) -> dict[str, float]:
    """Order-aware analogue of :func:`petri_net_nn.traces.anomaly_score`:
    total surprisal (−log P, nats) per transition along the replay.
    Unproducible events are reported under the key ``"<unfit:EVENT>"``."""
    scores: dict[str, float] = {}
    for _, ev, t, s in replay_surprisal(module, trace, attribute_to_marking=attribute_to_marking):
        k = t if t is not None else f"<unfit:{ev}>"
        scores[k] = scores.get(k, 0.0) + s
    return scores


def replay_log_likelihood(
    module: PetriNetModule,
    traces: list[XESTrace],
    *,
    attribute_to_marking: AttributeToMarking | None = None,
    max_markings: int = 64,
) -> dict[str, float]:
    """Held-out evaluation: mean log-likelihood per replayed decision,
    next-step accuracy (the fired transition was the most likely), and
    fit counts. Unmatched events are excluded from the means and
    reported separately."""
    compiled = _Compiled(module, traces, attribute_to_marking, max_markings)
    with torch.no_grad():
        lp, top = compiled.log_probs(module, with_argmax=True)
    acc = float(top.float().mean()) if len(lp) else float("nan")
    return {
        "mean_log_likelihood": float(lp.mean()) if len(lp) else float("nan"),
        "decisions": compiled.n_decisions,
        "unmatched_events": compiled.n_unmatched,
        "fitting_traces": compiled.n_fitting,
        "traces": len(traces),
        "accuracy": acc,
    }


@dataclass(frozen=True)
class NextStepRule:
    """Learned next-step distribution at one place (a named state for
    nets from :func:`discover_transition_system`)."""

    place: str
    place_label: str
    probabilities: tuple[tuple[str, float], ...]  # (transition label, P), descending
    value: float

    def description(self) -> str:
        parts = ", ".join(f"{lab} {p:.0%}" for lab, p in self.probabilities)
        return f"after {self.place_label!r}: next → {parts}"


def extract_next_step_rules(
    module: PetriNetModule, *, value: float = 1.0, min_choices: int = 2
) -> list[NextStepRule]:
    """For every place with at least ``min_choices`` visible output
    transitions, the trained next-step distribution when that place
    holds the single token carrying case value ``value``."""
    net = module.net
    rules = []
    for p in sorted(net.places):
        outs = [t for t in net.postset(p) if t not in net.silent_transitions]
        if len(outs) < min_choices:
            continue
        dist = next_step_distribution(module, {p: 1}, place_values={p: value})
        probs = sorted(
            ((net.transition_labels.get(t, t), q) for t, q in dist.items()),
            key=lambda x: -x[1],
        )
        rules.append(NextStepRule(p, net.place_labels.get(p, p), tuple(probs), value))
    return rules
