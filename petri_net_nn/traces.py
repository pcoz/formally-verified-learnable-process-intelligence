"""Utilities for training PetriNetModule on XES execution logs.

This module is the bridge from §10 Step 3 (XES logs as training data) to
§7.1 (process execution prediction) and §7.2 (anomaly detection).

The training signal is straightforward: each trace tells us which task
transitions actually fired during that process instance. We turn that
into a per-trace target vector — 1.0 for transitions whose label
matches an event in the trace, 0.0 for the rest — and fit the
compiled network so its forward-pass transition activations approach
that vector.

Auto-generated gateway transitions (those whose label contains "->",
which `parse_bpmn` emits for XOR-split / XOR-join branches) are
excluded from the supervision by default, since XES logs don't record
gateway firings.

Anomaly detection (§7.2): once trained, ``anomaly_score`` reports the
per-transition absolute deviation between the network's prediction and
the trace's observed occurrence vector. An out-of-distribution trace —
one whose path through the process doesn't match what the learned
weights predict — produces large per-transition residuals on the
diverging arcs, which §7.2 calls "interpretable" at the granularity of
the BPMN element.
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from petri_net_nn.compiler import PetriNetModule
from petri_net_nn.petri_net import PetriNet
from petri_net_nn.xes import XESTrace


AttributeToMarking = Callable[[XESTrace], dict[str, float]]
AttributeToValues = Callable[[XESTrace], dict[str, float]]


def _scored_transitions(net: PetriNet) -> list[str]:
    return sorted(
        t
        for t in net.transitions
        if "->" not in net.transition_labels.get(t, t)
    )


def _label_to_transition(net: PetriNet, transitions: Iterable[str]) -> dict[str, str]:
    return {net.transition_labels.get(t, t): t for t in transitions}


def trace_occurrence_vector(
    net: PetriNet,
    trace: XESTrace,
    transitions: list[str],
) -> torch.Tensor:
    """Return a 0/1 tensor indicating which of ``transitions`` fired in
    ``trace`` — matched by transition label against each event's
    ``concept:name``. Unknown event names are silently ignored."""
    label_map = _label_to_transition(net, transitions)
    fired = {label_map[e.name] for e in trace.events if e.name in label_map}
    return torch.tensor([1.0 if t in fired else 0.0 for t in transitions])


def _stack_input_markings(
    markings: list[dict[str, float]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    all_places = set().union(*(m.keys() for m in markings)) if markings else set()
    return {
        p: torch.tensor([m.get(p, 0.0) for m in markings], device=device)
        for p in all_places
    }


def train_on_traces(
    module: PetriNetModule,
    traces: list[XESTrace],
    *,
    attribute_to_marking: AttributeToMarking,
    attribute_to_values: AttributeToValues | None = None,
    steps: int = 500,
    lr: float = 0.1,
    transitions: list[str] | None = None,
) -> list[float]:
    """Fit ``module`` so its transition activations match each trace's
    observed occurrence vector under that trace's derived input marking.

    The default supervision target is every transition whose label
    doesn't look auto-generated (i.e. doesn't contain ``->``); pass
    ``transitions`` to override that selection. Returns the per-step
    loss trajectory.

    ``attribute_to_values`` is the coloured-Petri-net analogue of
    ``attribute_to_marking``: it returns the scalar value carried by
    the token at each source place for a given trace. Optional —
    omit it for plain (uncoloured) training. When present, the
    compiled module's learnable structural-guard thresholds train
    against these values, so the model can refine the declared
    threshold from data."""
    net = module.net
    scored = transitions if transitions is not None else _scored_transitions(net)
    if not scored:
        raise ValueError("no transitions to score against")

    device = next(module.parameters()).device
    input_marking = _stack_input_markings(
        [attribute_to_marking(t) for t in traces], device
    )
    input_values = (
        _stack_input_markings(
            [attribute_to_values(t) for t in traces], device
        )
        if attribute_to_values is not None
        else None
    )
    targets = torch.stack(
        [trace_occurrence_vector(net, t, scored) for t in traces]
    ).to(device)

    opt = torch.optim.Adam(module.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        out = module(
            input_marking=input_marking,
            input_values=input_values,
            batch_size=len(traces),
        )
        predicted = torch.stack([out[t] for t in scored], dim=1)
        loss = F.binary_cross_entropy(predicted.clamp(1e-6, 1 - 1e-6), targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


class SharpnessScheduler:
    """Anneal ``PetriNetModule.sharpness`` over training.

    The continuous relaxation in §4.2 trades faithfulness for gradient
    flow: a low ``sharpness`` gives smooth sigmoids that train well but
    don't really enforce step-like firing; a high ``sharpness`` gives
    near-step firing but with small gradients far from the threshold.
    Annealing — start low, finish high — gets both benefits in one
    training run.

    Use it like a learning-rate scheduler: call ``.step()`` once per
    optimizer step and the module's ``sharpness`` attribute is updated
    in place. The forward pass picks up the new value on its next call.
    """

    def __init__(
        self,
        module,
        *,
        start: float = 1.0,
        end: float = 8.0,
        num_steps: int,
        kind: str = "linear",
    ) -> None:
        if kind not in ("linear", "exponential"):
            raise ValueError(f"kind must be 'linear' or 'exponential', got {kind!r}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if kind == "exponential" and (start <= 0 or end <= 0):
            raise ValueError(
                "exponential schedule requires positive start and end"
            )
        self.module = module
        self.start = float(start)
        self.end = float(end)
        self.num_steps = num_steps
        self.kind = kind
        self._step = 0
        module.sharpness = self.start

    def step(self) -> float:
        self._step += 1
        frac = min(1.0, self._step / self.num_steps)
        if self.kind == "linear":
            value = self.start + frac * (self.end - self.start)
        else:
            value = self.start * (self.end / self.start) ** frac
        self.module.sharpness = value
        return value


def sweep_trace_count(
    module_factory,
    traces: list[XESTrace],
    *,
    attribute_to_marking: AttributeToMarking,
    sample_sizes: list[int],
    steps: int = 500,
    lr: float = 0.1,
) -> dict[int, float]:
    """Train a fresh module on the first N traces for each N in
    ``sample_sizes`` and return the final loss for each. Used to
    answer §8's "training data requirements" question — plotting
    the returned dict shows how quickly each subnet shape converges
    as more traces become available.

    ``module_factory`` must return a fresh module on each call so that
    the runs don't share parameters."""
    results: dict[int, float] = {}
    for n in sample_sizes:
        subset = traces[:n]
        if not subset:
            results[n] = float("nan")
            continue
        module = module_factory()
        losses = train_on_traces(
            module,
            subset,
            attribute_to_marking=attribute_to_marking,
            steps=steps,
            lr=lr,
        )
        results[n] = losses[-1]
    return results


def trace_anomaly_score(
    module: PetriNetModule,
    trace: XESTrace,
    *,
    attribute_to_marking: AttributeToMarking,
    attribute_to_values: AttributeToValues | None = None,
    transitions: list[str] | None = None,
) -> float:
    """Trace-level scalar anomaly score: the sum of per-transition
    absolute residuals returned by :func:`anomaly_score`. Higher means
    more anomalous; useful for ranking traces and computing AUC."""
    per_transition = anomaly_score(
        module, trace,
        attribute_to_marking=attribute_to_marking,
        attribute_to_values=attribute_to_values,
        transitions=transitions,
    )
    return sum(per_transition.values())


def expected_cost(
    module,
    transition_costs: dict[str, float],
    *,
    input_marking: dict[str, "torch.Tensor"] | None = None,
    batch_size: int | None = None,
) -> "torch.Tensor":
    """Expected cost-to-completion under a trained module.

    For each transition with a configured cost weight, multiply its
    forward-pass activation by the weight and sum. This is the
    realised-execution cost the §6-framing-bullet promises: a number
    you can read off two bisimilar variants under the same trace
    distribution and use to rank them by total cost.

    Transitions absent from ``transition_costs`` contribute zero;
    transitions in the dict that aren't in the net are ignored
    (caller's responsibility to keep them aligned)."""
    out = module(input_marking=input_marking, batch_size=batch_size)
    total = None
    for t, cost in transition_costs.items():
        if t not in out:
            continue
        contribution = out[t] * float(cost)
        total = contribution if total is None else total + contribution
    if total is None:
        return torch.zeros(batch_size or 1, device=module._device())
    return total


def auc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Mann-Whitney U / ROC AUC: the probability that a randomly drawn
    positive (anomalous) trace scores higher than a randomly drawn
    negative (normal) trace. Ties count as half. Returns ``nan`` if
    either list is empty."""
    if not positive_scores or not negative_scores:
        return float("nan")
    pos = torch.as_tensor(positive_scores, dtype=torch.float64)
    neg = torch.as_tensor(negative_scores, dtype=torch.float64)
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum().item()
    ties = (diff == 0).sum().item()
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def anomaly_score(
    module: PetriNetModule,
    trace: XESTrace,
    *,
    attribute_to_marking: AttributeToMarking,
    attribute_to_values: AttributeToValues | None = None,
    transitions: list[str] | None = None,
) -> dict[str, float]:
    """Score one trace under a trained module. Returns a dict mapping
    each scored transition to the absolute residual between the
    network's predicted activation and the trace's observed firing
    (0 or 1). Summing the values gives a single trace-level anomaly
    score; inspecting individual entries identifies *which* parts of
    the process structure diverge — the §7.2 interpretability claim.

    ``attribute_to_values`` mirrors the same kwarg on
    ``train_on_traces``: pass it when scoring coloured-Petri-net
    traces so the trained guard thresholds get the value channel
    they need to gate firings correctly."""
    net = module.net
    scored = transitions if transitions is not None else _scored_transitions(net)

    device = next(module.parameters()).device
    input_marking = _stack_input_markings([attribute_to_marking(trace)], device)
    input_values = (
        _stack_input_markings([attribute_to_values(trace)], device)
        if attribute_to_values is not None
        else None
    )
    target = trace_occurrence_vector(net, trace, scored).to(device)

    module.eval()
    with torch.no_grad():
        out = module(
            input_marking=input_marking,
            input_values=input_values,
            batch_size=1,
        )
        predicted = torch.stack([out[t] for t in scored], dim=1).squeeze(0)
        residuals = (predicted - target).abs()
    return {t: residuals[i].item() for i, t in enumerate(scored)}
