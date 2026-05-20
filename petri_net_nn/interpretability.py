"""Distil learned weights back into readable decision rules.

Addresses §8's fourth open problem from the architecture spec:

  > The structural interpretability of the architecture — which subnet
  > shows the anomaly — is a genuine advantage over black-box ML. But
  > the learned weights within each subnet are not directly
  > interpretable. … The next research question is whether the learned
  > weights can be distilled back into readable decision rules —
  > closing the loop from neural learning to interpretable business
  > logic.

Three pieces:

  * **Rule extraction.** ``extract_routing_rules`` walks the compiled
    module's structure for XOR-shape transitions (N transitions
    sharing one input place, no other consumers) and reads the
    learned weights / thresholds out as a single crossover value per
    pair: "input above X → transition A, otherwise → transition B".
    ``extract_and_join_rules`` does the same for synchronisation
    transitions (multi-input AND-joins), reading the input weights
    and threshold as a weighted-vote or uniform-quorum rule.

  * **Bootstrap confidence intervals.** ``bootstrap_xor_rule`` and
    ``bootstrap_and_join_rule`` resample the training trace list
    with replacement N times, retrain a fresh module per resample,
    extract the rule, and report the distribution of rule
    parameters. Returns ``XORRuleCI`` / ``AndJoinRuleCI`` with
    percentile-based confidence intervals and a direction-agreement
    rate — the Phase 13 answer to "should I trust this rule in
    production?".

  * **Prose explanations.** ``explain_anomaly`` walks residuals and
    formats them as a paragraph; ``prose_for_xor_rule`` and
    ``prose_for_and_join_rule`` do the same for rules (with or
    without bootstrap CIs attached). All three turn the extractors'
    structured output into a plain-English paragraph a non-technical
    reader can act on.

The scaffold restricts XOR rule extraction to two-way splits — the
common case. N-way splits compute pairwise crossovers between the top
two transitions but flag the rule as approximate; full N-way rule
extraction is a follow-up.
"""
from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field
from typing import Callable

import torch

from petri_net_nn.compiler import PetriNetModule
from petri_net_nn.petri_net import PetriNet
from petri_net_nn.traces import (
    AttributeToMarking,
    AttributeToValues,
    anomaly_score,
    train_on_traces,
)
from petri_net_nn.xes import XESTrace


@dataclass(frozen=True)
class XORRule:
    """A distilled binary routing rule.

    The trained network routes to ``transition_above`` when the
    activation of ``input_place`` exceeds ``crossover``, and to
    ``transition_below`` otherwise. ``label_above`` / ``label_below``
    are the BPMN names of those transitions (falling back to the
    transition IDs when unlabelled).
    """

    input_place: str
    transition_above: str
    transition_below: str
    label_above: str
    label_below: str
    crossover: float
    confidence: float

    def description(self) -> str:
        return (
            f"if {self.input_place!r} > {self.crossover:.3f} → "
            f"{self.label_above!r} (else {self.label_below!r}); "
            f"weight gap |Δw|={self.confidence:.3f}"
        )


@dataclass(frozen=True)
class XORRegion:
    """One interval of an N-way XOR routing partition: the trained
    network selects ``transition`` (BPMN label ``label``) whenever the
    activation of the shared input place lies in ``[lower, upper]``."""

    lower: float
    upper: float
    transition: str
    label: str


@dataclass(frozen=True)
class AndJoinRule:
    """A distilled AND-join (synchronisation) rule.

    The transition fires when ``sum(weight_i * a(input_i)) > threshold``
    — a weighted vote over the input places. ``inputs`` lists each
    input place's BPMN label and its learned weight; ``summary`` is a
    short natural-language gloss (uniform-weight quorum, or explicit
    weighted vote if weights vary)."""

    transition: str
    label: str
    inputs: tuple[tuple[str, str, float], ...]
    threshold: float
    summary: str

    def description(self) -> str:
        return f"{self.label!r}: {self.summary}"


@dataclass(frozen=True)
class XORPartition:
    """N-way routing rule for a single XOR-shape group, expressed as
    contiguous input intervals each mapped to a winning transition."""

    input_place: str
    regions: tuple[XORRegion, ...]

    def description(self) -> str:
        parts = [
            f"[{r.lower:.3f}, {r.upper:.3f}] → {r.label!r}"
            for r in self.regions
        ]
        return f"{self.input_place!r}: " + "; ".join(parts)


def _downstream_label(net: PetriNet, transition: str) -> str:
    """Return the most informative label for a transition. If the
    transition's own label is auto-generated (contains "->") and the
    transition routes to a single downstream task, use the task's
    label instead — that's the BPMN element a business reader cares
    about. Otherwise fall back to the transition's own label."""
    own = net.transition_labels.get(transition, transition)
    if "->" not in own:
        return own
    outputs = net.postset(transition)
    if len(outputs) != 1:
        return own
    place = next(iter(outputs))
    consumers = [t for t in net.transitions if place in net.preset(t)]
    if len(consumers) != 1:
        return own
    return net.transition_labels.get(consumers[0], consumers[0])


def find_xor_groups(net: PetriNet) -> list[tuple[str, list[str]]]:
    """Return XOR-shape transition groups detected structurally.

    Two patterns count as XOR groups:

      * **Single-input XOR** — N transitions all sharing a single
        input place, each with that place as their only input. This
        is the §5 Subnet 2 / classic BPMN XOR-split.

      * **Shared-preset XOR** — N transitions all having the same
        (multi-place) input set. They compete for the same combined
        precondition; the trained weights and thresholds determine
        which fires. This shape arises in protocols like 2PC where
        a routing decision is gated by both a control token and a
        data message — the decision is XOR-shaped but the precondition
        is conjunctive.

    The returned tuple's first element is a *discriminative input
    place* — for the single-input case it is the unique input; for
    the shared-preset case it is one place from the shared preset
    (sorted, first). Rule-extraction callers can use this place as
    the input axis to read crossovers from, treating other shared
    inputs as constant context."""
    groups: list[tuple[str, list[str]]] = []
    seen_groups: set[tuple[str, ...]] = set()

    by_preset: dict[frozenset[str], list[str]] = {}
    for t in sorted(net.transitions):
        preset = frozenset(net.preset(t))
        if preset:
            by_preset.setdefault(preset, []).append(t)
    for preset, consumers in by_preset.items():
        if len(consumers) < 2:
            continue
        key = tuple(sorted(consumers))
        if key in seen_groups:
            continue
        seen_groups.add(key)
        groups.append((sorted(preset)[0], sorted(consumers)))

    return sorted(groups, key=lambda g: g[0])


def extract_xor_rule(
    module: PetriNetModule,
    input_place: str,
    transition_a: str,
    transition_b: str,
) -> XORRule:
    """Derive the routing crossover for one binary XOR pair from the
    trained weights. The continuous-relaxation comparison

        w_A·a(P) - θ_A   vs   w_B·a(P) - θ_B

    has a single crossover point at a(P) = (θ_A - θ_B) / (w_A - w_B)
    when the weight gap is non-zero, with direction determined by the
    sign of the gap. The crossover is the rule's threshold; the
    magnitude of the gap is its confidence (small gap = the two
    transitions discriminate weakly, regardless of where the crossover
    sits)."""
    net = module.net
    key = module._arc_key
    threshold_key = module._threshold_key

    w_a = module.arc_weights[key[(input_place, transition_a)]].item()
    w_b = module.arc_weights[key[(input_place, transition_b)]].item()
    theta_a = module.transition_thresholds[threshold_key[transition_a]].item()
    theta_b = module.transition_thresholds[threshold_key[transition_b]].item()

    delta_w = w_a - w_b
    delta_theta = theta_a - theta_b

    if abs(delta_w) < 1e-6:
        crossover = float("nan")
        above, below = transition_a, transition_b
    else:
        crossover = delta_theta / delta_w
        if delta_w > 0:
            above, below = transition_a, transition_b
        else:
            above, below = transition_b, transition_a

    return XORRule(
        input_place=input_place,
        transition_above=above,
        transition_below=below,
        label_above=_downstream_label(net, above),
        label_below=_downstream_label(net, below),
        crossover=crossover,
        confidence=abs(delta_w),
    )


def _discriminative_input(module: PetriNetModule, transitions: list[str]) -> str:
    """For a group of transitions that share their preset, return the
    input place whose learned weight gap across the group is largest.
    For single-input transitions this is just the lone input. For
    shared-preset multi-input groups (2PC-style decision gated on
    multiple places) this picks the place that the trained network
    actually uses to discriminate."""
    net = module.net
    preset = set(net.preset(transitions[0]))
    for t in transitions[1:]:
        preset &= set(net.preset(t))
    if not preset:
        raise ValueError(
            f"transitions {transitions!r} do not share any input place"
        )
    if len(preset) == 1:
        return next(iter(preset))
    gaps: dict[str, float] = {}
    for p in preset:
        weights = [
            module.arc_weights[module._arc_key[(p, t)]].item()
            for t in transitions
        ]
        gaps[p] = max(weights) - min(weights)
    return max(gaps, key=gaps.get)


def extract_routing_rules(module: PetriNetModule) -> list[XORRule]:
    """Apply :func:`extract_xor_rule` to every two-way XOR group in the
    net. N-way groups are skipped — use
    :func:`extract_routing_partitions` to get the piecewise
    representation that covers all arities. For shared-preset groups
    (multi-input competing transitions), the discriminative input is
    chosen automatically from the trained weight gaps."""
    rules: list[XORRule] = []
    for _, transitions in find_xor_groups(module.net):
        if len(transitions) != 2:
            continue
        place = _discriminative_input(module, transitions)
        rules.append(
            extract_xor_rule(module, place, transitions[0], transitions[1])
        )
    return rules


def extract_xor_partition(
    module: PetriNetModule,
    input_place: str,
    transitions: list[str],
    *,
    input_range: tuple[float, float] = (0.0, 1.0),
) -> XORPartition:
    """N-way XOR partition. For N competing transitions sharing
    ``input_place`` as their only input, derive the piecewise winner
    on the input interval ``[input_range[0], input_range[1]]``. Each
    transition is treated as a linear pre-activation
    ``w_i · a(P) - θ_i``; the partition records which transition has
    the maximum pre-activation on each contiguous sub-interval."""
    if len(transitions) < 2:
        raise ValueError(
            f"XOR partition requires at least 2 competing transitions, "
            f"got {len(transitions)}"
        )

    net = module.net
    key = module._arc_key
    threshold_key = module._threshold_key

    weights = [
        module.arc_weights[key[(input_place, t)]].item() for t in transitions
    ]
    thetas = [
        module.transition_thresholds[threshold_key[t]].item() for t in transitions
    ]

    boundaries: set[float] = set()
    lo, hi = input_range
    for i in range(len(transitions)):
        for j in range(i + 1, len(transitions)):
            dw = weights[i] - weights[j]
            if abs(dw) < 1e-9:
                continue
            x = (thetas[i] - thetas[j]) / dw
            if lo < x < hi:
                boundaries.add(x)

    edges = [lo] + sorted(boundaries) + [hi]
    raw_regions: list[tuple[float, float, int]] = []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        mid = (a + b) / 2.0
        scores = [(weights[i] * mid - thetas[i], i) for i in range(len(transitions))]
        winner = max(scores)[1]
        raw_regions.append((a, b, winner))

    merged: list[list] = []
    for a, b, w in raw_regions:
        if merged and merged[-1][2] == w:
            merged[-1][1] = b
        else:
            merged.append([a, b, w])

    regions = tuple(
        XORRegion(
            lower=a,
            upper=b,
            transition=transitions[w],
            label=_downstream_label(net, transitions[w]),
        )
        for a, b, w in merged
    )
    return XORPartition(input_place=input_place, regions=regions)


def find_and_join_transitions(net: PetriNet) -> list[str]:
    """Return transitions with at least two distinct input places — the
    synchronisation (AND-join) shape from §5 Subnet 4. The BPMN
    parser's XOR-join translation produces multiple transitions each
    with one input arc, so those are not picked up here."""
    return sorted(t for t in net.transitions if len(net.preset(t)) >= 2)


def extract_and_join_rule(
    module: PetriNetModule, transition: str
) -> AndJoinRule:
    """Distil an AND-join transition's learned weights into a
    weighted-vote rule. If the learned weights are roughly uniform the
    rule is rendered as a quorum (\"fires when at least k of n inputs
    are active\"); otherwise it is rendered as an explicit weighted
    sum threshold."""
    net = module.net
    preset = sorted(net.preset(transition))
    if len(preset) < 2:
        raise ValueError(
            f"transition {transition!r} has {len(preset)} inputs, "
            f"need at least 2 for an AND-join rule"
        )

    weights = [
        module.arc_weights[module._arc_key[(p, transition)]].item()
        for p in preset
    ]
    threshold = module.transition_thresholds[
        module._threshold_key[transition]
    ].item()

    n = len(preset)
    mean_w = sum(weights) / n
    label = net.transition_labels.get(transition, transition)
    inputs = tuple(
        (p, net.place_labels.get(p, p), w) for p, w in zip(preset, weights)
    )

    uniform = mean_w > 1e-3 and (max(weights) - min(weights)) < 0.3 * abs(mean_w)
    if uniform:
        effective_k = threshold / mean_w
        rounded_k = max(1, min(n, round(effective_k + 0.5)))
        if rounded_k == n:
            summary = f"fires when all {n} inputs are active"
        elif rounded_k == 1:
            summary = f"fires when at least 1 of {n} inputs is active"
        else:
            summary = (
                f"fires when at least {rounded_k} of {n} inputs are active"
            )
    else:
        parts = [f"{lbl!r}×{w:.2f}" for (_, lbl, w) in inputs]
        summary = f"fires when {' + '.join(parts)} > {threshold:.2f}"

    return AndJoinRule(
        transition=transition,
        label=label,
        inputs=inputs,
        threshold=threshold,
        summary=summary,
    )


def extract_and_join_rules(module: PetriNetModule) -> list[AndJoinRule]:
    """Apply :func:`extract_and_join_rule` to every synchronisation
    transition in the net."""
    return [
        extract_and_join_rule(module, t)
        for t in find_and_join_transitions(module.net)
    ]


def extract_routing_partitions(
    module: PetriNetModule,
    *,
    input_range: tuple[float, float] = (0.0, 1.0),
) -> list[XORPartition]:
    """Apply :func:`extract_xor_partition` to every XOR-shape group in
    the net, regardless of arity. For shared-preset groups the
    discriminative input is chosen automatically. Returns one
    ``XORPartition`` per group."""
    out: list[XORPartition] = []
    for _, transitions in find_xor_groups(module.net):
        place = _discriminative_input(module, transitions)
        out.append(
            extract_xor_partition(
                module, place, transitions, input_range=input_range
            )
        )
    return out


def explain_anomaly(
    module: PetriNetModule,
    trace: XESTrace,
    *,
    attribute_to_marking: AttributeToMarking,
    top_n: int = 5,
    threshold: float = 0.1,
) -> str:
    """Produce a human-readable explanation of why ``trace`` is
    anomalous under ``module``. Walks the per-transition residuals
    from :func:`petri_net_nn.traces.anomaly_score`, sorts by
    magnitude, and formats the top contributors as prose using the
    transitions' BPMN labels."""
    scores = anomaly_score(
        module, trace, attribute_to_marking=attribute_to_marking
    )
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    significant = [(t, r) for t, r in sorted_scores if r >= threshold]
    if not significant:
        return "No significant anomalies detected."

    fired_labels = {e.name for e in trace.events}
    net = module.net

    lines = [
        f"Trace flagged with {len(significant)} divergent "
        f"transition(s) (residual ≥ {threshold:.2f}):"
    ]
    for tid, residual in significant[:top_n]:
        label = net.transition_labels.get(tid, tid)
        observed = 1.0 if label in fired_labels else 0.0
        expected = (residual + observed) if observed == 0.0 else (observed - residual)
        if expected < 0:
            expected = max(0.0, observed - residual)
        elif expected > 1:
            expected = min(1.0, observed + residual)
        lines.append(
            f"  • {label!r}: expected ≈ {expected:.2f}, observed = "
            f"{observed:.0f}, residual = {residual:.3f}"
        )
    if len(significant) > top_n:
        lines.append(f"  • … {len(significant) - top_n} more below threshold")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 13 — bootstrap confidence intervals on extracted rules.
#
# Rule extraction reads a single point estimate out of the trained
# weights. That's only useful if the rule is stable: training a fresh
# module on a slightly-different trace sample should produce a similar
# rule. Bootstrap resampling is the standard statistical answer:
# resample the trace list with replacement N times, retrain a fresh
# module per resample, extract the rule each time, report the
# distribution of rule parameters. The percentile interval over the
# samples gives a confidence interval; the fraction of samples that
# agree with the point estimate's direction gives a stability score.
#
# Bootstrap is computationally heavy (N full training runs) but the
# resulting CI is what makes a rule trustworthy enough to ship to
# production. We default to N=100 — large enough for reasonable
# percentile estimates, small enough to fit a unit-test budget.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XORRuleCI:
    """An :class:`XORRule` annotated with bootstrap-derived confidence
    intervals.

    Attributes
    ----------
    rule
        The point-estimate rule — extracted from the module trained
        on the *full* trace list (no resampling). This is the rule
        you would report if you weren't computing CIs at all.
    n_bootstrap
        How many bootstrap resamples were trained.
    confidence
        The nominal coverage of the percentile interval below
        (e.g. 0.95 for a 95% CI). Used to compute the interval
        endpoints as the (1 − confidence)/2 and (1 + confidence)/2
        percentiles of the bootstrap samples.
    crossover_samples
        The crossover values extracted from each bootstrap-trained
        module. NaN samples (occur when the trained weight gap is
        too small to discriminate) are filtered out before computing
        the summary statistics — they're effectively "no rule" and
        shouldn't pull the CI around.
    crossover_mean / crossover_median
        Summary statistics over the (non-NaN) samples.
    crossover_ci_low / crossover_ci_high
        Percentile-based confidence interval bounds. ``nan`` if too
        few valid samples to compute.
    direction_agreement
        Fraction of bootstrap samples whose ``transition_above`` /
        ``transition_below`` direction matches the point estimate.
        1.0 means every resample agrees; values below ~0.8 are a
        red flag that the rule's direction is data-dependent.
    """

    rule: XORRule
    n_bootstrap: int
    confidence: float
    crossover_samples: tuple[float, ...]
    crossover_mean: float
    crossover_median: float
    crossover_ci_low: float
    crossover_ci_high: float
    direction_agreement: float

    def description(self) -> str:
        """One-line summary including the CI; useful for log lines
        and quick assertions."""
        return (
            f"{self.rule.input_place!r} > {self.rule.crossover:.3f} "
            f"(CI [{self.crossover_ci_low:.3f}, "
            f"{self.crossover_ci_high:.3f}], "
            f"direction agreement {self.direction_agreement:.0%}) → "
            f"{self.rule.label_above!r}"
        )


@dataclass(frozen=True)
class AndJoinRuleCI:
    """An :class:`AndJoinRule` annotated with bootstrap-derived
    confidence intervals on its threshold.

    Attributes
    ----------
    rule
        The point-estimate rule from training on the full trace list.
    n_bootstrap, confidence
        As for :class:`XORRuleCI`.
    threshold_samples
        Bootstrap-sample thresholds.
    threshold_mean / threshold_median / threshold_ci_low /
    threshold_ci_high
        Summary statistics over the samples.
    quorum_agreement
        Fraction of bootstrap samples whose extracted quorum
        ("all N", "≥ k of N", or "weighted") matches the point
        estimate's quorum gloss. A low value suggests the join's
        synchronisation rule is brittle under data resampling.
    """

    rule: AndJoinRule
    n_bootstrap: int
    confidence: float
    threshold_samples: tuple[float, ...]
    threshold_mean: float
    threshold_median: float
    threshold_ci_low: float
    threshold_ci_high: float
    quorum_agreement: float

    def description(self) -> str:
        return (
            f"{self.rule.label!r}: {self.rule.summary} "
            f"(threshold CI [{self.threshold_ci_low:.3f}, "
            f"{self.threshold_ci_high:.3f}], quorum agreement "
            f"{self.quorum_agreement:.0%})"
        )


def _percentile_ci(
    samples: list[float], confidence: float
) -> tuple[float, float]:
    """Compute the lower / upper percentile bounds of ``samples``
    matching the requested coverage. Returns ``(nan, nan)`` when
    too few samples are available — the caller decides how to
    surface "not computable" rather than us inventing a number."""
    if len(samples) < 2:
        return float("nan"), float("nan")
    sorted_samples = sorted(samples)
    # Type-2 percentile (linear interpolation between sample
    # quantiles) via statistics.quantiles. Asking for n=100 buckets
    # then indexing gives us a near-arbitrary percentile resolution.
    alpha = (1.0 - confidence) / 2.0
    n = len(sorted_samples)
    lo_idx = max(0, int(alpha * n))
    hi_idx = min(n - 1, int((1.0 - alpha) * n))
    return sorted_samples[lo_idx], sorted_samples[hi_idx]


def _bootstrap_indices(
    n: int, n_samples: int, rng: "torch.Generator | None"
) -> list[list[int]]:
    """Generate ``n_samples`` bootstrap index lists of length ``n``,
    each drawn with replacement from ``range(n)``. Pulling this out
    as a helper keeps the bootstrap callers symmetric and gives a
    single place to swap in a deterministic RNG."""
    # We use torch's RNG (passed in or default) so seeding behaves
    # the same way as the rest of training does.
    out: list[list[int]] = []
    for _ in range(n_samples):
        idx = torch.randint(
            0, n, (n,), generator=rng
        ).tolist()
        out.append(idx)
    return out


def bootstrap_xor_rule(
    module_factory: Callable[[], PetriNetModule],
    traces: list[XESTrace],
    *,
    attribute_to_marking: AttributeToMarking,
    input_place: str,
    transition_a: str,
    transition_b: str,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    steps: int = 500,
    lr: float = 0.1,
    attribute_to_values: AttributeToValues | None = None,
    seed: int | None = None,
) -> XORRuleCI:
    """Bootstrap-resample the trace list, train a fresh module per
    resample, extract the XOR rule, and report the distribution of
    crossovers as an :class:`XORRuleCI`.

    Parameters
    ----------
    module_factory
        A zero-argument callable that builds a fresh
        :class:`PetriNetModule` ready to train. The bootstrap
        algorithm calls it ``n_bootstrap + 1`` times — once for the
        point estimate on the full trace list, then once per
        resample. Each call MUST return an independent module: the
        factory exists to capture the net's structure while keeping
        the random initialisation fresh per call.
    traces
        The training trace list. Bootstrap samples are drawn with
        replacement from this list.
    attribute_to_marking
        Same role as in :func:`petri_net_nn.train_on_traces` — maps
        each trace to its input-marking dict.
    attribute_to_values
        Optional CPN value channel. Passed through to
        :func:`train_on_traces` for each bootstrap run.
    input_place, transition_a, transition_b
        The XOR pair to extract — the same arguments as
        :func:`extract_xor_rule`.
    n_bootstrap
        Number of bootstrap resamples. Default 100. Higher gives
        tighter CIs at higher training cost.
    confidence
        Coverage of the percentile interval (default 0.95).
    steps, lr
        Forwarded to the per-resample training. The defaults match
        ``train_on_traces``.
    seed
        Optional integer seed for the bootstrap RNG. The per-resample
        training is *not* re-seeded — initialisation variance across
        resamples is part of what bootstrap is measuring.

    Returns
    -------
    XORRuleCI
        Bundle of the point estimate, the bootstrap distribution,
        and the percentile-CI / direction-agreement stats.
    """
    # Point estimate first — train on the full trace list, extract
    # the rule. This is the "headline" rule the caller would report
    # without any CIs.
    point_module = module_factory()
    train_on_traces(
        point_module,
        traces,
        attribute_to_marking=attribute_to_marking,
        attribute_to_values=attribute_to_values,
        steps=steps,
        lr=lr,
    )
    point_rule = extract_xor_rule(
        point_module, input_place, transition_a, transition_b
    )

    # Bootstrap RNG — torch generator so behaviour matches the rest
    # of training. If no seed is supplied, the default global RNG
    # is used (non-deterministic across runs, which is fine for
    # diagnostic bootstrap stats).
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    samples: list[float] = []
    direction_matches = 0
    direction_total = 0
    for indices in _bootstrap_indices(len(traces), n_bootstrap, rng):
        resampled = [traces[i] for i in indices]
        module = module_factory()
        train_on_traces(
            module,
            resampled,
            attribute_to_marking=attribute_to_marking,
            attribute_to_values=attribute_to_values,
            steps=steps,
            lr=lr,
        )
        rule = extract_xor_rule(
            module, input_place, transition_a, transition_b
        )
        # The direction is the (above, below) ordering of the rule's
        # transitions. We compare against the point estimate to
        # measure stability across resamples.
        direction_total += 1
        if (rule.transition_above, rule.transition_below) == (
            point_rule.transition_above,
            point_rule.transition_below,
        ):
            direction_matches += 1
        # NaN crossovers come from weight-gap-too-small samples;
        # they're "no rule" and shouldn't pull the percentile CI.
        if not _is_nan(rule.crossover):
            samples.append(rule.crossover)

    direction_agreement = (
        direction_matches / direction_total if direction_total else 0.0
    )

    ci_low, ci_high = _percentile_ci(samples, confidence)
    mean = statistics.fmean(samples) if samples else float("nan")
    median = statistics.median(samples) if samples else float("nan")

    return XORRuleCI(
        rule=point_rule,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        crossover_samples=tuple(samples),
        crossover_mean=mean,
        crossover_median=median,
        crossover_ci_low=ci_low,
        crossover_ci_high=ci_high,
        direction_agreement=direction_agreement,
    )


def bootstrap_and_join_rule(
    module_factory: Callable[[], PetriNetModule],
    traces: list[XESTrace],
    *,
    attribute_to_marking: AttributeToMarking,
    transition: str,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    steps: int = 500,
    lr: float = 0.1,
    attribute_to_values: AttributeToValues | None = None,
    seed: int | None = None,
) -> AndJoinRuleCI:
    """Same shape as :func:`bootstrap_xor_rule` but for AND-join rules.

    Returns an :class:`AndJoinRuleCI` with the threshold distribution
    and a *quorum agreement* score — the fraction of bootstrap
    samples whose extracted quorum gloss matches the point estimate's.
    A drop in quorum agreement means the join's synchronisation
    behaviour is data-dependent: across plausible re-samples of the
    training data, sometimes it looks like an "all N" join and
    sometimes like a "≥ k of N" weighted vote.
    """
    point_module = module_factory()
    train_on_traces(
        point_module,
        traces,
        attribute_to_marking=attribute_to_marking,
        attribute_to_values=attribute_to_values,
        steps=steps,
        lr=lr,
    )
    point_rule = extract_and_join_rule(point_module, transition)

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    threshold_samples: list[float] = []
    quorum_matches = 0
    quorum_total = 0
    for indices in _bootstrap_indices(len(traces), n_bootstrap, rng):
        resampled = [traces[i] for i in indices]
        module = module_factory()
        train_on_traces(
            module,
            resampled,
            attribute_to_marking=attribute_to_marking,
            attribute_to_values=attribute_to_values,
            steps=steps,
            lr=lr,
        )
        rule = extract_and_join_rule(module, transition)
        quorum_total += 1
        if rule.summary == point_rule.summary:
            quorum_matches += 1
        if not _is_nan(rule.threshold):
            threshold_samples.append(rule.threshold)

    quorum_agreement = (
        quorum_matches / quorum_total if quorum_total else 0.0
    )

    ci_low, ci_high = _percentile_ci(threshold_samples, confidence)
    mean = (
        statistics.fmean(threshold_samples) if threshold_samples else float("nan")
    )
    median = (
        statistics.median(threshold_samples)
        if threshold_samples
        else float("nan")
    )

    return AndJoinRuleCI(
        rule=point_rule,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        threshold_samples=tuple(threshold_samples),
        threshold_mean=mean,
        threshold_median=median,
        threshold_ci_low=ci_low,
        threshold_ci_high=ci_high,
        quorum_agreement=quorum_agreement,
    )


def _is_nan(x: float) -> bool:
    """``math.isnan`` would work but introduces an import; the
    ``x != x`` trick is the canonical Pythonic NaN test."""
    return x != x


# ---------------------------------------------------------------------------
# Phase 13 — prose for rules.
#
# Both rule shapes (XOR and AND-join) already carry a one-line
# ``description()`` method. The functions below produce a longer
# paragraph-form rendering suitable for a report or a regulator-
# facing document — explanatory prose, not a debug log line.
# ---------------------------------------------------------------------------


def prose_for_xor_rule(
    rule: "XORRule | XORRuleCI",
    *,
    input_label: str | None = None,
) -> str:
    """Render an XOR rule (with or without bootstrap CI) as a
    paragraph in plain English.

    Parameters
    ----------
    rule
        Either a bare :class:`XORRule` (just the point estimate)
        or an :class:`XORRuleCI` (point estimate + confidence
        interval). When a CI is supplied, the prose includes both
        the percentile interval and the direction-agreement rate.
    input_label
        Optional human-readable name for the input quantity. The
        rule stores the *place id* (e.g. ``p_application``), which
        is fine in code but ugly in a regulator-facing paragraph.
        Supply ``input_label="application amount"`` to substitute
        a domain term.
    """
    if isinstance(rule, XORRuleCI):
        point = rule.rule
        ci_clause = (
            f" (95% confidence interval "
            f"[{rule.crossover_ci_low:.3f}, "
            f"{rule.crossover_ci_high:.3f}] over "
            f"{rule.n_bootstrap} bootstrap resamples)"
        )
        agreement_clause = (
            f" The direction of this routing rule was consistent "
            f"across {rule.direction_agreement:.0%} of the bootstrap "
            f"resamples."
        )
    else:
        point = rule
        ci_clause = ""
        agreement_clause = ""

    label = input_label or point.input_place
    return (
        f"When {label} is above {point.crossover:.3f}"
        f"{ci_clause}, the trained model routes to "
        f"{point.label_above!r} rather than {point.label_below!r}."
        f"{agreement_clause}"
    )


def prose_for_and_join_rule(
    rule: "AndJoinRule | AndJoinRuleCI",
) -> str:
    """Render an AND-join rule (with or without bootstrap CI) as a
    paragraph in plain English. The point-estimate rule's
    ``summary`` field already carries a gloss like *"all 3 inputs"*
    or *"at least 2 of 3 inputs"* or a weighted-vote description;
    we wrap that in narrative prose plus any CI annotations."""
    if isinstance(rule, AndJoinRuleCI):
        point = rule.rule
        ci_clause = (
            f" The threshold's 95% confidence interval is "
            f"[{rule.threshold_ci_low:.3f}, "
            f"{rule.threshold_ci_high:.3f}] over "
            f"{rule.n_bootstrap} bootstrap resamples."
        )
        agreement_clause = (
            f" The quorum shape was consistent across "
            f"{rule.quorum_agreement:.0%} of resamples."
        )
    else:
        point = rule
        ci_clause = ""
        agreement_clause = ""

    return (
        f"The synchronisation step {point.label!r} fires when "
        f"{point.summary}.{ci_clause}{agreement_clause}"
    )


# ---------------------------------------------------------------------------
# Phase 13 — counterfactual explanations.
#
# "The model declined this loan; what minimum change to the
# application amount would have got it approved?" — a counterfactual
# question is the one a regulator or a customer-facing case worker
# actually wants answered. It's the "actionable why" complement to
# the anomaly score's "where did things diverge."
#
# Algorithm: binary search the input value at a chosen place
# (along either the marking channel or the per-token value
# channel) until the target transition's firing activation crosses
# the firing threshold (0.5). The crossing point is the
# counterfactual — the smallest change to that input that would
# have flipped the outcome.
#
# Scope notes:
#   * We vary one input at a time. Multi-dimensional counterfactuals
#     (find the minimum joint change to multiple inputs that flips
#     the outcome) need a non-trivial optimisation; this is a
#     natural follow-up.
#   * We assume the target transition's activation is monotonic in
#     the flipped input — the standard sigmoidal-routing case. If
#     the activation isn't monotonic (e.g. a soft-guard with a
#     non-monotone torch_guard), the binary search will find some
#     crossing point but not necessarily the closest one to the
#     base value. A monotonicity check is left as a follow-up.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Counterfactual:
    """A minimal-change explanation: what input value at one place
    would have flipped a target transition's firing decision.

    Attributes
    ----------
    target_transition
        The transition whose firing is being explained.
    target_label
        Human label for ``target_transition``.
    flipped_place
        The input place whose value was varied to flip the outcome.
    flipped_channel
        Either ``"marking"`` (the place activation) or ``"value"``
        (the coloured-token value channel — used for CPN routing
        where the structural guard reads per-token values).
    original_input
        The value of ``flipped_place`` in the base scenario.
    counterfactual_input
        The value at which ``target_transition``'s firing
        activation crosses the 0.5 threshold.
    original_activation
        Firing activation of ``target_transition`` at
        ``original_input``.
    counterfactual_activation
        Firing activation of ``target_transition`` at
        ``counterfactual_input``. Should be near 0.5 by
        construction.
    """

    target_transition: str
    target_label: str
    flipped_place: str
    flipped_channel: str
    original_input: float
    counterfactual_input: float
    original_activation: float
    counterfactual_activation: float

    def description(self) -> str:
        """One-line description suitable for log lines."""
        return (
            f"flip {self.target_label!r}: change {self.flipped_place!r} "
            f"({self.flipped_channel}) from {self.original_input:.3f} "
            f"to {self.counterfactual_input:.3f}"
        )


def find_counterfactual(
    module: PetriNetModule,
    base_marking: dict[str, float],
    *,
    flip_place: str,
    target_transition: str,
    base_values: dict[str, float] | None = None,
    flip_channel: str = "marking",
    search_range: tuple[float, float] = (0.0, 1.0),
    tolerance: float = 1e-3,
    interval_tolerance: float | None = None,
    max_iterations: int = 50,
) -> Counterfactual | None:
    """Binary-search the input at ``flip_place`` to find the
    crossing point where ``target_transition``'s firing activation
    flips across 0.5.

    Parameters
    ----------
    module
        The trained module to query. Its forward pass is what
        drives the search.
    base_marking
        The starting input marking — the original scenario whose
        outcome we want to flip. Used to compute the *original*
        firing activation at ``target_transition``.
    flip_place
        The place whose input we vary.
    target_transition
        The transition whose firing we want to flip. The
        counterfactual is the smallest change to ``flip_place``'s
        value that makes ``target_transition``'s activation cross
        0.5.
    base_values
        Optional CPN value channel for ``module(input_values=...)``.
        Required when ``flip_channel="value"`` since the search
        varies an entry of this dict; ignored when
        ``flip_channel="marking"``.
    flip_channel
        ``"marking"`` (default) varies the activation at
        ``flip_place``; ``"value"`` varies the per-token value
        the compiled guards read. Use ``"value"`` for
        coloured-Petri-net scenarios like ``credit_approval_coloured``
        where the relevant decision quantity is the token value,
        not the activation.
    search_range
        Lower and upper bound for the binary search. For marking
        channels, the usual [0, 1]; for value channels, the
        domain of the relevant attribute (e.g. (0, 10000) for
        loan amounts).
    tolerance
        Activation tolerance: stop when the midpoint's firing
        activation is within this distance of 0.5. Default
        ``1e-3``, which puts the counterfactual well inside the
        decision boundary.
    interval_tolerance
        Search-interval tolerance: stop when the bracketing
        interval shrinks below this width. Defaults to
        ``(search_range[1] - search_range[0]) * 1e-4`` — i.e.
        four decimal digits of resolution on the input scale.
        Set explicitly for non-default search ranges if you want
        a different floor (e.g. ``1.0`` for monetary amounts where
        finer than £1 doesn't matter).
    max_iterations
        Hard cap on bisection iterations — protects against
        non-monotone activations that won't converge.

    Returns
    -------
    Counterfactual | None
        ``None`` when no crossing exists in ``search_range`` —
        the activation stays on the same side of 0.5 at both
        endpoints, meaning a counterfactual within those bounds
        doesn't exist. Otherwise the ``Counterfactual`` pinning
        the original-input, counterfactual-input, and activation
        pair.
    """
    if flip_channel not in ("marking", "value"):
        raise ValueError(
            f"flip_channel must be 'marking' or 'value', got {flip_channel!r}"
        )
    if flip_channel == "value" and base_values is None:
        raise ValueError(
            "flip_channel='value' requires base_values to be supplied — "
            "the counterfactual search varies an entry of that dict"
        )

    # Pre-compute the original activation by running the module
    # on the base inputs. This is what the counterfactual is
    # "flipping away from."
    original_input = (
        base_values[flip_place]
        if flip_channel == "value"
        else base_marking.get(flip_place, 0.0)
    )
    original_activation = _activation(
        module, base_marking, base_values, target_transition
    )

    # Probe the two endpoints. If both produce activations on the
    # same side of 0.5, no crossing exists in the range — bail.
    lo, hi = search_range
    lo_act = _activation_with_override(
        module, base_marking, base_values,
        flip_place, flip_channel, lo, target_transition,
    )
    hi_act = _activation_with_override(
        module, base_marking, base_values,
        flip_place, flip_channel, hi, target_transition,
    )
    if (lo_act - 0.5) * (hi_act - 0.5) > 0:
        # Same side — no zero crossing of (activation − 0.5) in
        # the search range.
        return None

    # Default the interval tolerance to a fixed fraction of the
    # search range — works equally well for [0, 1] marking
    # searches and [0, 10000] value searches without the caller
    # needing to think about it.
    if interval_tolerance is None:
        interval_tolerance = (hi - lo) * 1e-4

    # Bisect: maintain (lo, hi) bracketing the 0.5 crossing.
    counterfactual_input = 0.5 * (lo + hi)
    counterfactual_activation = 0.5
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        mid_act = _activation_with_override(
            module, base_marking, base_values,
            flip_place, flip_channel, mid, target_transition,
        )
        counterfactual_input = mid
        counterfactual_activation = mid_act
        if abs(mid_act - 0.5) < tolerance:
            break
        if (hi - lo) < interval_tolerance:
            break
        # Keep the side where the activation crosses 0.5. Sign of
        # (lo_act − 0.5) tells us which bracket retains the
        # opposite-sign endpoint.
        if (mid_act - 0.5) * (lo_act - 0.5) < 0:
            hi, hi_act = mid, mid_act
        else:
            lo, lo_act = mid, mid_act

    return Counterfactual(
        target_transition=target_transition,
        target_label=module.net.transition_labels.get(
            target_transition, target_transition
        ),
        flipped_place=flip_place,
        flipped_channel=flip_channel,
        original_input=float(original_input),
        counterfactual_input=float(counterfactual_input),
        original_activation=float(original_activation),
        counterfactual_activation=float(counterfactual_activation),
    )


def _activation(
    module: PetriNetModule,
    base_marking: dict[str, float],
    base_values: dict[str, float] | None,
    transition: str,
) -> float:
    """Helper: run the module on the supplied inputs and return
    the firing activation of ``transition`` as a Python float.
    Wraps the input dicts into the batched torch tensors the
    forward pass expects."""
    marking_tensor = {
        p: torch.tensor([v]) for p, v in base_marking.items()
    }
    values_tensor = (
        {p: torch.tensor([v]) for p, v in base_values.items()}
        if base_values is not None
        else None
    )
    module.eval()
    with torch.no_grad():
        out = module(
            input_marking=marking_tensor,
            input_values=values_tensor,
            batch_size=1,
        )
    return float(out[transition][0].item())


def _activation_with_override(
    module: PetriNetModule,
    base_marking: dict[str, float],
    base_values: dict[str, float] | None,
    flip_place: str,
    flip_channel: str,
    new_value: float,
    transition: str,
) -> float:
    """Helper: like :func:`_activation` but substitutes
    ``new_value`` for ``flip_place``'s entry in the appropriate
    channel before the forward pass. Used by the binary search
    inner loop."""
    if flip_channel == "marking":
        new_marking = dict(base_marking)
        new_marking[flip_place] = new_value
        new_values = base_values
    else:
        new_marking = base_marking
        new_values = dict(base_values or {})
        new_values[flip_place] = new_value
    return _activation(module, new_marking, new_values, transition)


def prose_for_counterfactual(
    cf: Counterfactual,
    *,
    input_label: str | None = None,
) -> str:
    """Render a counterfactual as a paragraph in plain English.

    Parameters
    ----------
    cf
        The :class:`Counterfactual` to describe.
    input_label
        Optional human-readable name for the flipped input.
        Useful when the place id is something like
        ``p_application`` but the regulator-facing prose wants
        *"application amount"*.
    """
    label = input_label or cf.flipped_place
    direction = (
        "increased" if cf.counterfactual_input > cf.original_input
        else "decreased"
    )
    activation_change = (
        "would have started firing"
        if cf.counterfactual_activation > cf.original_activation
        else "would have stopped firing"
    )
    return (
        f"To flip the outcome at {cf.target_label!r}, the input "
        f"{label} would need to be {direction} from "
        f"{cf.original_input:.3f} to {cf.counterfactual_input:.3f}. "
        f"At that point the step {activation_change} (activation "
        f"changes from {cf.original_activation:.3f} to "
        f"{cf.counterfactual_activation:.3f})."
    )


# ---------------------------------------------------------------------------
# Phase 13 — sensitivity analysis.
#
# Sensitivity answers "which input is the model leaning on hardest?"
# at the level of named places. Two scopes:
#
#   * Local sensitivity at one base point — partial derivatives of a
#     target transition's firing activation with respect to each
#     input. Large magnitude = the firing decision pivots on that
#     input locally; near-zero magnitude = irrelevant at this point.
#     Useful for "which input should I look at before reaching for a
#     counterfactual?"
#
#   * Aggregate input importance across a trace set — sum the
#     absolute gradients across many traces and many scored
#     transitions, then normalise. Tells you which inputs the
#     trained network leans on for its overall decision-making, not
#     just at a single instance.
#
# Both rely on torch autograd. We set ``requires_grad=True`` on the
# input tensors, run a forward pass, call ``.backward()`` on the
# target activation, and read the gradients off the inputs.
#
# Caveat: gradients are *local* properties of the trained function.
# At saturation (activations very near 0 or 1), the sigmoid's gradient
# is small even for the "right" input — the model has already
# decided. For meaningful sensitivity readings, evaluate near a
# decision boundary, not deep in a saturated regime.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityReport:
    """Local sensitivity of one transition's firing activation to
    each input at a given base point.

    Attributes
    ----------
    target_transition
        The transition whose firing we measured.
    target_label
        Human label for ``target_transition``.
    base_activation
        Firing activation of ``target_transition`` at the base
        inputs — useful for sanity-checking that we're near a
        decision boundary (gradients far from 0.5 are less
        informative).
    marking_gradients
        Per-place gradient of the target's activation with respect
        to the marking channel at that place. Larger magnitude
        means the firing decision pivots more on this input.
    value_gradients
        Per-place gradient on the coloured-token value channel.
        Empty when ``base_values`` wasn't supplied.
    """

    target_transition: str
    target_label: str
    base_activation: float
    marking_gradients: dict[str, float] = field(default_factory=dict)
    value_gradients: dict[str, float] = field(default_factory=dict)

    def ranked(self, *, top_n: int | None = None) -> list[tuple[str, str, float]]:
        """Inputs sorted by descending absolute gradient. Each entry
        is ``(place, channel, gradient)`` where ``channel`` is
        ``"marking"`` or ``"value"`` — so a place can appear twice
        if it's driven through both channels (rare). When
        ``top_n`` is given, returns only the top N."""
        entries: list[tuple[str, str, float]] = []
        for place, grad in self.marking_gradients.items():
            entries.append((place, "marking", grad))
        for place, grad in self.value_gradients.items():
            entries.append((place, "value", grad))
        entries.sort(key=lambda e: abs(e[2]), reverse=True)
        return entries[:top_n] if top_n is not None else entries


def transition_sensitivity(
    module: PetriNetModule,
    base_marking: dict[str, float],
    target_transition: str,
    *,
    base_values: dict[str, float] | None = None,
) -> SensitivityReport:
    """Compute per-input gradient magnitudes of
    ``target_transition``'s firing activation at the base point.

    The gradient of the activation with respect to each input
    captures the local sensitivity — how much the firing decision
    changes when we nudge that input. Larger gradient magnitude
    means the model is leaning more on that input *at this base
    point*. Inputs with near-zero gradient are locally irrelevant
    (either the model genuinely doesn't use them, or we're in a
    saturated regime far from any decision boundary).

    Parameters
    ----------
    module
        The trained module whose sensitivities to compute.
    base_marking
        Input marking at which to evaluate gradients. Typically
        the original case in a counterfactual investigation, or
        a representative instance from the training distribution.
    target_transition
        The transition whose firing is being explained.
    base_values
        Optional coloured-token value channel. When supplied, the
        report also contains value-channel gradients — useful for
        CPN scenarios where the routing decision pivots on the
        per-token value rather than the place activation.
    """
    marking_tensors = {
        p: torch.tensor([float(v)], requires_grad=True)
        for p, v in base_marking.items()
    }
    values_tensors: dict[str, torch.Tensor] | None = None
    if base_values is not None:
        values_tensors = {
            p: torch.tensor([float(v)], requires_grad=True)
            for p, v in base_values.items()
        }

    # Forward pass — we *don't* wrap in torch.no_grad here because
    # we need the autograd graph. ``module.eval()`` disables any
    # training-time stochastic behaviour (dropout etc.) while
    # leaving the graph in place.
    module.eval()
    out = module(
        input_marking=marking_tensors,
        input_values=values_tensors,
        batch_size=1,
    )

    target_activation = out[target_transition].squeeze()
    base_activation = float(target_activation.detach().item())

    # Single backward pass — gradients populate the .grad attribute
    # of every requires_grad=True input tensor.
    target_activation.backward()

    marking_gradients = {
        p: float(t.grad.item()) if t.grad is not None else 0.0
        for p, t in marking_tensors.items()
    }
    value_gradients: dict[str, float] = {}
    if values_tensors is not None:
        value_gradients = {
            p: float(t.grad.item()) if t.grad is not None else 0.0
            for p, t in values_tensors.items()
        }

    return SensitivityReport(
        target_transition=target_transition,
        target_label=module.net.transition_labels.get(
            target_transition, target_transition
        ),
        base_activation=base_activation,
        marking_gradients=marking_gradients,
        value_gradients=value_gradients,
    )


def input_importance(
    module: PetriNetModule,
    traces: list[XESTrace],
    *,
    attribute_to_marking: AttributeToMarking,
    transitions: list[str] | None = None,
    attribute_to_values: AttributeToValues | None = None,
) -> dict[str, float]:
    """Aggregate per-input importance across a trace set.

    For each trace and each scored transition, computes the
    gradient of that transition's firing activation with respect
    to each input. Aggregates by *summing the absolute gradients*
    across traces and transitions — places with a large summed
    gradient are the ones the trained network leans on most
    heavily for its overall decision-making.

    The aggregate is reported separately for the marking channel
    and the value channel: a key like ``"marking:p_request"`` or
    ``"value:p_submitted"`` keeps the two channels distinguishable
    in scenarios where both matter.

    Parameters
    ----------
    module
        The trained module.
    traces
        The trace set across which to aggregate. Importance is
        averaged-by-summation, so a larger trace set gives more
        stable scores.
    attribute_to_marking
        Maps each trace to its input marking dict.
    transitions
        Which transitions' activations to score against. Defaults
        to "all transitions whose label doesn't look auto-generated"
        — matches the convention used by ``train_on_traces`` and
        ``anomaly_score``.
    attribute_to_values
        Optional value channel — same role as in
        ``train_on_traces``.

    Returns
    -------
    dict[str, float]
        Importance scores keyed by ``"<channel>:<place_id>"``.
        Sorted-by-descending-value through ``sorted(..., reverse=True)``
        on ``.items()`` gives the ranked importance list.
    """
    net = module.net
    if transitions is None:
        # Same default as train_on_traces / anomaly_score — skip
        # auto-generated gateway transitions.
        transitions = sorted(
            t for t in net.transitions
            if "->" not in net.transition_labels.get(t, t)
        )

    importance: dict[str, float] = {}

    for trace in traces:
        base_marking = attribute_to_marking(trace)
        base_values = (
            attribute_to_values(trace)
            if attribute_to_values is not None
            else None
        )
        for t in transitions:
            report = transition_sensitivity(
                module, base_marking, t, base_values=base_values,
            )
            for place, grad in report.marking_gradients.items():
                key = f"marking:{place}"
                importance[key] = importance.get(key, 0.0) + abs(grad)
            for place, grad in report.value_gradients.items():
                key = f"value:{place}"
                importance[key] = importance.get(key, 0.0) + abs(grad)

    return importance


def prose_for_sensitivity(
    report: SensitivityReport,
    *,
    top_n: int = 3,
    input_labels: dict[str, str] | None = None,
) -> str:
    """Render a sensitivity report as a paragraph in plain English.

    Lists the top-N inputs ranked by absolute gradient magnitude,
    with directional language ("increasing this input raises /
    lowers the firing of …") so the reader doesn't have to map a
    signed number to behaviour themselves.

    Parameters
    ----------
    report
        The :class:`SensitivityReport` to describe.
    top_n
        How many inputs to enumerate (default 3). The full ranking
        is available via ``report.ranked()``; the prose just picks
        the leading items.
    input_labels
        Optional ``{place_id: human_label}`` mapping for
        substituting domain terms ("application amount" rather
        than "p_application"). Each key is a place id (without
        the channel prefix); both channels of the same place will
        be relabelled together.
    """
    labels = input_labels or {}
    ranking = report.ranked(top_n=top_n)
    if not ranking:
        return (
            f"No measurable sensitivity at the base point for "
            f"{report.target_label!r}."
        )

    lines = [
        f"At the base point, the firing of {report.target_label!r} "
        f"(activation {report.base_activation:.3f}) is most "
        f"sensitive to:"
    ]
    for i, (place, channel, grad) in enumerate(ranking, start=1):
        label = labels.get(place, place)
        direction = "raises" if grad > 0 else "lowers"
        if grad == 0:
            direction = "leaves unchanged"
        channel_note = (
            "" if channel == "marking" else " (per-token value)"
        )
        lines.append(
            f"  {i}. {label}{channel_note} — gradient "
            f"{grad:+.3f}; increasing it {direction} the firing."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 13 — cross-variant comparison reports.
#
# When two trained variants are *bisimilar* (Phase 2 / Phase 11),
# they're guaranteed to exhibit the same observable behaviour from
# their initial markings. But that's a structural guarantee — it
# doesn't tell you how often the two variants' learned routing
# *decisions* agree across a given operating range. Two bisimilar
# nets, trained on different data, will still differ in their soft
# activations at intermediate input values.
#
# Cross-variant comparison answers the practical question:
#   "Across this input range, on what fraction of points do these
#    two variants reach the same firing decision at each
#    corresponding transition? Where do they disagree?"
#
# Algorithm: take a Cartesian-product grid over the user-supplied
# input ranges, evaluate both modules at every grid point, and
# tally per-transition agreement. Transitions are corresponded by
# label by default — variants of the same process usually share
# transition labels, only the IDs differ. An explicit
# ``correspondence`` mapping overrides this when labels don't line
# up.
#
# Two agreement notions, both reported:
#   * "Hard" — both activations on the same side of 0.5.
#   * "Soft" — both activations within ``tolerance`` of each other.
#
# This pairs naturally with the cost-ranked refactoring scenario:
# bisimulation proves two variants are *behaviourally* equivalent;
# cost-ranking shows which is cheaper to run; cross-variant
# comparison shows *where in the input domain* they actually
# overlap and where their soft routing differs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisagreementSample:
    """One grid point where the two variants disagree on at least
    one corresponded transition.

    Attributes
    ----------
    input_marking
        Place activations at this sample point.
    input_values
        Per-token value channel at this sample point, or ``None``
        when the comparison didn't use a value grid.
    diverging
        Map of label → (variant_a_activation, variant_b_activation)
        for transitions where the two variants disagreed at this
        point. Only diverging transitions appear here — the report
        is a *disagreement* sample, not a full state dump.
    """

    input_marking: dict[str, float]
    input_values: dict[str, float] | None
    diverging: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class ComparisonReport:
    """Outcome of comparing two trained variants over a grid of
    input points.

    Attributes
    ----------
    n_samples
        How many grid points were evaluated.
    paired_labels
        Transition labels that were matched between the two
        variants. Variants of the same process usually share
        labels; this list pins which ones the comparison actually
        ran on. Labels present in only one variant are skipped
        (and recorded in ``unmatched_a`` / ``unmatched_b`` for
        diagnostic purposes).
    unmatched_a, unmatched_b
        Labels present in variant A / B but not in the other.
        Useful for catching variants that differ structurally
        before reading the agreement numbers.
    hard_agreement_rate
        Fraction of (sample, transition) pairs where both variants'
        firing activations fell on the same side of 0.5 — the
        same firing *decision*.
    soft_agreement_rate
        Fraction where the two activations were within
        ``tolerance`` of each other. Stricter when tolerance is
        small, looser when large.
    per_transition_agreement
        Per-label hard-agreement rate. Sorted-descending lists
        which transitions agree most reliably; sorted-ascending
        surfaces the ones most prone to divergence.
    disagreement_samples
        Specific grid points where at least one transition
        diverged (under the hard-agreement definition). Capped at
        ``max_disagreement_samples`` to keep the report a
        bounded-size object.
    tolerance
        Echo of the soft-agreement tolerance used. Recorded so
        the report is self-describing.
    """

    n_samples: int
    paired_labels: tuple[str, ...]
    unmatched_a: tuple[str, ...]
    unmatched_b: tuple[str, ...]
    hard_agreement_rate: float
    soft_agreement_rate: float
    per_transition_agreement: dict[str, float]
    disagreement_samples: tuple[DisagreementSample, ...]
    tolerance: float


def _scored_label_map(module: PetriNetModule) -> dict[str, str]:
    """Map of label → transition_id for the scoreable transitions
    of a module. Skips auto-generated gateway labels (those that
    contain ``->``) — the same convention used by
    ``train_on_traces`` and the rule extractors."""
    return {
        module.net.transition_labels.get(t, t): t
        for t in module.net.transitions
        if "->" not in module.net.transition_labels.get(t, t)
    }


def _grid_points(
    input_grid: dict[str, list[float]],
) -> list[dict[str, float]]:
    """Cartesian product over the supplied input ranges. Returns a
    list of marking dicts, one per grid point.

    Empty / missing grids return ``[{}]`` — the single empty
    marking, meaning "no input override." The caller can layer
    this with a value grid through the symmetric helper."""
    if not input_grid:
        return [{}]
    places = sorted(input_grid)
    value_lists = [input_grid[p] for p in places]
    points: list[dict[str, float]] = []
    for combo in itertools.product(*value_lists):
        points.append(dict(zip(places, combo)))
    return points


def compare_variants(
    module_a: PetriNetModule,
    module_b: PetriNetModule,
    *,
    input_grid: dict[str, list[float]] | None = None,
    input_value_grid: dict[str, list[float]] | None = None,
    correspondence: dict[str, tuple[str, str]] | None = None,
    tolerance: float = 0.1,
    max_disagreement_samples: int = 50,
) -> ComparisonReport:
    """Compare two trained variants across a grid of input points.

    For each input combination in the Cartesian product of
    ``input_grid`` (marking channel) and ``input_value_grid``
    (coloured-token value channel), evaluate both modules and
    compare each pair of corresponded transitions' firing
    activations.

    Parameters
    ----------
    module_a, module_b
        The two trained variants to compare. Usually two trainings
        of structurally-bisimilar nets — see ``are_bisimilar`` /
        ``are_weakly_bisimilar``. The function itself doesn't
        check bisimilarity; it compares whatever you pass in.
    input_grid
        Place activations to vary. Keys are place ids that exist
        in *both* modules (when in doubt, pass the same place id
        appearing as a source in both — variants of the same
        process share input names). Values are lists of activation
        values to evaluate. The Cartesian product gives the
        sample set.
    input_value_grid
        Optional coloured-token value channel grid. Same shape as
        ``input_grid``. When supplied, each combined grid point
        runs through both the marking and value channels of both
        modules.
    correspondence
        Override the default label-matching. Keys are an arbitrary
        identifier the user chooses (often the human label), values
        are ``(transition_id_in_a, transition_id_in_b)`` pairs.
        Use this when the two variants use different labels for
        the same conceptual step (e.g. *"Approve loan"* vs
        *"Loan approval"*).
    tolerance
        Soft-agreement tolerance: two activations are softly
        equal when ``|a - b| <= tolerance``. Default 0.1.
    max_disagreement_samples
        Hard cap on the number of disagreement samples carried in
        the report — protects against a single grid sweep
        producing thousands of divergent points and bloating the
        report.

    Returns
    -------
    ComparisonReport
        Bundled outcome: agreement rates, per-transition rates,
        and up to ``max_disagreement_samples`` divergent grid
        points for inspection.
    """
    # Build the correspondence map. The user's explicit
    # ``correspondence`` takes precedence; whatever's left is
    # filled in by label-matching.
    pairs: dict[str, tuple[str, str]] = {}
    if correspondence is not None:
        pairs.update(correspondence)
    a_labels = _scored_label_map(module_a)
    b_labels = _scored_label_map(module_b)
    for label in a_labels:
        if label in pairs:
            continue
        if label in b_labels:
            pairs[label] = (a_labels[label], b_labels[label])

    unmatched_a = sorted(label for label in a_labels if label not in pairs)
    unmatched_b = sorted(label for label in b_labels if label not in pairs)

    # Build the joint grid: cross marking grid × value grid.
    marking_points = _grid_points(input_grid or {})
    value_points = _grid_points(input_value_grid or {})

    n_samples = 0
    n_hard_agree = 0
    n_hard_total = 0
    n_soft_agree = 0
    per_transition_hits: dict[str, int] = {label: 0 for label in pairs}
    per_transition_totals: dict[str, int] = {label: 0 for label in pairs}
    disagreement_samples: list[DisagreementSample] = []

    module_a.eval()
    module_b.eval()
    with torch.no_grad():
        for marking, values in itertools.product(marking_points, value_points):
            n_samples += 1
            # Build the tensor inputs once per grid point and pass
            # them to both modules.
            marking_tensor = {
                p: torch.tensor([v]) for p, v in marking.items()
            }
            values_tensor = (
                {p: torch.tensor([v]) for p, v in values.items()}
                if values
                else None
            )
            out_a = module_a(
                input_marking=marking_tensor,
                input_values=values_tensor,
                batch_size=1,
            )
            out_b = module_b(
                input_marking=marking_tensor,
                input_values=values_tensor,
                batch_size=1,
            )

            diverging: dict[str, tuple[float, float]] = {}
            for label, (tid_a, tid_b) in pairs.items():
                act_a = float(out_a[tid_a].item())
                act_b = float(out_b[tid_b].item())
                # Hard agreement = same side of 0.5.
                hard = (act_a >= 0.5) == (act_b >= 0.5)
                # Soft agreement = within tolerance.
                soft = abs(act_a - act_b) <= tolerance
                per_transition_totals[label] += 1
                if hard:
                    per_transition_hits[label] += 1
                    n_hard_agree += 1
                else:
                    diverging[label] = (act_a, act_b)
                n_hard_total += 1
                if soft:
                    n_soft_agree += 1

            if diverging and len(disagreement_samples) < max_disagreement_samples:
                disagreement_samples.append(
                    DisagreementSample(
                        input_marking=dict(marking),
                        input_values=dict(values) if values else None,
                        diverging=diverging,
                    )
                )

    hard_rate = n_hard_agree / n_hard_total if n_hard_total else 1.0
    soft_rate = n_soft_agree / n_hard_total if n_hard_total else 1.0
    per_transition = {
        label: (
            per_transition_hits[label] / per_transition_totals[label]
            if per_transition_totals[label]
            else 1.0
        )
        for label in pairs
    }

    return ComparisonReport(
        n_samples=n_samples,
        paired_labels=tuple(sorted(pairs)),
        unmatched_a=tuple(unmatched_a),
        unmatched_b=tuple(unmatched_b),
        hard_agreement_rate=hard_rate,
        soft_agreement_rate=soft_rate,
        per_transition_agreement=per_transition,
        disagreement_samples=tuple(disagreement_samples),
        tolerance=tolerance,
    )


def prose_for_comparison_report(
    report: ComparisonReport,
    *,
    top_disagreement: int = 3,
) -> str:
    """Render a cross-variant comparison report as a paragraph.

    Reports overall hard- and soft-agreement rates, names the
    transitions most prone to disagreement, and flags any
    label mismatches between the two variants. Suitable for
    inclusion in a refactoring proposal or a model-risk review.

    Parameters
    ----------
    report
        The :class:`ComparisonReport` to describe.
    top_disagreement
        How many of the most-divergent transitions to list (default
        3). Surfaces the worst offenders without burying the reader
        in a long table; the full per-transition breakdown is
        always available on ``report.per_transition_agreement``.
    """
    lines = [
        f"Variants compared across {report.n_samples} input "
        f"point(s); {len(report.paired_labels)} transitions "
        f"corresponded by label.",
        f"Hard agreement (same firing decision): "
        f"{report.hard_agreement_rate:.1%}.",
        f"Soft agreement (activations within "
        f"{report.tolerance:.2f}): {report.soft_agreement_rate:.1%}.",
    ]
    # Disagreement ranking: sort transitions by ascending agreement
    # rate, so the ones most prone to diverge surface first.
    sorted_transitions = sorted(
        report.per_transition_agreement.items(),
        key=lambda kv: kv[1],
    )
    diverging = [
        (label, rate)
        for label, rate in sorted_transitions
        if rate < 1.0
    ]
    if diverging:
        lines.append(
            f"Transitions most prone to disagreement "
            f"(top {min(top_disagreement, len(diverging))}):"
        )
        for label, rate in diverging[:top_disagreement]:
            lines.append(
                f"  • {label!r}: agree on {rate:.1%} of points."
            )
    else:
        lines.append(
            "Every paired transition agreed on every sample — the "
            "two variants are functionally indistinguishable across "
            "this input domain."
        )
    if report.unmatched_a:
        lines.append(
            f"Labels in variant A with no match in variant B: "
            f"{list(report.unmatched_a)}."
        )
    if report.unmatched_b:
        lines.append(
            f"Labels in variant B with no match in variant A: "
            f"{list(report.unmatched_b)}."
        )
    return "\n".join(lines)
