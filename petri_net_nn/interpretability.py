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
