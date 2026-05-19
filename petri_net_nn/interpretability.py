"""Distil learned weights back into readable decision rules.

Addresses §8's fourth open problem from the architecture spec:

  > The structural interpretability of the architecture — which subnet
  > shows the anomaly — is a genuine advantage over black-box ML. But
  > the learned weights within each subnet are not directly
  > interpretable. … The next research question is whether the learned
  > weights can be distilled back into readable decision rules —
  > closing the loop from neural learning to interpretable business
  > logic.

Two distillation routines:

  * ``extract_routing_rules`` walks the compiled module's structure
    for XOR-shape transitions (N transitions sharing one input place,
    no other consumers) and reads the learned weights / thresholds
    out as a single crossover value per pair: "input above X →
    transition A, otherwise → transition B". This is the most common
    decision pattern in a BPMN process and the one closest to a
    classical business rule.

  * ``explain_anomaly`` takes the residual dict from
    :func:`petri_net_nn.traces.anomaly_score` and the trace under
    test, and returns a human-readable narrative pinned to the
    diverging transitions by their BPMN labels.

The scaffold restricts XOR rule extraction to two-way splits — the
common case. N-way splits compute pairwise crossovers between the top
two transitions but flag the rule as approximate; full N-way rule
extraction is a follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass

from petri_net_nn.compiler import PetriNetModule
from petri_net_nn.petri_net import PetriNet
from petri_net_nn.traces import AttributeToMarking, anomaly_score
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
