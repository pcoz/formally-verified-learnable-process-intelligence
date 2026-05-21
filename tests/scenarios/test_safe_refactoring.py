"""End-to-end test for the safe-refactoring scenario.

Two variants of a loan-approval process:

* **Variant A** — the baseline. Plain sequence
  ``submitted -> triage -> triaged -> assess -> assessed
    -> (approve | decline)``. Loaded from `scenario.toml`.

* **Variant B** — Variant A with a silent (τ) audit-log
  transition inserted between ``p_triaged`` and ``t_assess_risk``.
  Constructed in this test from Variant A's structure.

The test pins four claims:

1. ``are_bisimilar`` rejects the two as different — the τ step
   is an observable difference at the LTS level under strong
   bisimulation.
2. ``are_weakly_bisimilar`` accepts the two as equivalent — the
   τ step collapses under weak bisimulation, restoring the
   external behaviour match.
3. After training on the same trace data, ``compare_variants``
   reports per-transition activations within tolerance across
   the credit-score domain — the trained variants soft-route
   identically.
4. ``bootstrap_xor_rule`` on each variant produces crossover
   confidence intervals that overlap and bracket the empirical
   decision band — the distilled rule survives bootstrap
   resampling under both shapes.
"""
from __future__ import annotations

from pathlib import Path

import torch

from petri_net_nn import (
    PetriNet,
    PetriNetModule,
    are_bisimilar,
    are_weakly_bisimilar,
    bootstrap_xor_rule,
    compare_variants,
    extract_xor_rule,
    load_scenario,
    train_on_traces,
)


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "safe_refactoring"
    / "scenario.toml"
)


# ---------------------------------------------------------------------------
# Helpers — construct Variant B from Variant A's topology.
# ---------------------------------------------------------------------------


def _build_variant_b(variant_a: PetriNet) -> PetriNet:
    """Return Variant B — Variant A with a silent audit-log step
    inserted between ``p_triaged`` and ``t_assess_risk``.

    Concretely, Variant A's direct ``p_triaged -> t_assess_risk``
    arc is replaced by:

        p_triaged -> t_audit_log (silent) -> p_post_audit
                                          -> t_assess_risk

    Variant B carries one extra place (``p_post_audit``) and one
    extra silent transition (``t_audit_log``); every other
    structural element is identical to Variant A.
    """
    b = PetriNet()
    # Copy Variant A's places, including initial-marking tokens.
    for p in sorted(variant_a.places):
        b.add_place(
            p,
            tokens=variant_a.initial_marking.get(p, 0),
            label=variant_a.place_labels.get(p),
        )
    # The new audit-log waiting place.
    b.add_place("p_post_audit", label="awaiting risk assessment")

    # Copy Variant A's transitions verbatim.
    for t in sorted(variant_a.transitions):
        b.add_transition(t, label=variant_a.transition_labels.get(t))
    # The silent audit-log step — fires normally in the token-game
    # and trained network, but carries no observable label as far
    # as the weak-bisimulation checker is concerned.
    b.add_transition("t_audit_log", label="audit log", silent=True)

    # Copy every arc from Variant A except the one we're rerouting.
    for src, dst in variant_a.flow:
        if (src, dst) == ("p_triaged", "t_assess_risk"):
            continue
        b.add_arc(src, dst)

    # Inserted arcs realising the audit-log detour.
    b.add_arc("p_triaged", "t_audit_log")
    b.add_arc("t_audit_log", "p_post_audit")
    b.add_arc("p_post_audit", "t_assess_risk")

    return b


def _credit_score_to_marking(trace) -> dict[str, float]:
    """Same attribute mapping the scenario.toml declares — lifted
    out so the bootstrap factory can reuse it."""
    return {"p_assessed": float(trace.attributes.get("credit_score", 0.5))}


# ---------------------------------------------------------------------------
# Bisimulation — strong rejects, weak accepts
# ---------------------------------------------------------------------------


def test_strong_bisimulation_rejects_the_refactored_variant():
    """The two variants have structurally different reachability
    graphs — Variant B has an extra state (``{p_post_audit: 1}``)
    and an extra τ-labelled edge. Strong bisimulation, which
    treats every transition label as observable, must reject
    them as different."""
    ctx = load_scenario(SCENARIO)
    variant_a = ctx.net
    variant_b = _build_variant_b(variant_a)
    assert not are_bisimilar(variant_a, variant_b), (
        "strong bisimulation should reject the refactored variant — "
        "the silent step is an observable structural difference"
    )


def test_weak_bisimulation_accepts_the_refactored_variant():
    """The silent transition is a τ step by construction. Weak
    bisimulation collapses τ paths and recognises the two
    variants as behaviourally equivalent — the headline
    safe-refactoring claim."""
    ctx = load_scenario(SCENARIO)
    variant_a = ctx.net
    variant_b = _build_variant_b(variant_a)
    assert are_weakly_bisimilar(variant_a, variant_b), (
        "weak bisimulation should accept the refactored variant — "
        "the τ step collapses, external behaviour matches"
    )


# ---------------------------------------------------------------------------
# Trained behaviour — same routing rule on the same data
# ---------------------------------------------------------------------------


def _train_variant(net: PetriNet, ctx, seed: int = 0) -> PetriNetModule:
    """Train a fresh module on the scenario's traces. Seeded so
    the two variants are compared at the same random initial
    point (modulo their different structural parameter counts)."""
    torch.manual_seed(seed)
    module = PetriNetModule(
        net,
        firing=ctx.training.firing,
        routing=ctx.training.routing,
        sharpness=ctx.training.sharpness,
    )
    train_on_traces(
        module,
        ctx.traces,
        attribute_to_marking=_credit_score_to_marking,
        steps=ctx.training.steps,
        lr=ctx.training.lr,
    )
    return module


def test_both_variants_learn_the_same_routing_rule():
    """Sanity-check — both variants train on the same traces and
    must learn the same credit_score → approve/decline rule.
    Tested on the two clear cases: strong applications approved,
    weak applications declined."""
    ctx = load_scenario(SCENARIO)
    module_a = _train_variant(ctx.net, ctx)
    module_b = _train_variant(_build_variant_b(ctx.net), ctx)

    with torch.no_grad():
        strong_a = module_a(input_marking={"p_assessed": torch.tensor([0.9])})
        strong_b = module_b(input_marking={"p_assessed": torch.tensor([0.9])})
        weak_a = module_a(input_marking={"p_assessed": torch.tensor([0.1])})
        weak_b = module_b(input_marking={"p_assessed": torch.tensor([0.1])})

    assert strong_a["t_approve"].item() > strong_a["t_decline"].item()
    assert strong_b["t_approve"].item() > strong_b["t_decline"].item()
    assert weak_a["t_decline"].item() > weak_a["t_approve"].item()
    assert weak_b["t_decline"].item() > weak_b["t_approve"].item()


# ---------------------------------------------------------------------------
# Cross-variant comparison — soft routing matches across the input domain
# ---------------------------------------------------------------------------


def test_cross_variant_comparison_shows_high_agreement():
    """Sweep credit_score across the unit interval and compare
    every per-transition firing decision between the two trained
    variants. The shared routing transitions (t_approve /
    t_decline / t_assess_risk / t_triage) should agree at every
    grid point — same threshold, same direction.

    Soft agreement (activations within tolerance) is the more
    delicate metric; hard agreement (same firing decision on
    each side of 0.5) is the easier bar. Both should be near 1.0
    after training; we assert on hard agreement to keep the
    test deterministic across small numerical drift."""
    ctx = load_scenario(SCENARIO)
    module_a = _train_variant(ctx.net, ctx)
    module_b = _train_variant(_build_variant_b(ctx.net), ctx)

    # Sweep the credit_score input domain at a 20-point grid.
    grid_values = [round(x * 0.05, 3) for x in range(1, 20)]
    report = compare_variants(
        module_a,
        module_b,
        input_grid={"p_assessed": grid_values},
        tolerance=0.1,
    )
    assert report.hard_agreement_rate >= 0.95, (
        f"refactored variant should make the same firing decision as the "
        f"baseline at essentially every grid point; got "
        f"{report.hard_agreement_rate:.2%}"
    )


# ---------------------------------------------------------------------------
# Bootstrap CIs — both variants' XOR rules survive resampling
# ---------------------------------------------------------------------------


def test_bootstrap_xor_rule_thresholds_overlap_between_variants():
    """Bootstrap the trace list, retrain each variant per
    resample, extract the credit-score → approve/decline rule,
    and report the distribution of crossover thresholds. The two
    variants' CIs should overlap meaningfully — both are
    learning the same rule from the same data, the structural
    refactoring shouldn't shift the threshold.

    Bootstrap is expensive (n trainings per variant), so this
    test uses a deliberately small ``n_bootstrap`` and shorter
    training. The CI width is not the headline — overlap and
    rough alignment with the empirical decision band (0.31
    declined, 0.71 approved → threshold near 0.5) is."""
    ctx = load_scenario(SCENARIO)
    variant_b = _build_variant_b(ctx.net)

    def _factory_a() -> PetriNetModule:
        torch.manual_seed(0)
        return PetriNetModule(
            ctx.net,
            firing=ctx.training.firing,
            routing=ctx.training.routing,
            sharpness=ctx.training.sharpness,
        )

    def _factory_b() -> PetriNetModule:
        torch.manual_seed(0)
        return PetriNetModule(
            variant_b,
            firing=ctx.training.firing,
            routing=ctx.training.routing,
            sharpness=ctx.training.sharpness,
        )

    ci_a = bootstrap_xor_rule(
        _factory_a,
        ctx.traces,
        attribute_to_marking=_credit_score_to_marking,
        input_place="p_assessed",
        transition_a="t_approve",
        transition_b="t_decline",
        n_bootstrap=20,
        steps=400,
        lr=ctx.training.lr,
        seed=0,
    )
    ci_b = bootstrap_xor_rule(
        _factory_b,
        ctx.traces,
        attribute_to_marking=_credit_score_to_marking,
        input_place="p_assessed",
        transition_a="t_approve",
        transition_b="t_decline",
        n_bootstrap=20,
        steps=400,
        lr=ctx.training.lr,
        seed=0,
    )

    # The two CI intervals must overlap — i.e. the refactoring
    # did not shift the learned threshold.
    lo_a, hi_a = ci_a.crossover_ci_low, ci_a.crossover_ci_high
    lo_b, hi_b = ci_b.crossover_ci_low, ci_b.crossover_ci_high
    assert max(lo_a, lo_b) <= min(hi_a, hi_b), (
        f"variant CIs should overlap: A=[{lo_a:.3f}, {hi_a:.3f}], "
        f"B=[{lo_b:.3f}, {hi_b:.3f}]"
    )
    # The point estimates should also land in the empirical
    # decision band (declined applications ≤ 0.31, approved
    # applications ≥ 0.71).
    assert 0.25 < ci_a.rule.crossover < 0.75
    assert 0.25 < ci_b.rule.crossover < 0.75


# ---------------------------------------------------------------------------
# Distilled-rule sanity — directionally consistent across variants
# ---------------------------------------------------------------------------


def test_extracted_rule_direction_matches_across_variants():
    """Single-training-run sanity check companion to the bootstrap
    test above: the point-estimate distilled XOR rule should
    direct *high* credit_score to ``t_approve`` on both variants.
    Cheaper than bootstrapping and catches direction-of-routing
    drift that the CI overlap test can't see."""
    ctx = load_scenario(SCENARIO)
    module_a = _train_variant(ctx.net, ctx)
    module_b = _train_variant(_build_variant_b(ctx.net), ctx)

    rule_a = extract_xor_rule(
        module_a, "p_assessed", "t_approve", "t_decline",
    )
    rule_b = extract_xor_rule(
        module_b, "p_assessed", "t_approve", "t_decline",
    )
    # Both rules must say "high credit_score → approve". The
    # `transition_above` field carries the transition that wins
    # when the input is above the crossover.
    assert rule_a.transition_above == "t_approve"
    assert rule_b.transition_above == "t_approve"
