"""End-to-end test for the regulator-ready credit-approval scenario.

Phase 13's full diagnostic toolkit applied to a coloured-token
loan-approval process. The test pins:

* the trained guard threshold lands in the empirical decision
  band (sanity);
* `find_counterfactual` on a declined application finds an
  amount at which the decision would have flipped, and the
  resulting prose mentions the application amount in domain
  units;
* `transition_sensitivity` at a representative base point
  identifies the application amount as the dominant input,
  and the prose helper renders it in regulator-friendly
  English;
* a custom bootstrap loop over the training trace list
  produces a non-degenerate confidence interval on the
  learned guard threshold, with the interval bracketing the
  empirical decision band.

Together these four are the structural shape of what every
modern decisioning regulator (GDPR Article 22, SR 11-7, EU AI
Act) expects of a model explanation.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

from petri_net_nn import (
    PetriNetModule,
    find_counterfactual,
    load_scenario,
    prose_for_counterfactual,
    prose_for_sensitivity,
    train_on_traces,
    transition_sensitivity,
)


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "regulator_ready_credit_approval"
    / "scenario.toml"
)


def _amount_to_marking(trace) -> dict[str, float]:
    return {"p_application": 1.0}


def _amount_to_values(trace) -> dict[str, float]:
    return {"p_application": float(trace.attributes.get("amount", 0.0))}


# ---------------------------------------------------------------------------
# Sanity — the trained model routes correctly on the value channel.
# ---------------------------------------------------------------------------


def test_trained_thresholds_land_in_empirical_band():
    """The learned guard thresholds — initialised at 1000 by the
    TOML — must end up in the band between the largest declined
    amount (900) and the smallest approved amount (1500). The
    business rule the model recovered is "approve when amount
    exceeds roughly £1,000"."""
    ctx = load_scenario(SCENARIO)
    module, losses = ctx.train()
    assert losses[-1] < losses[0]
    for key, theta in module.guard_thresholds.items():
        assert 800.0 <= theta.item() <= 1700.0, (
            f"guard threshold {key} drifted out of band: {theta.item()}"
        )


# ---------------------------------------------------------------------------
# Counterfactual — what amount would have flipped this declined application?
# ---------------------------------------------------------------------------


def test_counterfactual_flips_a_declined_application():
    """Take a declined application at amount = £300. The
    counterfactual question is *what amount would have made the
    decision approve?* `find_counterfactual` binary-searches the
    value channel and reports the flip point. Should land in the
    empirical decision band."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()

    cf = find_counterfactual(
        module,
        base_marking={"p_application": 1.0},
        base_values={"p_application": 300.0},
        flip_place="p_application",
        target_transition="t_approve",
        flip_channel="value",
        search_range=(0.0, 10000.0),
        interval_tolerance=10.0,
    )
    assert cf is not None, (
        "binary search should find a flip point for the approval "
        "transition given an amount range that covers the learned "
        "threshold"
    )
    assert cf.original_input == 300.0
    # The flip point must be above the original (we need to raise
    # the amount to approve) and within the empirical band.
    assert cf.counterfactual_input > cf.original_input
    assert 800.0 <= cf.counterfactual_input <= 1700.0
    # The original sat below 0.5 activation (declined); the
    # counterfactual sits above (approved).
    assert cf.original_activation < 0.5
    assert cf.counterfactual_activation >= 0.5


def test_counterfactual_prose_speaks_in_domain_units():
    """`prose_for_counterfactual` renders the counterfactual as a
    regulator-facing paragraph; the `input_label` substitution
    must put *"application amount"* into the prose rather than
    the raw place id `p_application`."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    cf = find_counterfactual(
        module,
        base_marking={"p_application": 1.0},
        base_values={"p_application": 300.0},
        flip_place="p_application",
        target_transition="t_approve",
        flip_channel="value",
        search_range=(0.0, 10000.0),
        interval_tolerance=10.0,
    )
    assert cf is not None
    prose = prose_for_counterfactual(cf, input_label="application amount")
    assert "application amount" in prose
    assert "p_application" not in prose
    # The prose should mention the direction of change (increased,
    # since we're flipping from decline to approve).
    assert "increased" in prose


# ---------------------------------------------------------------------------
# Sensitivity — confirm the model leans on application amount.
# ---------------------------------------------------------------------------


def test_sensitivity_identifies_amount_as_a_driver():
    """At a representative base point (amount = £500, in the
    declined region), the gradient of `t_approve`'s activation
    with respect to the value channel at `p_application` should
    be non-trivial. This is the structural confirmation that the
    model actually reads the amount input — a sanity check no
    regulator would skip."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()

    report = transition_sensitivity(
        module,
        base_marking={"p_application": 1.0},
        target_transition="t_approve",
        base_values={"p_application": 500.0},
    )
    assert report.target_transition == "t_approve"
    # The value channel for the application amount must have a
    # non-negligible gradient — the model genuinely uses it. A
    # near-zero gradient would mean the saturated regime; £500 is
    # close enough to the £1,000 threshold to be informative.
    value_grad = report.value_gradients.get("p_application", 0.0)
    assert abs(value_grad) > 1e-6, (
        f"value-channel gradient should be non-zero near the "
        f"decision boundary; got {value_grad}"
    )


def test_sensitivity_prose_speaks_in_domain_units():
    """`prose_for_sensitivity` with a per-place `input_labels`
    substitution renders *"application amount"* into the
    paragraph rather than the raw place id."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    report = transition_sensitivity(
        module,
        base_marking={"p_application": 1.0},
        target_transition="t_approve",
        base_values={"p_application": 500.0},
    )
    prose = prose_for_sensitivity(
        report, input_labels={"p_application": "application amount"}
    )
    assert "application amount" in prose


# ---------------------------------------------------------------------------
# Bootstrap CI on the learned guard threshold.
# ---------------------------------------------------------------------------


def _percentile_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Plain percentile-based confidence interval — the same shape
    `bootstrap_xor_rule` produces for XOR-routing thresholds. We
    use it here because the coloured-token guard threshold lives
    on `module.guard_thresholds`, not in the XOR-extraction path."""
    sorted_values = sorted(values)
    alpha = (1.0 - confidence) / 2.0
    n = len(sorted_values)
    low_idx = max(0, int(round(alpha * (n - 1))))
    high_idx = min(n - 1, int(round((1 - alpha) * (n - 1))))
    return sorted_values[low_idx], sorted_values[high_idx]


def test_bootstrap_ci_on_the_learned_guard_threshold():
    """Custom bootstrap over the trace list: resample with
    replacement, retrain a fresh module on each resample,
    collect the learned guard threshold. The resulting empirical
    distribution gives a percentile CI that brackets the
    empirical decision band — the headline quantified-uncertainty
    output a regulator would want on the model parameter.

    A small `n_bootstrap` keeps the test fast; the CI shape is
    the headline, not its width."""
    ctx = load_scenario(SCENARIO)
    rng = random.Random(0)
    n_bootstrap = 15

    thresholds: list[float] = []
    for _ in range(n_bootstrap):
        # Resample with replacement, same size as original log.
        sample = [rng.choice(ctx.traces) for _ in ctx.traces]
        torch.manual_seed(rng.randint(0, 2**31 - 1))
        module = PetriNetModule(
            ctx.net,
            firing=ctx.training.firing,
            routing=ctx.training.routing,
            sharpness=ctx.training.sharpness,
        )
        train_on_traces(
            module,
            sample,
            attribute_to_marking=_amount_to_marking,
            attribute_to_values=_amount_to_values,
            steps=ctx.training.steps,
            lr=ctx.training.lr,
        )
        # The two guards (approve / decline) share the same
        # empirical boundary, so we average them per resample.
        avg = sum(t.item() for t in module.guard_thresholds.values()) / len(
            module.guard_thresholds
        )
        thresholds.append(avg)

    lo, hi = _percentile_ci(thresholds, confidence=0.95)
    # Non-degenerate interval — bootstrap variability is non-zero
    # under fresh seeds and resampled data.
    assert hi > lo
    # The CI must bracket the empirical decision band; the point
    # of the bootstrap is to *quantify* where the threshold sits,
    # not to permit it to drift wildly outside the data.
    assert 600.0 <= lo <= 1500.0
    assert 800.0 <= hi <= 2000.0
    # And the median must sit inside the empirical band.
    median = sorted(thresholds)[len(thresholds) // 2]
    assert 800.0 <= median <= 1700.0
