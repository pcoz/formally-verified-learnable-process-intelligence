"""End-to-end test for the cost-ranked variant search scenario —
point #6 of the ROADMAP framing.

Both variants share the same Petri-net topology (and thus pass the
bisimulation check trivially); the only difference is the
per-transition cost weights. After training, the framework reports
the expected cost-to-completion under each variant's cost vector,
and the cheaper variant is selected with formal guarantees that the
behaviour is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    PetriNet,
    are_bisimilar,
    expected_cost,
    load_scenario,
)


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "cost_ranked_refactoring"
    / "scenario.toml"
)


def test_variants_are_bisimilar():
    """The two variants share the same net structure (same TOML
    topology); they're trivially bisimilar. The point: the
    framework verifies this BEFORE training, so the cost comparison
    that follows is over genuinely equivalent processes."""
    ctx = load_scenario(SCENARIO)
    # Build a "second copy" of the net with the same topology — in a
    # real workflow this would be a separately-defined variant.
    same_topology = PetriNet()
    for p in sorted(ctx.net.places):
        tokens = ctx.net.initial_marking.get(p, 0)
        same_topology.add_place(p, tokens=tokens, label=ctx.net.place_labels.get(p))
    for t in sorted(ctx.net.transitions):
        same_topology.add_transition(t, label=ctx.net.transition_labels.get(t))
    for arc in ctx.net.flow:
        same_topology.add_arc(*arc)
    assert are_bisimilar(ctx.net, same_topology)


def test_trained_model_routes_on_credit_score():
    """Sanity-check: the credit_score → approve/decline rule is
    learned. Both variants would learn the same rule (same training
    data, same topology) so we test once."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    with torch.no_grad():
        strong = module(input_marking={"p_assessed": torch.tensor([0.95])})
        weak = module(input_marking={"p_assessed": torch.tensor([0.05])})
    assert strong["t_approve"].item() > strong["t_decline"].item()
    assert weak["t_decline"].item() > weak["t_approve"].item()


def test_variant_b_cheaper_than_variant_a_on_trained_distribution():
    """The load-bearing test. Train once, then evaluate the expected
    cost under each variant's cost weights over a realistic input
    distribution (50/50 strong/weak credit scores). Variant B
    should be meaningfully cheaper because every transition in its
    cost vector is lower than variant A's."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()

    cost_a = ctx.config["cost_ranking"]["variant_a"]
    cost_b = ctx.config["cost_ranking"]["variant_b"]

    weights_a = {k: float(v) for k, v in cost_a.items() if k.startswith("t_")}
    weights_b = {k: float(v) for k, v in cost_b.items() if k.startswith("t_")}

    inputs = torch.linspace(0.05, 0.95, 50)
    cost_a_per_input = expected_cost(
        module, weights_a, input_marking={"p_assessed": inputs}
    )
    cost_b_per_input = expected_cost(
        module, weights_b, input_marking={"p_assessed": inputs}
    )

    mean_a = cost_a_per_input.mean().item()
    mean_b = cost_b_per_input.mean().item()
    assert mean_b < mean_a
    ratio = mean_a / mean_b
    assert ratio > 3.0, f"Expected variant A to be >3x variant B; got {ratio:.2f}"


def test_cost_ranking_consistent_across_inputs():
    """If variant B is cheaper at the routing-decision boundary, it
    must also be cheaper at the extremes — because both variants
    fire the same downstream transitions in the same proportions
    (they're behaviourally equivalent)."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()

    weights_a = {
        k: float(v) for k, v in ctx.config["cost_ranking"]["variant_a"].items()
        if k.startswith("t_")
    }
    weights_b = {
        k: float(v) for k, v in ctx.config["cost_ranking"]["variant_b"].items()
        if k.startswith("t_")
    }

    for credit in (0.05, 0.5, 0.95):
        marking = {"p_assessed": torch.tensor([credit])}
        cost_a = expected_cost(module, weights_a, input_marking=marking).item()
        cost_b = expected_cost(module, weights_b, input_marking=marking).item()
        assert cost_b < cost_a, (
            f"variant B should be cheaper at every credit score; "
            f"failed at {credit}"
        )
