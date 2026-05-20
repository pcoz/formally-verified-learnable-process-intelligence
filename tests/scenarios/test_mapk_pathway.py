"""End-to-end test for the MAPK pathway scenario.

Demonstrates the Phase 10 SIF importer: PETRA loads a Pathway
Commons-style SIF file, compiles it into a Petri net, and runs a
forward pass that propagates activation through the canonical
EGF → ERK signalling cascade.
"""
from __future__ import annotations

from pathlib import Path

import torch

from petri_net_nn import load_scenario


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "mapk_pathway"
    / "scenario.toml"
)


def test_sif_scenario_loads_into_a_well_formed_net():
    """The scenario.toml's ``source = "sif_file"`` adapter branch
    reads the SIF and produces a structurally well-formed Petri
    net — no leftover validation issues."""
    ctx = load_scenario(SCENARIO)
    assert ctx.net.validate() == []


def test_net_contains_expected_signalling_entities():
    """The canonical EGF → MAPK1/3 cascade has these named entities
    — checking a representative subset pins the SIF parse to the
    biology, not just to byte count."""
    ctx = load_scenario(SCENARIO)
    expected = {
        "EGF", "EGFR", "GRB2", "SOS1", "HRAS", "RAF1",
        "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
        "ELK1", "FOS", "RPS6KA1", "DUSP6",
    }
    assert expected <= ctx.net.places


def test_net_contains_expected_cascade_arcs():
    """A few load-bearing arcs from the cascade — RAF1 phosphorylating
    MAP2K1, MAP2K1 phosphorylating MAPK1, ELK1 controlling FOS
    expression."""
    ctx = load_scenario(SCENARIO)
    # RAF1 -> t -> MAP2K1
    assert "RAF1__controls-phosphorylation-of__MAP2K1" in ctx.net.transitions
    assert "MAP2K1__controls-phosphorylation-of__MAPK1" in ctx.net.transitions
    assert "ELK1__controls-expression-of__FOS" in ctx.net.transitions


def test_forward_pass_propagates_activation_from_egf_to_mapk():
    """Compile and run forward in acyclic mode. With EGF held high,
    activation should reach the downstream MAP kinases through the
    full cascade — receptor → adapter → small GTPase → MAP3K →
    MAP2K → MAPK."""
    ctx = load_scenario(SCENARIO)
    module = ctx.compile()

    with torch.no_grad():
        out = module(input_marking={"EGF": torch.tensor([1.0])})

    # Activation should be > 0 at each downstream node — the
    # cascade is connected end-to-end.
    for downstream in ["EGFR", "GRB2", "SOS1", "HRAS", "RAF1",
                        "MAP2K1", "MAPK1", "ELK1", "FOS"]:
        assert out[downstream].item() > 0.0, (
            f"activation did not reach {downstream}"
        )


def test_mapk1_receives_from_both_mek_isoforms_and_dusp6():
    """MAPK1's preset should be exactly the three transitions
    feeding it — both MAP2K activations *and* the DUSP6 regulator.
    Multi-input convergence is a signature feature of the cascade
    that the structural import must preserve."""
    ctx = load_scenario(SCENARIO)
    mapk1_preset = ctx.net.preset("MAPK1")
    assert mapk1_preset == {
        "MAP2K1__controls-phosphorylation-of__MAPK1",
        "MAP2K2__controls-phosphorylation-of__MAPK1",
        "DUSP6__controls-state-change-of__MAPK1",
    }
