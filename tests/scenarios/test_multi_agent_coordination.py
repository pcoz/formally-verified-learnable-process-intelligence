"""End-to-end test for the contract-net coordination scenario."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    XESEvent,
    XESTrace,
    load_scenario,
)


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "multi_agent_coordination"
    / "scenario.toml"
)


def test_contract_net_loads_and_validates():
    ctx = load_scenario(SCENARIO)
    assert ctx.net.validate() == []
    assert {"t_announce", "t_evaluate", "t_award_a", "t_award_b"}.issubset(
        ctx.net.transitions
    )


def test_contract_net_token_game_walks_a_wins_path():
    """Walk a normal coordination round end-to-end: announce → both
    bid → evaluate → award A → perform."""
    ctx = load_scenario(SCENARIO)
    marking = dict(ctx.net.initial_marking)
    marking = ctx.net.fire("t_announce", marking)
    assert "p_msg_announce_a" in marking
    assert "p_msg_announce_b" in marking
    marking = ctx.net.fire("t_bid_a", marking)
    marking = ctx.net.fire("t_bid_b", marking)
    assert "p_msg_bid_a" in marking
    assert "p_msg_bid_b" in marking
    marking = ctx.net.fire("t_evaluate", marking)
    assert "p_manager_evaluated" in marking
    marking = ctx.net.fire("t_award_a", marking)
    marking = ctx.net.fire("t_perform_a", marking)
    assert "p_a_done" in marking


def test_evaluate_is_three_way_and_join():
    """The evaluate transition demonstrates the AND-join shape on a
    real coordination primitive — it requires both bid messages
    plus the manager's awaiting state."""
    ctx = load_scenario(SCENARIO)
    preset = ctx.net.preset("t_evaluate")
    assert preset == {"p_manager_awaiting", "p_msg_bid_a", "p_msg_bid_b"}


def test_award_routing_learned_on_a_advantage():
    """High a_advantage routes to award_a; low routes to award_b."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    with torch.no_grad():
        a_wins = module(input_marking={"p_manager_evaluated": torch.tensor([0.95])})
        b_wins = module(input_marking={"p_manager_evaluated": torch.tensor([0.05])})
    assert a_wins["t_award_a"].item() > a_wins["t_award_b"].item()
    assert b_wins["t_award_b"].item() > b_wins["t_award_a"].item()


def test_award_rule_distilled_in_protocol_vocabulary():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    rules = ctx.extract_rules(module)
    award_rule = next(
        (r for r in rules["xor"] if "award to A" in (r.label_above, r.label_below)),
        None,
    )
    assert award_rule is not None
    assert award_rule.label_above == "award to A"
    assert award_rule.label_below == "award to B"


def test_evaluate_distilled_as_three_input_and_join_rule():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    rules = ctx.extract_rules(module)
    evaluate_rule = next(
        (r for r in rules["and_join"] if r.transition == "t_evaluate"),
        None,
    )
    assert evaluate_rule is not None
    assert len(evaluate_rule.inputs) == 3


def test_anomaly_award_before_bids():
    """A trace where an award is issued without a preceding bid is a
    coordination violation. The framework should flag a higher
    anomaly score than a clean trace."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    clean = XESTrace(
        attributes={"a_advantage": "0.95"},
        events=[
            XESEvent(name="announce task"),
            XESEvent(name="A submits bid"),
            XESEvent(name="B submits bid"),
            XESEvent(name="evaluate bids"),
            XESEvent(name="award to A"),
            XESEvent(name="A performs task"),
        ],
    )
    coordination_violation = XESTrace(
        attributes={"a_advantage": "0.95"},
        events=[
            XESEvent(name="announce task"),
            XESEvent(name="award to A"),
            XESEvent(name="A performs task"),
        ],
    )
    clean_score = sum(ctx.anomaly_score(module, clean).values())
    bad_score = sum(ctx.anomaly_score(module, coordination_violation).values())
    assert bad_score > clean_score + 0.3
