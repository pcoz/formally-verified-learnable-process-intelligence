"""End-to-end test for the 2-phase commit scenario.

Driven entirely from `examples/distributed_consensus/scenario.toml`.
Validates that the cross-pool / shared-message-place primitives work
for distributed protocols, that routing on the vote attribute is
learned, and that anomaly detection catches Byzantine traces.
"""
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
    / "distributed_consensus"
    / "scenario.toml"
)


def test_two_phase_commit_loads_and_validates():
    ctx = load_scenario(SCENARIO)
    assert ctx.net.validate() == []
    assert "p_msg_prepare" in ctx.net.places
    assert "p_msg_vote" in ctx.net.places
    assert {"t_co_decide_commit", "t_co_decide_abort"}.issubset(ctx.net.transitions)


def test_two_phase_commit_token_game_walks_commit_path():
    """Manually fire the transitions of a normal commit run to confirm
    the cross-pool message flow propagates tokens between coordinator
    and participant correctly."""
    ctx = load_scenario(SCENARIO)
    net = ctx.net
    marking = dict(net.initial_marking)

    marking = net.fire("t_co_send_prepare", marking)
    assert "p_co_waiting" in marking
    assert "p_msg_prepare" in marking

    marking = net.fire("t_part_vote", marking)
    assert "p_msg_vote" in marking
    assert "p_part_voted" in marking
    assert "p_msg_prepare" not in marking

    marking = net.fire("t_co_decide_commit", marking)
    assert "p_co_committed" in marking
    assert "p_msg_decision_commit" in marking

    marking = net.fire("t_part_commit", marking)
    assert "p_part_done" in marking


def test_training_learns_vote_to_decision_routing():
    """High vote → commit branch fires more strongly than abort branch
    on the coordinator side; low vote → opposite."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()

    with torch.no_grad():
        high = module(input_marking={"p_msg_vote": torch.tensor([0.95])})
        low = module(input_marking={"p_msg_vote": torch.tensor([0.05])})

    assert high["t_co_decide_commit"].item() > high["t_co_decide_abort"].item()
    assert low["t_co_decide_abort"].item() > low["t_co_decide_commit"].item()


def test_byzantine_anomaly_detected():
    """A trace where the participant 'applies commit' despite a low
    vote — the Byzantine pattern §7.2 catches as a structural
    deviation."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()

    normal = XESTrace(
        attributes={"vote": "0.95"},
        events=[
            XESEvent(name="send prepare"),
            XESEvent(name="submit vote"),
            XESEvent(name="decide commit"),
            XESEvent(name="apply commit"),
        ],
    )
    byzantine = XESTrace(
        attributes={"vote": "0.05"},
        events=[
            XESEvent(name="send prepare"),
            XESEvent(name="submit vote"),
            XESEvent(name="decide commit"),
            XESEvent(name="apply commit"),
        ],
    )
    normal_scores = ctx.anomaly_score(module, normal)
    byzantine_scores = ctx.anomaly_score(module, byzantine)
    assert sum(byzantine_scores.values()) > sum(normal_scores.values()) + 0.3


def test_distilled_rule_routes_commit_on_high_vote():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    rules = ctx.extract_rules(module)
    commit_rule = next(
        (r for r in rules["xor"] if "decide commit" in (r.label_above, r.label_below)),
        None,
    )
    assert commit_rule is not None
    assert commit_rule.label_above == "decide commit"
    assert commit_rule.label_below == "decide abort"
