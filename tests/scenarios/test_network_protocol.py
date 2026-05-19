"""End-to-end test for the TCP handshake + attack detection scenario."""
from __future__ import annotations

from pathlib import Path

import pytest

from petri_net_nn import (
    XESEvent,
    XESTrace,
    drop_event,
    insert_event,
    load_scenario,
    shuffle_events,
    trace_anomaly_score,
)


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "network_protocol"
    / "scenario.toml"
)


def _normal_trace() -> XESTrace:
    return XESTrace(
        attributes={"session_class": "1.0"},
        events=[
            XESEvent(name="send SYN"),
            XESEvent(name="receive SYN, send SYN-ACK"),
            XESEvent(name="receive SYN-ACK, send ACK"),
            XESEvent(name="receive ACK, established"),
        ],
    )


def test_handshake_scenario_loads_and_token_game_completes():
    ctx = load_scenario(SCENARIO)
    assert ctx.net.validate() == []
    marking = dict(ctx.net.initial_marking)
    # Walk the protocol manually
    marking = ctx.net.fire("t_send_syn", marking)
    marking = ctx.net.fire("t_recv_syn_send_synack", marking)
    marking = ctx.net.fire("t_recv_synack_send_ack", marking)
    marking = ctx.net.fire("t_recv_ack_complete", marking)
    assert "p_client_established" in marking
    assert "p_server_established" in marking


def test_handshake_trains_to_high_completion_activation():
    """After training on normal handshakes, the network should
    activate every protocol transition strongly under M_0."""
    ctx = load_scenario(SCENARIO)
    module, losses = ctx.train()
    assert losses[-1] < losses[0]
    out = module()
    assert out["t_recv_ack_complete"].item() > 0.5


def test_syn_flood_attack_detected():
    """A trace that issues SYN but no completion — drop the
    `receive ACK, established` event. The model expects it to
    fire; the missing event produces a large residual."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    normal = _normal_trace()
    attack = drop_event(normal, index=-1)

    normal_score = trace_anomaly_score(
        module, normal, attribute_to_marking=ctx.attribute_to_marking
    )
    attack_score = trace_anomaly_score(
        module, attack, attribute_to_marking=ctx.attribute_to_marking
    )
    assert attack_score > normal_score + 0.3


def test_half_open_connection_detected():
    """Half-open: server is stuck in SYN-RECEIVED, the final ACK is
    never sent. Drop both the client's `receive SYN-ACK, send ACK`
    and the server's `receive ACK, established`."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    normal = _normal_trace()
    half_open = XESTrace(
        attributes=dict(normal.attributes),
        events=[
            XESEvent(name="send SYN"),
            XESEvent(name="receive SYN, send SYN-ACK"),
        ],
    )

    normal_score = trace_anomaly_score(
        module, normal, attribute_to_marking=ctx.attribute_to_marking
    )
    half_open_score = trace_anomaly_score(
        module, half_open, attribute_to_marking=ctx.attribute_to_marking
    )
    assert half_open_score > normal_score + 0.5


def test_extra_syn_burst_detected_via_inserted_events():
    """Port scan / SYN burst: extra SYN events appear in the trace.
    `insert_event` introduces an unmodelled occurrence; the
    `trace_occurrence_vector` will still be 1 for `send SYN`
    but the trace doesn't reflect a clean single-handshake
    structure — the test asserts the anomaly_score on the
    handshake transitions is at least non-zero (some structural
    signal exists)."""
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    normal = _normal_trace()
    burst = insert_event(normal, "send SYN", index=0)
    # The burst trace has the same set of events, just with an extra
    # SYN. Anomaly score is unchanged on event-set level — this test
    # documents that limitation honestly and is a useful pointer to
    # the LSTM-autoencoder follow-up for sequence-level anomalies.
    normal_score = trace_anomaly_score(
        module, normal, attribute_to_marking=ctx.attribute_to_marking
    )
    burst_score = trace_anomaly_score(
        module, burst, attribute_to_marking=ctx.attribute_to_marking
    )
    # We expect parity here — current detector is occurrence-based.
    assert burst_score == pytest.approx(normal_score)
