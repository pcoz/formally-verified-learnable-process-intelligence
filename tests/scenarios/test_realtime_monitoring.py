"""End-to-end test for the real-time monitoring scenario.

Trains the simplified incident-management net on the happy-path
trace distribution, then pushes a simulated live event stream
through `StreamingEvaluator` and verifies the load-bearing
properties:

* The streaming evaluator emits one `StreamingEvaluation` per
  case as it closes.
* Anomalous cases (the ones that skip the *Resolved* step)
  produce a strictly higher trace-level score than the happy-
  path cases.
* The per-transition residual on `t_resolved` — the step that
  should have fired but didn't — is the dominant signal for
  the anomalous case.
* The on-every-event mode emits one evaluation per incoming
  event for live-dashboard consumers.

Events from the three simulated cases arrive interleaved on
the stream (as they would in production), exercising the
per-case state separation `StreamingEvaluator` is built around.
"""
from __future__ import annotations

from pathlib import Path

from petri_net_nn import (
    StreamingEvaluator,
    StreamingEvent,
    load_scenario,
)


SCENARIO = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "realtime_monitoring"
    / "scenario.toml"
)


# All cases share the same input marking — a single open
# incident — regardless of case_id.
def _open_incident(trace) -> dict[str, float]:
    return {"p_opened": 1.0}


# ---------------------------------------------------------------------------
# Simulated live event stream — three interleaved cases.
# ---------------------------------------------------------------------------


# Two normal incidents (cases A and B) plus one anomalous
# (case C, which skips *Resolved* and goes from *In Progress*
# straight to *Closed*). Events are interleaved to mimic a real
# stream where many cases are in flight at once.
SIMULATED_STREAM: list[StreamingEvent] = [
    StreamingEvent(case_id="A", name="In Progress"),
    StreamingEvent(case_id="B", name="In Progress"),
    StreamingEvent(case_id="A", name="Resolved"),
    StreamingEvent(case_id="C", name="In Progress"),
    StreamingEvent(case_id="B", name="Resolved"),
    StreamingEvent(case_id="A", name="Closed"),
    StreamingEvent(case_id="C", name="Closed"),  # <-- skip Resolved
    StreamingEvent(case_id="B", name="Closed"),
]


def _train_module():
    ctx = load_scenario(SCENARIO)
    module, _ = ctx.train()
    return module, ctx


# ---------------------------------------------------------------------------
# Closed-case scoring — the typical production pattern.
# ---------------------------------------------------------------------------


def test_streaming_evaluator_emits_one_evaluation_per_case_at_close():
    """In the default on-close mode, `on_event` returns ``None``
    and the case is scored on `close_case`. After all three
    cases close, we have exactly three final evaluations."""
    module, _ = _train_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_open_incident
    )
    for event in SIMULATED_STREAM:
        result = evaluator.on_event(event)
        assert result is None, (
            "on-close mode should not emit per-event evaluations"
        )

    finals = {}
    for case_id in ("A", "B", "C"):
        ev = evaluator.close_case(case_id)
        assert ev is not None
        assert ev.closed
        assert ev.case_id == case_id
        finals[case_id] = ev
    assert set(finals) == {"A", "B", "C"}


def test_anomalous_case_scores_strictly_higher_than_normal_cases():
    """The headline real-time monitoring claim: the case that
    skipped *Resolved* must produce a strictly higher trace-
    level anomaly score than either of the two happy-path
    cases. The streaming evaluator's scoring delegates to the
    same offline `anomaly_score` machinery, so this is a
    correctness check on the live path as much as a demo."""
    module, _ = _train_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_open_incident
    )
    for event in SIMULATED_STREAM:
        evaluator.on_event(event)

    final_a = evaluator.close_case("A")
    final_b = evaluator.close_case("B")
    final_c = evaluator.close_case("C")
    assert final_c.trace_score > final_a.trace_score
    assert final_c.trace_score > final_b.trace_score


def test_anomalous_case_residual_is_pinned_to_t_resolved():
    """The actionable signal a live alert would attach to: the
    `t_resolved` transition residual must be the largest
    per-transition signal for the anomalous case. Operators see
    *which* step was skipped, not just *that something* was
    off."""
    module, _ = _train_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_open_incident
    )
    for event in SIMULATED_STREAM:
        evaluator.on_event(event)
    final_c = evaluator.close_case("C")

    # The largest residual on the anomalous case must be on
    # t_resolved — the step the case skipped.
    largest_transition = max(
        final_c.per_transition_residuals,
        key=lambda t: final_c.per_transition_residuals[t],
    )
    assert largest_transition == "t_resolved", (
        f"expected t_resolved to dominate the anomalous case's "
        f"residuals; got {largest_transition} "
        f"({final_c.per_transition_residuals})"
    )


def test_streaming_evaluator_frees_state_on_close():
    """After `close_case`, the case's state is freed. A
    subsequent `score_case` on the same id returns ``None``,
    and a subsequent `on_event` for the same id starts a fresh
    state bucket. Important for production deployments that
    can't afford to leak memory per case."""
    module, _ = _train_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_open_incident
    )
    for event in SIMULATED_STREAM:
        evaluator.on_event(event)
    evaluator.close_case("A")
    assert evaluator.score_case("A") is None


# ---------------------------------------------------------------------------
# On-every-event mode — the live-dashboard pattern.
# ---------------------------------------------------------------------------


def test_on_every_event_mode_emits_one_evaluation_per_event():
    """`score_on_every_event=True` produces a
    `StreamingEvaluation` from every `on_event` call. An
    operator watching a live dashboard sees the anomaly
    accumulate as each event arrives, not just at case close."""
    module, _ = _train_module()
    evaluator = StreamingEvaluator(
        module,
        attribute_to_marking=_open_incident,
        score_on_every_event=True,
    )
    per_event_scores: list = []
    for event in SIMULATED_STREAM:
        result = evaluator.on_event(event)
        assert result is not None, (
            "score_on_every_event=True should emit on every event"
        )
        per_event_scores.append(result)
    assert len(per_event_scores) == len(SIMULATED_STREAM)
    # All in-flight (not-yet-closed) evaluations carry closed=False.
    assert all(not ev.closed for ev in per_event_scores)


def test_process_stream_returns_iterator_of_evaluations():
    """`process_stream` is the pull-shape consumer — pass it an
    iterator of events, get back an iterator of evaluations. In
    on-every-event mode it yields one per event; default mode
    yields none until `close_case` is called explicitly. The
    pull shape is the most direct fit for a Kafka / RabbitMQ /
    Redis Streams consumer."""
    module, _ = _train_module()
    evaluator = StreamingEvaluator(
        module,
        attribute_to_marking=_open_incident,
        score_on_every_event=True,
    )
    evaluations = list(evaluator.process_stream(iter(SIMULATED_STREAM)))
    assert len(evaluations) == len(SIMULATED_STREAM)
