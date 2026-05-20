"""Tests for the streaming anomaly evaluator.

We assemble a trained XOR module up front (the same fixture
the rest of the interpretability suite uses), then drive it
with a stream of :class:`StreamingEvent` objects to confirm:

  * basic lifecycle — on_event accumulates state, score_case
    returns sensible numbers, close_case frees the state;
  * the two scoring policies — on-close (the default) returns
    None from on_event; on-every-event returns an evaluation
    from every call;
  * multiple concurrent cases stay separate;
  * the pull helper process_stream wires up correctly;
  * attribute merging is latest-wins on key collisions;
  * scoring a case_id that's never been seen returns None;
  * the introspection surface (__len__, __contains__,
    open_cases) reflects the current state.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from petri_net_nn import (
    PetriNetModule,
    StreamingEvaluation,
    StreamingEvaluator,
    StreamingEvent,
    XESEvent,
    XESTrace,
    find_xor_groups,
    parse_bpmn,
    parse_xes,
    train_on_traces,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _xor_marking(trace: XESTrace) -> dict[str, float]:
    """Marking for the XOR fixture: a single input value at p_f0
    read from the trace's risk_score attribute."""
    return {"p_f0": float(trace.attributes["risk_score"])}


def _trained_xor_module():
    """Train the same XOR fixture every other test uses. Seeded
    so the test is reproducible."""
    torch.manual_seed(0)
    net = parse_bpmn(FIXTURES / "xor_branch.bpmn")
    module = PetriNetModule(net)
    traces = parse_xes(FIXTURES / "xor_log.xes")
    train_on_traces(
        module, traces, attribute_to_marking=_xor_marking,
        steps=1500, lr=0.1,
    )
    return module


# ---------------------------------------------------------------------------
# Basic lifecycle: on_event accumulates, close_case scores + frees
# ---------------------------------------------------------------------------


def test_on_event_in_default_mode_does_not_emit_evaluations():
    """With score_on_every_event=False (the default), on_event
    just buffers state. The caller must explicitly score or
    close to get an evaluation."""
    module = _trained_xor_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_xor_marking,
    )
    out = evaluator.on_event(
        StreamingEvent(
            case_id="42",
            name="Path A",
            attributes={"risk_score": "0.9"},
        )
    )
    assert out is None
    assert "42" in evaluator
    assert len(evaluator) == 1


def test_close_case_emits_a_scored_evaluation_and_frees_state():
    """close_case returns the final StreamingEvaluation marked
    ``closed=True`` and removes the case from internal state."""
    module = _trained_xor_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_xor_marking,
    )
    evaluator.on_event(
        StreamingEvent(
            case_id="42", name="Path A",
            attributes={"risk_score": "0.9"},
        )
    )
    result = evaluator.close_case("42")
    assert isinstance(result, StreamingEvaluation)
    assert result.case_id == "42"
    assert result.closed is True
    assert result.n_events == 1
    # The XOR fixture has two scored transitions (Path A, Path B)
    # plus any other non-gateway transitions in the net; the
    # residuals dict carries one entry per scored transition.
    assert len(result.per_transition_residuals) >= 2
    # Case is freed.
    assert "42" not in evaluator
    assert len(evaluator) == 0


def test_close_case_unknown_returns_none():
    """Closing a case that was never opened is a no-op that
    returns None rather than raising. Lets the caller close
    optimistically without tracking which cases are open."""
    module = _trained_xor_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_xor_marking,
    )
    assert evaluator.close_case("nonexistent") is None


def test_score_case_does_not_close_or_free_state():
    """score_case returns the current trace's score but leaves
    the case open — subsequent events keep accumulating."""
    module = _trained_xor_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_xor_marking,
    )
    evaluator.on_event(
        StreamingEvent(
            case_id="42", name="Path A",
            attributes={"risk_score": "0.9"},
        )
    )
    snapshot = evaluator.score_case("42")
    assert snapshot is not None
    assert snapshot.closed is False
    assert snapshot.n_events == 1
    # Case is still open.
    assert "42" in evaluator
    # Add a second event; n_events should now be 2.
    evaluator.on_event(
        StreamingEvent(case_id="42", name="Path B", attributes={})
    )
    second = evaluator.score_case("42")
    assert second.n_events == 2


# ---------------------------------------------------------------------------
# Score-on-every-event mode
# ---------------------------------------------------------------------------


def test_score_on_every_event_emits_an_evaluation_per_call():
    """With score_on_every_event=True every on_event returns a
    fresh StreamingEvaluation against the partial trace."""
    module = _trained_xor_module()
    evaluator = StreamingEvaluator(
        module,
        attribute_to_marking=_xor_marking,
        score_on_every_event=True,
    )
    first = evaluator.on_event(
        StreamingEvent(
            case_id="42", name="Path A",
            attributes={"risk_score": "0.9"},
        )
    )
    second = evaluator.on_event(
        StreamingEvent(case_id="42", name="Path B", attributes={})
    )
    assert first is not None
    assert second is not None
    assert first.n_events == 1
    assert second.n_events == 2
    # Neither emission is "closed"; the case is still open.
    assert first.closed is False
    assert second.closed is False
    assert "42" in evaluator


# ---------------------------------------------------------------------------
# Multi-case independence
# ---------------------------------------------------------------------------


def test_multiple_cases_have_independent_state():
    """Events for different cases populate different buckets;
    closing one doesn't affect the other."""
    module = _trained_xor_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_xor_marking,
    )
    evaluator.on_event(
        StreamingEvent(
            case_id="A", name="Path A",
            attributes={"risk_score": "0.9"},
        )
    )
    evaluator.on_event(
        StreamingEvent(
            case_id="B", name="Path B",
            attributes={"risk_score": "0.1"},
        )
    )
    assert evaluator.open_cases() == ["A", "B"]
    a_result = evaluator.close_case("A")
    assert "A" not in evaluator
    assert "B" in evaluator
    assert a_result.case_id == "A"
    assert a_result.n_events == 1
    # B is unaffected — close it cleanly.
    b_result = evaluator.close_case("B")
    assert b_result.case_id == "B"
    assert len(evaluator) == 0


# ---------------------------------------------------------------------------
# process_stream pull helper
# ---------------------------------------------------------------------------


def test_process_stream_yields_evaluations_in_score_on_event_mode():
    """The pull helper just calls on_event and yields non-None
    results. In score-on-event mode, every event yields one
    evaluation; in default mode, nothing is yielded."""
    module = _trained_xor_module()
    evaluator = StreamingEvaluator(
        module,
        attribute_to_marking=_xor_marking,
        score_on_every_event=True,
    )
    events = iter([
        StreamingEvent(
            case_id="42", name="Path A",
            attributes={"risk_score": "0.9"},
        ),
        StreamingEvent(case_id="42", name="Path B", attributes={}),
    ])
    yielded = list(evaluator.process_stream(events))
    assert len(yielded) == 2
    assert all(isinstance(e, StreamingEvaluation) for e in yielded)


def test_process_stream_yields_nothing_in_on_close_mode():
    """In the default on-close policy, process_stream consumes
    the events (state accumulates) but yields nothing — the
    caller will close cases on their own schedule."""
    module = _trained_xor_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_xor_marking,
    )
    events = iter([
        StreamingEvent(
            case_id="42", name="Path A",
            attributes={"risk_score": "0.9"},
        ),
        StreamingEvent(case_id="42", name="Path B", attributes={}),
    ])
    yielded = list(evaluator.process_stream(events))
    assert yielded == []
    assert "42" in evaluator
    assert len(evaluator) == 1


# ---------------------------------------------------------------------------
# Attribute merging
# ---------------------------------------------------------------------------


def test_attributes_merge_with_latest_wins():
    """When a later event for the same case carries a different
    value for a previously-seen attribute key, the latest value
    wins. The trace handed to the offline scorer reflects the
    merged dict."""
    module = _trained_xor_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_xor_marking,
    )
    # First event sets risk_score=0.1.
    evaluator.on_event(
        StreamingEvent(
            case_id="42", name="Path A",
            attributes={"risk_score": "0.1"},
        )
    )
    snapshot_low = evaluator.score_case("42")
    # Second event for the same case updates risk_score=0.9.
    evaluator.on_event(
        StreamingEvent(
            case_id="42", name="Path A",
            attributes={"risk_score": "0.9"},
        )
    )
    snapshot_high = evaluator.score_case("42")
    # The two scoring calls saw different markings (the marking
    # is derived from risk_score), so the residuals must
    # genuinely differ unless the model is constant — in which
    # case at least one residual should differ in either
    # direction. A trained XOR fixture is far from constant; we
    # assert the trace_score actually changed.
    assert snapshot_low.trace_score != snapshot_high.trace_score


# ---------------------------------------------------------------------------
# Introspection surface
# ---------------------------------------------------------------------------


def test_len_and_contains_track_open_cases():
    """__len__ counts open cases; __contains__ checks one;
    open_cases lists them in sorted order."""
    module = _trained_xor_module()
    evaluator = StreamingEvaluator(
        module, attribute_to_marking=_xor_marking,
    )
    assert len(evaluator) == 0
    assert "any" not in evaluator
    evaluator.on_event(
        StreamingEvent(
            case_id="c1", name="Path A",
            attributes={"risk_score": "0.9"},
        )
    )
    evaluator.on_event(
        StreamingEvent(
            case_id="c2", name="Path B",
            attributes={"risk_score": "0.1"},
        )
    )
    assert len(evaluator) == 2
    assert "c1" in evaluator
    assert "c2" in evaluator
    assert "c3" not in evaluator
    assert evaluator.open_cases() == ["c1", "c2"]
