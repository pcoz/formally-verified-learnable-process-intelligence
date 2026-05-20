"""Streaming anomaly evaluation over a live event source.

The offline :func:`petri_net_nn.traces.anomaly_score` expects a
*finished* trace — every event known up front. Production
deployments usually don't have that shape. Cases land event by
event from a Kafka topic, a webhook, a file-tail of a workflow
engine's audit log. This module lets PETRA consume those streams
case-aware and emit anomaly scores in real time, without ever
seeing a "complete" trace.

The runtime model is small and case-aware:

  * Per-case state is held in a dict: each open case carries the
    list of events seen so far plus a merged attribute dict
    (latest-event-wins on key conflicts). New cases are created
    implicitly on the first event for an unseen ``case_id``.
  * Scoring is on-demand. Two policies are supported:
      - **on close** (the default) — call :meth:`close_case` when
        the case is known to be finished; the evaluator scores
        once, returns the result, and frees the case's state.
      - **on every event** — set ``score_on_every_event=True``
        and every :meth:`on_event` returns a fresh score against
        the partial trace as currently buffered. Useful for live
        dashboards; costlier because each event triggers a full
        forward pass.
  * Each scoring call delegates to the existing offline
    :func:`anomaly_score`, treating the partial event list as a
    :class:`XESTrace`. The semantics line up exactly with how
    PETRA already scores offline anomalies, so streaming and
    batch share the same correctness story.

Two API shapes are offered for symmetry with how streams get
plugged in:

  * **Push** — :meth:`on_event` is called from whatever source
    the user wires up. They handle threading, back-pressure, and
    error recovery; the evaluator just maintains state.
  * **Pull** — :meth:`process_stream` consumes an iterator and
    yields evaluations as they're produced. Useful for testing
    and for batch backfill of a stored log.

Threading: instances are **not** thread-safe. A real deployment
that reads from a multi-threaded source should serialise calls
through a single consumer goroutine / queue / lock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from petri_net_nn.compiler import PetriNetModule
from petri_net_nn.traces import (
    AttributeToMarking,
    AttributeToValues,
    anomaly_score,
)
from petri_net_nn.xes import XESEvent, XESTrace


@dataclass(frozen=True)
class StreamingEvent:
    """One event arriving on the stream.

    Attributes
    ----------
    case_id
        Identifier the events get grouped by. Different cases'
        events are scored independently and held in separate
        state buckets inside the evaluator.
    name
        The activity / transition label, matched against the
        net's :attr:`PetriNet.transition_labels` map.
    timestamp
        Optional epoch-seconds timestamp. Carried through but
        not interpreted by the scoring code — it's there so
        downstream consumers can attach it to the emitted
        evaluation. ``None`` is acceptable for sources that
        don't carry a timestamp.
    attributes
        Per-event attributes (channel, amount, sensor reading,
        whatever the source carries). Merged into the case's
        attribute dict on arrival: latest-event-wins when keys
        collide with earlier events for the same case.
    """

    case_id: str
    name: str
    timestamp: float | None = None
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamingEvaluation:
    """One scoring outcome emitted by the evaluator.

    Attributes
    ----------
    case_id
        Which case this score is about.
    n_events
        How many events have been observed for this case so far,
        including the one that triggered this evaluation (if
        applicable).
    trace_score
        Trace-level scalar anomaly score — the sum of all
        per-transition residuals. Higher = more anomalous. Same
        semantics as :func:`petri_net_nn.traces.trace_anomaly_score`.
    per_transition_residuals
        Per-transition |predicted − observed| residuals. Pinned
        to transition IDs; consumers join against
        :attr:`PetriNet.transition_labels` for human-readable
        names.
    closed
        ``True`` if this evaluation was produced by
        :meth:`close_case` (case is done, state freed),
        ``False`` if it was produced by an in-flight
        :meth:`on_event` or :meth:`score_case` call.
    """

    case_id: str
    n_events: int
    trace_score: float
    per_transition_residuals: dict[str, float]
    closed: bool


@dataclass
class _CaseState:
    """Per-case bucket held inside the evaluator. Internal —
    we never expose this directly.

    Carries the event list (so we can build a partial XESTrace
    at scoring time) and the merged attribute dict (so the
    trace-level attribute lookup in ``attribute_to_marking``
    finds whatever the latest event for the case said)."""

    events: list[XESEvent] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)


class StreamingEvaluator:
    """Maintains per-case state and scores cases against a trained
    module as events arrive.

    Parameters
    ----------
    module
        The trained :class:`PetriNetModule` to score against.
    attribute_to_marking
        Same role as in :func:`anomaly_score` — maps a
        :class:`XESTrace` (assembled here from the per-case
        events + merged attributes) to the place-activation
        marking the forward pass expects.
    attribute_to_values
        Optional coloured-Petri-net value channel; passed
        through to the underlying offline scorer when supplied.
    transitions
        Optional explicit list of transition IDs to score
        against. Defaults match the offline anomaly_score:
        every transition whose label doesn't contain ``->``
        (i.e. skip auto-generated gateway labels).
    score_on_every_event
        When ``True``, every :meth:`on_event` call returns a
        :class:`StreamingEvaluation` against the partial trace.
        When ``False`` (default), :meth:`on_event` returns
        ``None`` and the case is only scored on demand via
        :meth:`score_case` or :meth:`close_case`.
    """

    def __init__(
        self,
        module: PetriNetModule,
        *,
        attribute_to_marking: AttributeToMarking,
        attribute_to_values: AttributeToValues | None = None,
        transitions: list[str] | None = None,
        score_on_every_event: bool = False,
    ) -> None:
        self._module = module
        self._attr_to_marking = attribute_to_marking
        self._attr_to_values = attribute_to_values
        self._transitions = transitions
        self._score_on_every_event = score_on_every_event
        # Cases live in a plain dict — single-threaded access
        # assumed (see module docstring).
        self._cases: dict[str, _CaseState] = {}

    # ----- Push API ------------------------------------------------------

    def on_event(
        self, event: StreamingEvent
    ) -> StreamingEvaluation | None:
        """Receive one event. Creates the case if necessary,
        appends the event to that case's state, merges the
        event's attributes into the case attribute dict.

        Returns a :class:`StreamingEvaluation` iff
        ``score_on_every_event`` was set at construction; else
        returns ``None``. The case stays open either way."""
        case = self._cases.setdefault(event.case_id, _CaseState())
        case.events.append(
            XESEvent(name=event.name, attributes=dict(event.attributes))
        )
        # Merge attributes — last value wins on key collision.
        # This matches the typical "current state" semantics most
        # event streams have: the most recent value for a given
        # attribute key is the relevant one.
        case.attributes.update(event.attributes)
        if self._score_on_every_event:
            return self._score(event.case_id, closed=False)
        return None

    def score_case(self, case_id: str) -> StreamingEvaluation | None:
        """Compute and return the score for the case as currently
        seen. Does not close the case — state stays in memory and
        further :meth:`on_event` calls can add to it.

        Returns ``None`` if no events have been seen for
        ``case_id``."""
        if case_id not in self._cases:
            return None
        return self._score(case_id, closed=False)

    def close_case(self, case_id: str) -> StreamingEvaluation | None:
        """Score the case one last time, mark it closed, free its
        state. Returns the final score, or ``None`` if no events
        had been seen for ``case_id``.

        After this call the case_id is forgotten — the next
        :meth:`on_event` for that case_id (should one arrive)
        starts a fresh state bucket."""
        if case_id not in self._cases:
            return None
        evaluation = self._score(case_id, closed=True)
        del self._cases[case_id]
        return evaluation

    # ----- Pull helper ---------------------------------------------------

    def process_stream(
        self, events: Iterator[StreamingEvent]
    ) -> Iterator[StreamingEvaluation]:
        """Consume an iterator of events; yield each evaluation
        produced by :meth:`on_event`.

        Useful for testing and for batch backfill of a stored
        log: feed events in the order they happened and observe
        the same evaluations a live deployment would emit. Does
        NOT call :meth:`close_case` at end-of-stream — the user
        decides when a case is done."""
        for event in events:
            evaluation = self.on_event(event)
            if evaluation is not None:
                yield evaluation

    # ----- Introspection -------------------------------------------------

    def open_cases(self) -> list[str]:
        """Case IDs currently held in state, in deterministic
        sorted order so callers iterating for diagnostics get a
        stable view."""
        return sorted(self._cases)

    def __len__(self) -> int:
        """How many cases are currently open."""
        return len(self._cases)

    def __contains__(self, case_id: str) -> bool:
        """``case_id in evaluator`` — a case is "in" the
        evaluator iff at least one event for it has been seen
        and it hasn't been closed."""
        return case_id in self._cases

    # ----- Internal ------------------------------------------------------

    def _score(
        self, case_id: str, *, closed: bool
    ) -> StreamingEvaluation:
        """Build a :class:`XESTrace` from the case state, run the
        offline :func:`anomaly_score`, package up the result."""
        case = self._cases[case_id]
        trace = XESTrace(
            attributes=dict(case.attributes),
            events=list(case.events),
        )
        residuals = anomaly_score(
            self._module,
            trace,
            attribute_to_marking=self._attr_to_marking,
            attribute_to_values=self._attr_to_values,
            transitions=self._transitions,
        )
        return StreamingEvaluation(
            case_id=case_id,
            n_events=len(case.events),
            trace_score=sum(residuals.values()),
            per_transition_residuals=residuals,
            closed=closed,
        )
