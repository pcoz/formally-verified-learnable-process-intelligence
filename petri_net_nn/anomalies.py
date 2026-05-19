"""Synthetic anomaly generators for XES traces.

§7.2 of the architecture spec promises anomaly detection grounded in
the Petri-net substrate — a process instance that produces an unusual
activation pattern, relative to the trained distribution, is anomalous.
To quantify that promise we need controlled anomalies: traces that
deliberately violate the structure in known ways. These generators
produce them.

Each function takes a normal ``XESTrace`` and returns a new
``XESTrace`` with the corruption applied; the input trace is not
mutated. The corruption types correspond to the kinds of process
deviations that §10 Step 4 calls out as evaluation targets:

  * drop_event — a step is skipped entirely;
  * insert_event — a spurious step is added;
  * swap_event_labels — branch flipping (a step from the wrong
    BPMN branch is recorded);
  * shuffle_events — the right steps fire but out of order.

A frequency-baseline detector lives here too, so anomaly evaluations
can compare the structured Petri-net detection against a model that
sees only marginal event frequencies — isolating the contribution of
the structural prior plus attribute conditioning.
"""
from __future__ import annotations

import math
import random
from collections import Counter

from petri_net_nn.xes import XESEvent, XESTrace


def _copy_event(e: XESEvent) -> XESEvent:
    return XESEvent(name=e.name, attributes=dict(e.attributes))


def _copy_trace(trace: XESTrace) -> XESTrace:
    return XESTrace(
        attributes=dict(trace.attributes),
        events=[_copy_event(e) for e in trace.events],
    )


def drop_event(trace: XESTrace, index: int = -1) -> XESTrace:
    """Return a copy of ``trace`` with one event removed (last by
    default). Empty input traces are returned unchanged — there is
    nothing to drop."""
    if not trace.events:
        return _copy_trace(trace)
    new = _copy_trace(trace)
    if index < 0:
        index += len(new.events)
    del new.events[index]
    return new


def insert_event(
    trace: XESTrace, label: str, *, index: int | None = None
) -> XESTrace:
    """Return a copy of ``trace`` with a new event named ``label``
    inserted at ``index`` (end of the trace by default)."""
    new = _copy_trace(trace)
    pos = len(new.events) if index is None else index
    new.events.insert(pos, XESEvent(name=label))
    return new


def swap_event_labels(
    trace: XESTrace, label_a: str, label_b: str
) -> XESTrace:
    """Return a copy of ``trace`` with every occurrence of ``label_a``
    replaced by ``label_b`` and vice versa. Models BPMN branch
    flipping: the *wrong* branch's task name appears in the trace."""
    swap = {label_a: label_b, label_b: label_a}
    new = _copy_trace(trace)
    for e in new.events:
        e.name = swap.get(e.name, e.name)
    return new


def shuffle_events(trace: XESTrace, *, seed: int | None = None) -> XESTrace:
    """Return a copy of ``trace`` with its events reordered randomly.
    Useful for processes where the order matters (sequential / AND-join
    patterns); for single-event traces it is a no-op."""
    new = _copy_trace(trace)
    if len(new.events) <= 1:
        return new
    rng = random.Random(seed)
    rng.shuffle(new.events)
    return new


class FrequencyBaseline:
    """Marginal-frequency anomaly detector.

    Fits a unigram distribution over event labels in a training set of
    traces; the anomaly score of a new trace is the negative log
    probability of its events under that distribution, summed across
    events. Unseen labels get a small smoothing mass.

    This baseline deliberately ignores trace-level attributes and the
    process structure. Its purpose in Phase 7 is to provide a
    contrast: branch-flip anomalies that the structured Petri-net
    detector can catch — because it conditions on the input marking
    derived from trace attributes — are *invisible* to the frequency
    baseline, which sees only that the events are familiar event
    labels.
    """

    def __init__(self, *, smoothing: float = 0.5) -> None:
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.smoothing = smoothing
        self._log_probs: dict[str, float] = {}
        self._log_unseen: float = 0.0

    def fit(self, traces: list[XESTrace]) -> "FrequencyBaseline":
        counts: Counter[str] = Counter()
        for trace in traces:
            for event in trace.events:
                counts[event.name] += 1
        total = sum(counts.values()) + self.smoothing * (len(counts) + 1)
        self._log_probs = {
            label: math.log((c + self.smoothing) / total)
            for label, c in counts.items()
        }
        self._log_unseen = math.log(self.smoothing / total)
        return self

    def score(self, trace: XESTrace) -> float:
        if not self._log_probs and self._log_unseen == 0.0:
            raise RuntimeError("FrequencyBaseline.fit() must be called first")
        if not trace.events:
            return -self._log_unseen
        score = 0.0
        for event in trace.events:
            score -= self._log_probs.get(event.name, self._log_unseen)
        return score
