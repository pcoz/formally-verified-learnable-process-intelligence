# Real-time monitoring — live anomaly scoring on an event stream

PETRA's offline `anomaly_score` path scores stored traces in
batch. **`StreamingEvaluator`** does the same job *live*: events
arrive one at a time from a message bus, webhook handler, or
file tail; the evaluator maintains per-case state and emits a
score when the case closes (or, optionally, after every event).
Same module, same scoring semantics, different shape of
consumer — pick whichever fits your deployment.

This scenario exercises that end-to-end on a simulated incident-
management event stream.

## What this scenario demonstrates

End-to-end real-time monitoring:

1. **Train once on the happy-path distribution.** Six inline
   traces of the canonical *opened → in_progress → resolved
   → closed* lifecycle fit the transition activations.

2. **Push a mixed event stream through the evaluator.** Three
   simulated incidents arrive on the stream with their events
   interleaved (as they would in production where many cases
   are in flight simultaneously). Two follow the happy path;
   one is anomalous — it skips the *Resolved* step and goes
   straight from *In Progress* to *Closed*.

3. **Verify the live anomaly signal.** When each case closes,
   the evaluator emits a final `StreamingEvaluation`. The
   anomalous case's trace-level score is strictly higher than
   the normal cases', and the residual is pinned to the
   specific `t_resolved` transition that should have fired
   but didn't — the actionable signal a real-time alert
   would attach to.

4. **The on-every-event mode** (the live-dashboard pattern)
   is also tested: every incoming event produces an updated
   partial-trace score so an operator watching a dashboard
   sees anomaly accumulating mid-trace rather than only at
   close-time.

## Why this matters

Most workflow engines and ticket systems already emit
structured events to a message bus (Kafka, RabbitMQ, NATS,
or just an audit-log table). Wiring those events into a
~25-line consumer that calls `evaluator.on_event(...)` per
incoming message gives you live conformance monitoring with
no batch lag and per-transition pinning of where the
deviation occurred.

`StreamingEvaluator` is single-threaded by design (the
typical pattern is one consumer per topic partition); a
multi-threaded deployment serialises through a single
consumer per shard.

See [`docs/INTEGRATION_PATTERNS.md`](../../docs/INTEGRATION_PATTERNS.md)
for the three documented engine-integration recipes — webhook,
audit-log tail, streaming subscription. This scenario
exercises the third pattern in test form.

## Files

- `scenario.toml` — net, training traces, training
  hyperparameters. The simulated live event stream is
  constructed in the test, not in the config.
- `../../tests/scenarios/test_realtime_monitoring.py` — trains
  the module, pushes the simulated stream through
  `StreamingEvaluator`, pins the load-bearing assertions.

## Running

```
python -m pytest tests/scenarios/test_realtime_monitoring.py
```
