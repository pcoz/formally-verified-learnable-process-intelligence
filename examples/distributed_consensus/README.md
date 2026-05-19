# Distributed consensus — two-phase commit

A two-phase commit protocol modelled as a Petri net with one
coordinator and one participant exchanging prepare / vote / decision
messages. PETRA running on a distributed consensus protocol: each
agent is its own pool, the three message places link the pools,
and the trained network learns the commit-vs-abort routing from the
participant's vote attribute.

## What this scenario shows

- The same cross-pool / shared-message-place primitives that worked
  for BPMN collaborations (Phase 5) describe a distributed protocol
  exactly. Coordinator and participant are two "pools" linked by
  three message places (prepare, vote, decision).
- Routing on the participant's `vote` attribute determines the
  coordinator's commit-vs-abort decision and the participant's
  apply-commit-vs-apply-abort response — an XOR routing decision
  observable at both endpoints.
- §7.2 anomaly detection catches Byzantine patterns: a participant
  that applies commit after a low vote, or a coordinator that
  decides without receiving a vote message — these are structurally
  visible as residuals on the diverging transitions.
- Phase 8 interpretability extracts the routing rule from the
  trained network as "if vote > X → commit, else → abort", in the
  protocol's own vocabulary.

## Advantage over alternatives

- **vs. TLA+ / model-checking** of the protocol: model checking
  verifies the protocol is *correct* (invariants hold under all
  schedules). This framework verifies the protocol is *being
  executed normally* (the observed traces match the structurally
  expected firing distribution). These are complementary — model
  checking proves the spec, this framework monitors the
  implementation.
- **vs. distributed-tracing tools** (e.g. Jaeger, OpenTelemetry):
  tracing shows you each request's path but cannot tell you
  "this commit decision is anomalous given the participant's vote
  attribute". The structural prior plus attribute conditioning lets
  this framework flag Byzantine-style traces directly.
- **vs. log-anomaly ML** (e.g. DeepLog): unstructured log-anomaly
  detection cannot identify *which step of the protocol* deviated.
  Here, residuals pin to specific transitions, so the alert reads
  as "participant applied commit despite vote attribute below the
  learned threshold" rather than a generic anomaly score.
- **Bisimulation across protocol variants** (Phase 2): different
  implementations of the same protocol (e.g. variant 2PC with
  pre-commit phase, vs textbook 2PC) can be checked for
  behavioural equivalence before deployment.

## Real-world source

The protocol structure follows the canonical 2PC description in
distributed systems literature:

- Gray, "Notes on Database Operating Systems" (1978)
- Bernstein, Hadzilacos, Goodman, "Concurrency Control and Recovery
  in Database Systems" (1987)
- Lampson, "Atomic Transactions" lecture notes (1981)

Real-world 2PC implementations are widespread (XA, MS DTC, MySQL
group commit, Spanner's two-phase locking layer); this fixture
captures the minimal structural skeleton.

## Files

- `scenario.toml` — full specification
- `../../tests/scenarios/test_distributed_consensus.py` — adapter-driven
  end-to-end test (training, routing recovery, anomaly detection)

## Running

```
python -m pytest tests/scenarios/test_distributed_consensus.py
```
