# Discover and train pipeline — native log-to-net discovery, end to end

PETRA's basic Inductive Miner discovers a sound Petri net
directly from an event log, with no structural model required
up front. This scenario chains discovery, soundness verification,
compilation, and training into a single end-to-end run driven
entirely from `scenario.toml`.

## What this scenario demonstrates

- **`net.source = "discover"`** — the new adapter keyword that
  invokes the Inductive Miner. The TOML supplies only the log;
  the structure is mined.
- **Sound-by-construction guarantee** — every Petri net the
  basic IM emits is option-to-complete, properly terminating,
  and free of dead transitions. The scenario test asserts this
  explicitly.
- **All four canonical cut shapes** in one log: sequence,
  parallel, exclusive choice (and the trivial leaf case). The
  log encodes a loan-approval-style flow with a parallel
  verification step and an exclusive approve/decline decision:

  ```
  request → ( verify_id || credit_check ) → review →
                                              ( approve | decline ) → close
  ```

- **`discover_and_train` one-call API** — the test also
  exercises the convenience entry point that bundles
  discovery → soundness → compile → train into a single
  function, for users who have *only* a log.

## Scope and honest framing

The synthetic log here is small (four traces) and clean by
design — the four traces cover both interleavings of the
parallel block and both branches of the exclusive choice. The
basic Inductive Miner recovers the structure unambiguously and
the assertions in the test are accordingly strict.

For **noisy real-world logs**, the basic IM may collapse into
the flower-model fallback (an XOR over all activities inside a
self-loop). PETRA's documentation calls this out as the limit
of native discovery; the recommended preprocessing path is
ProM's infrequent-noise filters (IMf), whose PNML output PETRA
consumes via [`parse_pnml`](../../docs/api/pnml.md). Once the
log is clean, the rest of this scenario's pipeline applies
unchanged.

## Files

- `scenario.toml` — declarative configuration: inline traces,
  `net.source = "discover"`, training hyperparameters.
- `../../tests/scenarios/test_discover_and_train_pipeline.py` —
  end-to-end test asserting soundness, replay, and a falling
  training loss.

## Running

```
python -m pytest tests/scenarios/test_discover_and_train_pipeline.py
```
