# Batch packaging line

A bottling-and-crating line where 6 filled bottles are batched into a
single crate before inspection. Showcases the **Phase 9 multi-token
markings** feature — the crate transition has an input arc with weight
6, so it only fires once enough bottles have accumulated.

## What multi-token markings enable here

A real packaging line is full of N-to-1 transitions: 6 bottles per
case, 24 cases per pallet, 8 pallets per truck. Multi-token arcs
let the crate transition read exactly what the manufacturing
engineer would draw — *consume 6 bottles, produce 1 crate* — and
the framework's token-game waits for the buffer to fill before
allowing the transition to fire. PETRA's training and anomaly
detection layer cleanly on top of that semantics.

## Token-game behaviour

Starting from 12 raw bottles, the token-game proceeds:

```
fill ×6  → 6 filled bottles in buffer, crate transition now enabled
crate    → 1 crate, 0 bottles left in this batch
inspect  → 1 inspected crate, quality_score drives routing
ship     → 1 shipped (high quality_score)
```

The crate transition is *not* enabled at zero, one, …, five filled
bottles. The framework's `is_enabled` returns False until exactly six
have accumulated. Try to fire it early and `fire()` raises.

## Advantage over alternatives

- **vs. classical Petri-net modellers without learned weights:** they
  draw the same picture but can't tell you the *learned* threshold
  for quality-driven routing from observed traces.
- **vs. discrete-event simulation packages** (Arena, AnyLogic):
  simulators model batching but can't *verify* that two batch-line
  variants are behaviourally equivalent, and can't *learn* the
  routing rule from production data.
- **vs. process-mining tools:** mining tools discover the model from
  logs but don't represent batch semantics natively — they'd see six
  identical fills and miss that they belong together.

## Real-world source

Multi-token batch semantics are exactly how Petri nets are used in
manufacturing modelling — see Murata's 1989 survey ("Petri Nets:
Properties, Analysis and Applications", Proc. IEEE) for the textbook
treatment and Zurawski & Zhou ("Petri Nets and Industrial
Applications: A Tutorial", IEEE Trans. Industrial Electronics, 1994)
for canonical job-shop and flow-shop examples.

## Files

- `scenario.toml` — net + traces + training params, including the
  `weight = 6` on the bottle-to-crate arc
- `../../tests/scenarios/test_batch_packaging.py` — adapter-driven
  test exercising both the token-game (multi-token firing) and the
  trained network (quality routing)

## Running

```
python -m pytest tests/scenarios/test_batch_packaging.py
```
