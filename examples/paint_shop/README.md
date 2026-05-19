# Paint shop with curing delay

A production line with one long-running step — the cure transition
takes three time-units before the painted part is ready for
inspection. PETRA running on a workflow where some steps occupy
real wall-clock time, modelled directly with the
**Phase 9 transition-duration** feature.

## What transition durations enable here

Most real workflows have steps that block the line for measurable
time: curing paint, cooling metal, fermenting wine, running a
clinical scan, training an ML model. The transition-duration field
lets the modeller annotate that fact directly on the transition:
`duration = 3` says "this step holds the work for three time-units
before it produces its output."

In the compiler's time-unrolled forward pass that annotation turns
into a real delay. A firing of `t_cure` at step *n* contributes to
`p_inspected`'s activation at step *n+2* — exactly three time-units
later, counting the firing step itself. Subsequent pass / fail
routing still runs as normal once the cured part shows up.

This composes cleanly with the other Phase 9 features: a batch
crating transition can also have a duration, an inhibitor-guarded
mutex transition can have a duration, etc.

## Advantage over alternatives

- **vs. classical Petri-net analysis without durations**: classical
  Petri nets are timeless. Modelling timed behaviour requires
  workarounds (separate "start" / "complete" transitions with
  intermediate places, or explicit token-clock chains) that bloat
  the diagram. PETRA's `duration` keyword is a single annotation
  with the right structural effect.
- **vs. discrete-event simulation packages** (Arena, AnyLogic):
  simulators handle durations but cannot *learn* the routing
  decision (pass / fail in this case) from observed traces.
- **vs. monolithic ML on event logs**: a sequence model can spot
  unusual lead times but cannot tell you the cure step took longer
  than expected on a specific run. Here, the duration annotation
  makes the expected duration explicit in the structure.

## Real-world source

Timed Petri nets are textbook: Ramchandani's 1974 MIT PhD thesis
*"Analysis of Asynchronous Concurrent Systems by Timed Petri Nets"*
introduced fixed-duration transitions; the extension is described
in essentially every Petri-net textbook since. Real production
lines, lab protocols, chemical processes, and software CI
pipelines all have long-running steps that benefit from explicit
duration modelling.

## Files

- `scenario.toml` — net + traces + multi-step training, including
  `duration = 3` on `t_cure`
- `../../tests/scenarios/test_paint_shop.py` — adapter-driven test
  exercising the delayed output and the quality-driven routing

## Running

```
python -m pytest tests/scenarios/test_paint_shop.py
```
