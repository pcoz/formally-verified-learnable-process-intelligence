# Manufacturing cell

A two-station production line with quality-driven routing, modelled
as a Petri net. PETRA running on a manufacturing cell: tokens are
work-in-progress, station occupancy is a place holding the
"currently busy" marker, and the inspection step routes a finished
part to ship or rework depending on the learned quality threshold.

## What this scenario shows

- Token-as-resource: each station's "busy" condition is a token in a
  place; production progresses by moving the token through stations.
- Sequential stage progression (Subnet 1 pattern from §5) chained
  with an XOR routing decision at the inspection step (Subnet 2).
- The framework learns the quality-score → ship-or-rework routing
  from observed production traces without modification.
- The distilled rule reports the threshold in domain language
  ("ship" vs "rework").

## Advantage over alternatives

- **vs. classical Petri-net analysis** (token-game simulation, no
  learning): classical tools require you to *stipulate* the routing
  thresholds — e.g. "if quality > 0.7 then ship". This framework
  *learns* the threshold from observed production traces, so the
  model reflects how the line actually behaves rather than how the
  process engineer assumed it would.
- **vs. monolithic ML** (e.g. LSTM autoencoder over event streams):
  monolithic ML can detect anomalies but cannot tell you *which
  station's quality threshold has drifted* or *which routing rule
  was violated*. This framework reports the residual at the named
  BPMN element (Phase 8 interpretability) and can extract the
  decision rule directly from the trained weights.
- **vs. classical process mining** (e.g. Disco, ProM): process
  mining infers the model from logs but does not unify it with a
  trainable continuous dynamics model. Here, structure + learned
  parameters live in one object; the same artefact supports
  conformance checking, anomaly detection, simulation, and
  cost-ranked refactoring.
- **vs. SPC / control-chart approaches** to quality monitoring:
  control charts flag univariate threshold breaches; this framework
  catches *structural* anomalies (e.g. ship-without-inspect, ship
  with low quality, station-order violations) that SPC has no
  vocabulary for.

## Real-world source

Production-line modelling with Petri nets has a 40-year tradition in
operations research. Foundational references:

- Murata, "Petri Nets: Properties, Analysis and Applications"
  (Proceedings of the IEEE, 1989) — uses job-shop dynamics as a
  canonical example.
- DiCesare et al., "Practice of Petri Nets in Manufacturing" (1993).
- Zurawski & Zhou, "Petri Nets and Industrial Applications: A
  Tutorial" (IEEE Trans. Industrial Electronics, 1994).

Real production lines in semiconductor fabs, automotive assembly,
and pharmaceutical packaging all use this exact structural pattern.

## Files

- `scenario.toml` — the scenario
- `../../tests/scenarios/test_manufacturing_cell.py` — adapter-driven
  end-to-end test

## Running

```
python -m pytest tests/scenarios/test_manufacturing_cell.py
```
