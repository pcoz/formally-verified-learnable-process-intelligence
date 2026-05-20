# Changelog

All notable changes to PETRA are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); PETRA
follows [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-05-20

Initial PyPI release. Feature-complete on the original architecture
spec (Phases 1–8) plus the substantial extensions in Phases 9–13.

### What's in it

- **BPMN, PNML, and Pathway Commons SIF importers**; XES (plain and
  gzipped), CSV, and JSON trace loaders.
- **Differentiable Petri-net compiler** with sigmoid / STE firing
  modes, independent / softmax routing, time-unrolled mode for
  cyclic nets, multi-token arc multiplicities, inhibitor arcs,
  transition durations, firing-rate priors, and the
  coloured-Petri-net layer (learnable structural-guard thresholds,
  per-place value channel, optional torch-friendly callable guards
  and arc transforms).
- **Training** via `train_on_traces` against XES / CSV / JSON logs;
  `SharpnessScheduler` for annealing the continuous-relaxation
  sharpness over training.
- **Rule extraction** — `extract_routing_rules`,
  `extract_and_join_rules`, N-way XOR partitions, AND-join
  weighted-vote / quorum rules, downstream-label rewriting so
  distilled rules speak in BPMN vocabulary.
- **Bootstrap confidence intervals + prose explanations** on the
  extracted rules.
- **Counterfactual explanations** on both marking and value
  channels.
- **Sensitivity analysis** — per-input gradients at a base point
  and aggregate input-importance across a trace set.
- **Cross-variant comparison reports** — agreement rates,
  divergent grid points, per-transition breakdown.
- **Anomaly detection** with residuals pinned to BPMN element
  names; trace-level AUC ranking; the canonical corruption
  generators (drop / insert / swap / shuffle) and the
  `FrequencyBaseline` for direct comparison.
- **Bisimulation** — strong (`are_bisimilar`) and weak
  (`are_weakly_bisimilar`, with silent transitions collapsed via
  τ-saturation).
- **Aalst soundness verification** and **deadlock localisation**.
- **CTL temporal-logic model checking** with the six-primitive AST
  (`Atom`, `Not`, `And`, `Or`, `EX`, `EU`, `EG`) plus the derived
  `AX`, `AG`, `AF`, `EF`, `AU` constructors.
- **14 worked end-to-end scenarios** under `examples/` covering
  business processes (including a real Pathway Commons SIF cascade
  and the real BPI Challenge 2013 incidents log), distributed
  protocols, multi-agent coordination, manufacturing lines,
  laboratory protocols, and cell-biology signalling pathways.
- **379 passing tests.**

### Documentation

- `README.md` — the entry point.
- `docs/BUSINESS_ANALYST_GUIDE.md` — no-code primer covering every
  concept end to end, for process analysts and compliance teams.
- `docs/ROADMAP.md` — phase-by-phase product framing and status.
- `docs/DEV_MANUAL.md` — framework + adapter API reference.
- One README per scenario under `examples/`.

[Unreleased]: https://github.com/pcoz/formally-verified-learnable-process-intelligence/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pcoz/formally-verified-learnable-process-intelligence/releases/tag/v0.1.0
