# Changelog

All notable changes to PETRA are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); PETRA
follows [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.2.1] — 2026-06-02

Documentation-only patch. No code or API changes.

### Fixed

- **PyPI homepage links.** The README (which is the PyPI
  long-description) used repo-relative links (`docs/*.md`,
  `examples/*/`, the `LICENSE` badge). These render on PyPI but
  resolve to `pypi.org/project/petra-nn/...` and 404. Rewrote every
  relative link to an absolute GitHub URL (`blob/main/...` for files,
  `tree/main/...` for example directories) so they work from the PyPI
  page. The GitHub-rendered README is unaffected (absolute links work
  there too).

## [0.2.0] — 2026-05-21

Public-API-additive release. Minor bump (new public functions
exported, no breaking changes to the existing surface).

### Added

- **Phase 11 coverability analysis.** `petri_net_nn.coverability`
  exposes `coverability_graph(net)`, the convenience
  `is_bounded(net)`, the `CoverabilityReport` dataclass, and the
  `OMEGA` sentinel. Karp-Miller construction; exact on
  inhibitor-free nets; conservative (may overreport
  unboundedness) on nets with inhibitor arcs, per the Minsky /
  Turing undecidability of the general case. Long-form
  treatment in BUSINESS_ANALYST_GUIDE §17 with three industry
  examples (payment-reconciliation queues, NHS specialty
  waiting lists, manufacturing WIP at bottlenecks).
- **Phase 12 basic Inductive Miner.** `discover_inductive(traces)`
  mines a sound Petri net from a trace set; `discover_and_train`
  chains the whole `discover → verify → compile → train`
  pipeline in one call. Process-tree primitives `Activity`,
  `Tau`, `Sequence`, `ExclusiveChoice`, `Parallel`, `Loop`,
  `ProcessTree` are also exported.
- **Phase 14 productionisation toolkit.** ONNX export
  (`export_onnx`), streaming evaluator (`StreamingEvent`,
  `StreamingEvaluator`, `StreamingEvaluation`), FastAPI REST
  wrapper (`build_app`), and the `petra-train` / `petra-score`
  / `petra-serve` command-line entry points. Documented
  integration patterns for Camunda / Activiti / Flowable in
  `docs/INTEGRATION_PATTERNS.md`.
- **Adapter:** `net.source = "discover"` keyword invokes the
  Inductive Miner against the `[traces]` section, so the
  log-only pipeline stays TOML-driven with no per-scenario
  Python boilerplate.
- **`examples/discover_and_train_pipeline/`** worked-example
  scenario — a four-trace loan-approval-style log mined to a
  sound Petri net and trained end to end, with six pinning
  tests.
- **`examples/safe_refactoring/`** worked-example scenario —
  two structurally-different variants of a loan-approval
  process (Variant B adds a silent audit-log transition).
  Strong bisim rejects, weak bisim accepts; cross-variant
  comparison and bootstrap CIs both pinned by six tests.
  Demonstrates Phase 11 weak bisim + Phase 13 cross-variant
  comparison + bootstrap CIs in a single end-to-end story.
- **`examples/compliance_audit/`** worked-example scenario —
  loan-approval net with two regulatory CTL invariants
  (audit-after-approve, decline-after-credit-check); the
  deliberately-broken variant that bypasses the audit step
  fails the CTL invariant with a counterexample marking
  witnessing the violation. Demonstrates Phase 11 Aalst
  soundness + deadlock localisation + CTL model checking in a
  single end-to-end compliance story. Seven tests.
- **`examples/regulator_ready_credit_approval/`** worked-example
  scenario — coloured-token loan-approval net instrumented with
  Phase 13's full diagnostic toolkit (bootstrap CI on the
  learned guard threshold, counterfactual flipping a declined
  application, sensitivity ranking, prose helpers with domain
  labels). The combination is the model-explanation shape
  regulators (GDPR Article 22, SR 11-7, EU AI Act) expect on
  trained decisioning models. Six tests.
- **`examples/realtime_monitoring/`** worked-example scenario —
  Phase 14 streaming evaluator running on a simulated
  incident-management event stream; the anomalous case (which
  skips *Resolved*) scores strictly higher than the normal
  ones, with the per-transition residual pinned to
  `t_resolved` — the actionable real-time alert signal. Both
  the default on-close mode and the on-every-event
  live-dashboard mode exercised. Six tests.
- **`examples/onnx_deployment/`** worked-example scenario —
  Phase 14 ONNX export end to end. Small loan-approval
  XOR-routing model trained from inline traces, exported to
  `.onnx`, loaded via `onnxruntime`, parity-checked against
  the torch forward pass within 1e-4 across a sweep of
  inputs. The dynamic-batch axis is exercised
  (`batch_size=1` export accepts a batch of 50 at inference).
  The deployment bridge to JVM workflow engines, C++
  inference servers, browser-based decisioning UIs, mobile,
  and accelerator stacks. Four tests.
- **`examples/rest_serving/`** worked-example scenario — Phase
  14 REST inference API end to end. Small loan-approval model
  trained, wrapped in a FastAPI app via `build_app`, every
  endpoint exercised through FastAPI's `TestClient`:
  `/healthz` (liveness + module stats), `/schema` (net
  inventory for client SDK generators), `/forward`
  (per-event inference), `/anomaly` (trace conformance),
  `/counterfactual` (flip-point search), `/sensitivity`
  (gradient ranking), plus the auto-generated
  `/openapi.json` schema. The engine-integration walkthrough
  that closes the deployment story without needing a real
  workflow engine on the box. Eight tests.

With this scenario all seven of the originally-pending
worked-example scenarios are now landed.
- **Auto-built MkDocs Material documentation site** deployed on
  every push to `main`. URL:
  <https://pcoz.github.io/formally-verified-learnable-process-intelligence/>.

### Changed

- **Honest framing in README and BUSINESS_ANALYST_GUIDE** —
  native log-to-net discovery is now explicitly acknowledged
  (the BA guide had previously said PETRA was deliberately
  *not* a discovery tool; Phase 12 made that wrong). The
  framing calibrates which logs PETRA's basic IM handles well
  vs. when ProM's noise filters belong as a pre-step.
- **Worked-examples table split out** of the README into its
  own `docs/WORKED_EXAMPLES.md`. The README keeps a short
  intro paragraph and links to the dedicated document; the
  MkDocs site picks the page up in the reading order under
  the Developer Manual.

### Fixed

- **CI dependency for torch 2.12.** From torch 2.12.0
  onwards, `torch.onnx.export` requires the `onnx` schema
  package at import time. Added `onnx>=1.15` to the `[onnx]`
  optional extra; without this CI began failing on every push
  on 2026-05-20.

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

[Unreleased]: https://github.com/pcoz/formally-verified-learnable-process-intelligence/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pcoz/formally-verified-learnable-process-intelligence/releases/tag/v0.2.0
[0.1.0]: https://github.com/pcoz/formally-verified-learnable-process-intelligence/releases/tag/v0.1.0
