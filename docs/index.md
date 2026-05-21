# PETRA — formally-verified learnable process intelligence

**PETRA** (*Petri-Net Trained Architecture*) learns how a
**discrete-event system routes through its structure** from its
**execution logs** — fitting per-transition firing propensities
conditional on the input marking, over a fixed Petri-net
substrate — and turns the learned dynamics into things you can
act on: readable decision rules, anomaly scores pinned to
specific named elements, formal equivalence proofs between system
variants, and cost rankings over behaviour-preserving
refactorings.

PETRA sits at the intersection of two distinct guarantee
regimes: the **Petri-net substrate** carries strict structural
guarantees (bisimulation, soundness, CTL — all mathematical
proofs about reachable markings, not about anything learned),
while the **learned dynamics** within that substrate carry
empirical guarantees (interpretable rules with bootstrap CIs,
counterfactuals, per-input sensitivity, cross-variant
agreement). See the README's "Three layers of guarantee"
section for the precise three-layer story.

```
pip install petra-nn
```

Requires Python 3.11+ and brings `torch` in as a dependency.

## What this unlocks (the non-obvious parts)

The framework name says "Petri-Net Trained Architecture", which
sounds narrowly technical. The practical wins are larger:

- **A business process is one kind of process — and the same
  machinery handles all the others.** The 21 shipped scenarios
  span loan approvals, distributed-system protocols,
  manufacturing lines, biological signalling pathways, IT
  incident flows, laboratory procedures, multi-agent
  coordination, and network protocols. *Process* is a category
  that crosses industries; PETRA sits at that general level.
- **Process refactoring becomes as safe as code refactoring.**
  Bisimulation proves two designs behave identically *before*
  deployment; cost ranking picks the cheaper of the equivalent
  ones. The same shift that made software's iterate-fast
  culture possible, applied to operating-model design.
- **Regulator-grade model documentation is mechanical.**
  Counterfactuals, sensitivity rankings, bootstrap CIs,
  plain-English prose helpers — what GDPR Article 22, SR 11-7,
  and the EU AI Act all expect — are single function calls.
- **The trained model deploys anywhere.** Same `.onnx` file
  serves from JVM workflow engines, C++ inference servers,
  browser-based decisioning UIs, mobile, and accelerator
  stacks; a FastAPI app exposes the same surface over HTTP for
  non-Python consumers.
- **Log-only is a first-class entry path.** Native discovery
  via the basic Inductive Miner mines a sound Petri net from
  the log alone; `discover_and_train` chains *discover →
  verify → compile → train* in a single call.
- **Every scenario is declarative.** No Python boilerplate per
  scenario. The framework is honest about its limits — the
  Business Analyst Guide §17 carries the long-form treatment
  of where coverability hits the inhibitor-arc undecidability
  boundary. That honesty is what makes §22's
  *substrate-for-a-self-organising-system* claim credible
  rather than handwavy: knowing exactly where the
  formal-tractability boundary sits is what lets you build
  above it without descending into chaos. PETRA stays inside
  its boundary and tells you precisely where the boundary is —
  the precondition for anything else sitting safely on top.

## What to read first

The same reading order works for newcomers and reviewers:

1. **[Business Analyst Guide](BUSINESS_ANALYST_GUIDE.md)** —
   no-code, no-maths walkthrough of every concept end to end.
   Petri nets, BPMN translation, coloured tokens, bisimulation,
   training, rule extraction, counterfactuals, soundness, CTL
   temporal logic. Aimed at process analysts, compliance
   officers, and project managers.
2. **[Roadmap](ROADMAP.md)** — phase-by-phase product framing
   and status; explains *why* each piece exists.
3. **[Developer Manual](DEV_MANUAL.md)** — TOML config schema,
   adapter API, framework module reference, scenario authoring
   guide.
4. **[Worked Examples](WORKED_EXAMPLES.md)** — the twenty-one
   end-to-end scenarios under `examples/`, sorted by the use
   case each represents.
5. **[API Reference](api/index.md)** — module-by-module
   auto-generated docs pulled from the docstrings of every
   public function and class.

The [GitHub repository](https://github.com/pcoz/formally-verified-learnable-process-intelligence)
also carries the worked-example scenarios under `examples/` and
the [README](https://github.com/pcoz/formally-verified-learnable-process-intelligence#readme)'s
walkthrough showing the **whole toolchain** (ProM → CPN Tools →
GreatSPN → TINA → PETRA) composing on a bank-loan unification
case.

## What you get

| Output | What it means |
|---|---|
| **Readable decision rules** | Distilled from trained weights in your domain vocabulary, e.g. *"if amount > £1,000 → credit-review"*. |
| **Confidence intervals on those rules** | Bootstrap resampling reports the percentile-CI on every learned threshold. |
| **Counterfactual explanations** | *"To approve this declined loan, the amount would have needed to be £1,024 instead of £100."* |
| **Sensitivity analysis** | Per-input gradient ranking — *which* inputs the model leans on. |
| **Cross-variant comparison** | Agreement rate across the input domain between two trained variants. |
| **Anomaly scores** | Residuals pinned to specific named elements, not opaque whole-trace numbers. |
| **Equivalence proofs** | Strong and weak bisimulation between two variants, *before* either is deployed. |
| **Soundness verification** | Aalst's option-to-complete + proper-completion + no-dead-transitions check. |
| **Coverability analysis** | Karp-Miller — identifies which places (if any) can hold arbitrarily many tokens. *"Is this place a queue that's about to blow up?"* |
| **Temporal-logic checking** | CTL — *"every approved loan eventually fires the audit-log step"*. |
| **Cost rankings** | Over behaviour-preserving refactorings, fitted to your observed workload. |
