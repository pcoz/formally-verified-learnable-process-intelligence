# PETRA — formally-verified learnable process intelligence

**PETRA** (*Petri-Net Trained Architecture*) learns how a
**discrete-event system** actually behaves from its **execution
logs**, and turns the learned behaviour into things you can act
on: readable decision rules, anomaly scores pinned to specific
named elements, formal equivalence proofs between system
variants, and cost rankings over behaviour-preserving
refactorings.

```
pip install petra-nn
```

Requires Python 3.11+ and brings `torch` in as a dependency.

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
4. **[API Reference](api/index.md)** — module-by-module
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
| **Temporal-logic checking** | CTL — *"every approved loan eventually fires the audit-log step"*. |
| **Cost rankings** | Over behaviour-preserving refactorings, fitted to your observed workload. |
