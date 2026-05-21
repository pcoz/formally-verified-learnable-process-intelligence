# Regulator-ready credit approval — Phase 13's full diagnostic toolkit on one business case

The same coloured-Petri-net loan-approval process as
[`credit_approval_coloured/`](../credit_approval_coloured/) —
tokens at `p_application` carry the application amount, and
the approve / decline transitions guard structurally on that
value — but instrumented with the **full Phase 13 diagnostic
surface** that a regulator would expect on every trained
decisioning model.

## What this scenario demonstrates

Four diagnostic outputs, each of them mechanical, each of them
auditable, each of them rendered into plain-English prose for
inclusion in model documentation:

1. **Bootstrap confidence intervals** on the learned
   guard threshold (`module.guard_thresholds[...]`). The
   point estimate is the model's best-guess decision boundary;
   the CI reports the uncertainty around it. *"The decision
   threshold is at £1,034 plus-or-minus £58 with 95%
   confidence over 30 bootstrap resamples."*

2. **Counterfactual explanations** on a declined application.
   `find_counterfactual(flip_channel="value")` binary-searches
   the application amount and reports the value at which the
   decision would have flipped. *"This application was
   declined at £300. The decision would have flipped at
   £1,034 — i.e. an increase of approximately £734 in
   application amount."*

3. **Sensitivity analysis** at a representative base point.
   `transition_sensitivity` reports the gradient of the
   approve-firing activation with respect to each input. In
   this single-input model the application amount is the
   dominant driver; the analysis confirms that quantitatively
   rather than by assertion. *"The approval decision is most
   sensitive to the application amount; a £100 increase at
   this base point would raise the approval activation by
   approximately 0.13."*

4. **Prose generation** for all of the above via
   `prose_for_counterfactual` and `prose_for_sensitivity`. The
   output is plain English with domain labels substituted
   (`"application amount"` instead of `"p_application"`),
   ready to drop into a model-explanation document for
   compliance review.

## Why this matters for regulators

Decisioning models used in lending, insurance, employment,
and similar high-stakes settings are increasingly required by
regulators to produce per-decision explanations. The
applicable frameworks include:

- **GDPR Article 22** (right to explanation for automated
  decisions);
- **US OCC / Federal Reserve SR 11-7** (model risk management);
- **EU AI Act Annex III** (high-risk AI system requirements).

What all of these regimes demand, at the structural level, is:

- *Quantified uncertainty* on the model's parameters (PETRA's
  bootstrap CIs);
- *Per-prediction actionable explanations* — what would have
  changed the outcome (PETRA's counterfactuals);
- *Identification of which inputs the model uses* (PETRA's
  sensitivity ranking);
- *Documentation in language a non-technical reviewer can
  understand* (PETRA's prose helpers).

PETRA produces all four mechanically. The scenario test pins
the load-bearing assertions on each.

## Files

- `scenario.toml` — net structure, guards, trace data,
  training hyperparameters (same shape as
  `credit_approval_coloured`).
- `../../tests/scenarios/test_regulator_ready_credit_approval.py`
  — pins each diagnostic output with assertions on its content
  and the prose generated from it.

## Running

```
python -m pytest tests/scenarios/test_regulator_ready_credit_approval.py
```
