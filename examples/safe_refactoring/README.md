# Safe refactoring — proving structurally-different variants behaviourally equivalent

Two loan-approval process variants that differ in *shape* (not
just in cost weights) — and PETRA proves them equivalent
anyway, then compares their soft-routing across the input
domain and reports confidence intervals on the rules each one
learned from the same training data.

This scenario complements `cost_ranked_refactoring/`. There,
both variants shared the same topology and differed only in
per-transition cost weights — the bisimulation check was
structurally trivial. Here the variants have a genuinely
different *shape*: Variant B contains an extra silent (τ)
audit-log transition and an extra place that Variant A does
not. **Strong bisimulation rejects them as different** —
because the LTS has an additional state and edge — while
**weak bisimulation accepts them as equivalent**, because τ
transitions are by definition not observable.

That is the load-bearing case for "refactor with confidence"
in practice. Real refactorings usually *do* change the
structure (extracting an audit step, factoring out a helper,
splitting a transition for instrumentation). Strong bisim is
too tight for those cases; weak bisim is the right tool.

## What this scenario demonstrates

End-to-end across three Phase 11 + Phase 13 capabilities:

1. **Weak bisimulation** (Phase 11) — `are_weakly_bisimilar`
   collapses the τ audit-log step and recognises Variant B as
   equivalent to Variant A. The companion `are_bisimilar` call
   correctly rejects them as different, pinning the distinction
   between the two checkers.
2. **Cross-variant comparison** (Phase 13) — `compare_variants`
   sweeps `credit_score` across [0, 1] and reports per-transition
   firing-decision agreement between the two trained variants.
   They agree at essentially every grid point on the routing
   transitions — same threshold, same direction, same activation
   shape.
3. **Bootstrap confidence intervals on the distilled rule**
   (Phase 13) — `bootstrap_xor_rule` resamples the trace list,
   retrains each variant per resample, and reports the
   distribution of crossover thresholds. Both variants' rules
   land in overlapping CIs at the empirical decision boundary
   around credit_score = 0.5.

## What's in this scenario's TOML vs in the test code

`scenario.toml` carries **Variant A** — the unrefactored
baseline. The test code builds **Variant B** by augmenting
Variant A's topology with a silent audit-log transition (the
same pattern the cost-ranked-refactoring test uses for its
second variant). The training data and hyperparameters are
shared between the two variants — both train on the same
twelve traces with the same seed.

## The take-away

The combination *(structurally different) + (provably
equivalent under weak bisim) + (soft routing agrees across
the input domain) + (rules align with overlapping CIs)* is
the formal answer to "did the refactoring change anything?"
that today's process-redesign work cannot give. Today's
answer is qualitative consulting or production shadow-running.
This scenario gives it mechanically, in under a second.

## Files

- `scenario.toml` — Variant A net, shared trace data, training
  hyperparameters.
- `../../tests/scenarios/test_safe_refactoring.py` — pins
  strong bisim rejection, weak bisim acceptance, cross-variant
  comparison, and the bootstrap CIs.

## Running

```
python -m pytest tests/scenarios/test_safe_refactoring.py
```
