# Cost-ranked variant search

The point #6 demonstration: two implementation variants of a loan
approval process, provably behaviourally equivalent (via Phase 2
bisimulation), trained on the same trace data, with per-variant cost
weights yielding an expected-cost-to-completion ranking. **This is
the provably-safe-refactoring capability the framing claims.**

## What this scenario shows

- Two variants of an approval process — *In-house* (high
  human-effort) and *Automated* (programmatic). Same observable
  behaviour, different per-step costs.
- Phase 2 bisimulation: `are_bisimilar(variant_a_net,
  variant_b_net)` returns True. The behavioural equivalence is
  *proven*, not assumed.
- Same trace data trains both variants to the same routing rule
  (credit_score > X → approve, else decline). The functional output
  on any input is identical across variants.
- Per-variant cost weights are attached after training. The
  `expected_cost` helper multiplies each transition's activation
  by its cost weight and sums. Variant B (automated) comes out at
  ~6× cheaper than Variant A (in-house) on the trained routing
  distribution.
- The cost ranking is *attributable only to the cost-weight
  difference*, because the two variants implement the same routing
  function. This is what enables refactor-with-confidence.

## Advantage over alternatives

- **vs. heuristic process re-engineering**: today's process redesign
  is rolled out via consulting engagements that argue qualitatively
  that variant B is better. This framework lets you *prove*
  variant B is equivalent before deployment and *quantify* the cost
  difference from actual execution data.
- **vs. A/B testing in production**: shadow-running two process
  variants in production for weeks/months is expensive and slow.
  Here the equivalence proof is mechanical (Phase 2 — runs in
  milliseconds) and the cost comparison can be done on historical
  XES logs offline.
- **vs. Discrete-event simulation packages** (Arena, AnyLogic,
  SimPy): simulators run scenarios but cannot prove two simulators
  describe equivalent processes. They also rely on stipulated
  arrival/service distributions; this framework uses the *learned*
  distribution from real traces.
- **vs. monolithic ML**: monolithic models cannot guarantee that a
  refactored process produces the same outputs as the original.
  Without that guarantee, cost rankings are uninterpretable —
  cheaper might mean "different process, lower cost" rather than
  "same process, lower cost".

## Real-world source

The loan-approval process structure is faithful to BPMN models
widely used in retail banking and process-mining benchmarks:

- van der Aalst, *Process Mining* (2nd ed., 2016) — uses
  loan-application processes throughout as canonical examples.
- BPI Challenge 2012 (Dutch financial institution loan
  applications, hosted at 4TU.ResearchData) — same conceptual
  structure: submitted → triage → assessment → approve/decline.

The cost weights in `scenario.toml` are illustrative of the
in-house vs automated comparison common in process-automation
business cases.

## Files

- `scenario.toml` — net structure, traces, training params, AND
  per-variant cost weights under `[cost_ranking.*]`
- `../../tests/scenarios/test_cost_ranked_refactoring.py` — the
  end-to-end demonstration test

## Running

```
python -m pytest tests/scenarios/test_cost_ranked_refactoring.py
```
