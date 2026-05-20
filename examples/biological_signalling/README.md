# Biological signalling cascade

A kinase signalling cascade modelled as a Petri net — PETRA running
on a biological pathway. Pathway databases such as Reactome are
essentially Petri nets, so the same compilation, training, rule
extraction and anomaly detection apply directly.

## What this scenario shows

- The substrate accepts hand-coded non-BPMN nets (no BPMN parser
  involved).
- A multi-step biological cascade with conditional routing trains
  via the standard `train_on_traces` loop.
- Phase 8 interpretability extracts the routing rule in *biological*
  language ("fast pathway" vs "slow pathway") without any change.
- §7.2 anomaly detection flags off-pathway traces with residuals
  concentrated on the diverging pathway transitions.
- Phase 2 bisimulation works on the biological substrate identically
  to BPMN.

## Advantage over alternatives

- **vs. ODE-based pathway models** (e.g. COPASI, BioModels SBML
  simulations): ODE models require hand-fitted rate constants. This
  framework learns the routing dynamics from observed cellular
  responses, so the parameters reflect data instead of stipulated
  kinetics.
- **vs. Boolean network models** of signalling: Boolean models
  cannot encode graded signal-strength → response relationships.
  The continuous relaxation here does.
- **vs. black-box ML** trained on pathway data: black-box models
  can predict response but cannot explain *which pathway decision*
  changed. Phase 8 interpretability returns the rule "if signal >
  X → fast pathway, else slow", pinned to the literature names of
  the pathway components.
- **Bisimulation across model variants** (Phase 2): two structurally
  different cascade diagrams of the same pathway can be verified
  equivalent before training — useful when biologists publish
  competing diagrams of the same underlying biology.

## Real-world source

The cascade structure follows the canonical MAPK
(Mitogen-Activated Protein Kinase) cascade described in cell-signalling
literature: a receptor activates an upstream kinase, which can either
trigger a fast direct effector or proceed through an intermediate
kinase for a slow effector response. The fast/slow routing is
signal-strength dependent (a recurring motif in real signalling
networks).

> **See also** [`mapk_pathway/`](../mapk_pathway/) — the same
> family of biology loaded directly from a Pathway Commons SIF
> file (real Reactome entity names, real interaction-type
> vocabulary) rather than hand-coded. The two scenarios cover the
> two ends of the workflow: structurally-trainable hand-coded
> cascade here vs. format-driven import there.

## Files

- `scenario.toml` — full scenario specification: net topology, inline
  traces, training params, interpretability toggles
- `../../tests/test_adapter.py` — end-to-end test driving this
  scenario via `load_scenario`
- `../../tests/test_non_bpmn_substrate.py` — hand-coded variant of
  the same scenario, kept for direct-API regression coverage

## Running

```
python -m pytest tests/test_adapter.py
```

or programmatically:

```python
from petri_net_nn import load_scenario
ctx = load_scenario("examples/biological_signalling/scenario.toml")
module, losses = ctx.train()
rules = ctx.extract_rules(module)
```
