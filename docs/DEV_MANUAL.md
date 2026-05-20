# PETRA Developer Manual

How to use **PETRA** (Petri-Net Trained Architecture) — both the
high-level adapter (TOML config → trained model) and the underlying
framework modules. Treat `ROADMAP.md` as the *why*, this document as
the *how*. Update this manual whenever a new module, scenario, or
extension point is added.

---

## 1. Quick start

The fastest path: write a TOML config and use the adapter.

```python
from petri_net_nn import load_scenario

ctx = load_scenario("examples/biological_signalling/scenario.toml")
module, losses = ctx.train()
rules = ctx.extract_rules(module)
print(rules["xor"][0].description())
```

A scenario is a TOML file describing the net, traces, and training
parameters. The adapter resolves it into a `ScenarioContext` and the
context exposes `compile`, `train`, `attribute_to_marking`,
`extract_rules`, and `anomaly_score`. No per-scenario Python boilerplate
is needed.

---

## 2. The adapter

### 2.1 Config schema (TOML)

```toml
[scenario]
name = "..."             # human-readable identifier
description = "..."      # optional

[net]
source = "inline"        # or "bpmn_file"
# ---- inline form ----
[[net.places]]
id = "p_x"
tokens = 1               # initial-marking tokens (default 0)
label = "..."            # optional human label
[[net.transitions]]
id = "t_x"
label = "..."
[[net.arcs]]
src = "p_x"
dst = "t_x"
# ---- OR: bpmn_file form ----
# path = "process.bpmn"  # relative to the config file

[traces]
source = "inline"        # or "xes_file"
# ---- inline ----
[[traces.inline]]
attributes = { signal = "0.9" }
events = ["task_a", "task_b"]
# ---- OR ----
# path = "traces.xes"

[training.input_marking]
# Each key is a place id; value is either:
p_x = { attribute = "signal" }   # take the trace's "signal" attribute
p_y = { constant = 0.5 }         # pin to a constant

[training.input_values]
# Optional. Coloured-Petri-net value channel — same spec form as
# input_marking but feeds the per-token value the compiler reads
# through structural guards (see §3.3).
p_x = { attribute = "amount" }

[training]
steps = 1500
lr = 0.1
sharpness = 1.0
firing = "sigmoid"               # or "ste"
routing = "independent"          # or "softmax"
num_steps = 0                    # 0 = acyclic single-pass, >0 = time-unrolled
seed = 0                         # torch.manual_seed before module construction

[interpretability]
extract_xor_rules = true
extract_and_join_rules = false
```

### 2.2 `ScenarioContext` methods

| Method | Returns | What it does |
|---|---|---|
| `compile()` | `PetriNetModule` | Build the module with config training params; seeds torch first. |
| `train(module=None)` | `(PetriNetModule, list[float])` | Compile if needed and run `train_on_traces`. |
| `attribute_to_marking(trace)` | `dict[str, float]` | Resolve the config's input-marking spec for a single trace. |
| `attribute_to_values(trace)` | `dict[str, float]` | Resolve the config's `[training.input_values]` spec — the coloured-token value channel the compiler reads through structural guards. |
| `extract_rules(module)` | `dict[str, list]` | Run XOR / AND-join rule extraction per `[interpretability]` toggles. |
| `anomaly_score(module, trace)` | `dict[str, float]` | Per-transition residuals for one trace. |

Carrying the raw `config` dict (and `config_dir` `Path`) on the context
lets downstream callers read scenario-specific extras (e.g. anomaly
generator settings) without re-parsing the file.

---

## 3. Framework reference

When the adapter doesn't fit (e.g. you want to manipulate the net
structurally before compilation), use the underlying modules directly.

### 3.1 `petri_net.py` — the formal object

```python
from petri_net_nn import PetriNet

net = PetriNet()
net.add_place("p_a", tokens=1, label="start")
net.add_place("p_b")
net.add_transition("t_x", label="do thing")
net.add_arc("p_a", "t_x")
net.add_arc("t_x", "p_b")
```

Other methods: `preset`, `postset`, `is_enabled`, `fire`,
`enabled_transitions`, `validate`. Validation returns a list of
structural issues (empty list = well-formed).

### 3.2 `bpmn.py` — BPMN 2.0 → PetriNet

```python
from petri_net_nn import parse_bpmn
net = parse_bpmn("process.bpmn")
```

Supports: tasks, gateways, start/end events, compensation boundary
events (acyclic and throw-event forms), error/timer/signal/escalation/
message boundary events (interrupting and non-interrupting),
intermediate events, lanes, multi-pool `<collaboration>` with
`<messageFlow>`.

Does not support: subprocesses, intermediate event definitions
(timer/message/signal as intermediate, not boundary), message flows
to non-task nodes.

### 3.3 `compiler.py` — PetriNet → differentiable nn.Module

```python
from petri_net_nn import PetriNetModule
module = PetriNetModule(
    net,
    sharpness=1.0,           # inside-sigmoid multiplier
    num_steps=0,             # 0 = acyclic; >0 = time-unrolled
    firing="sigmoid",        # or "ste" (straight-through estimator)
    routing="independent",   # or "softmax" (over XOR groups)
)
out = module(input_marking={"p_a": torch.tensor([0.9])})
```

The compiler's structural constraint is enforced by construction:
one `nn.Parameter` per arc in F, one per transition (threshold). No
weights exist outside the flow relation.

The forward pass implements a continuous relaxation of the discrete
firing rule. For each transition:

    activation(t) = σ( sharpness · ( Σ_p w(p,t) · a(p) − θ(t) ) )

and for each downstream place:

    a(p) = Σ_{t : (t,p) ∈ F}  activation(t) · w(t,p)

That makes the whole network differentiable end to end — standard
backpropagation applies. ``firing="ste"`` swaps the sigmoid for a
hard step in the forward pass while keeping the sigmoid gradient
flowing backward (the standard straight-through estimator);
``routing="softmax"`` replaces independent sigmoids over an XOR
group with a softmax that sums to 1; inhibitor arcs multiply the
resulting activation by ``(1 − a(p))`` for each inhibitor place;
transition durations buffer the activation for D−1 time-unrolled
steps before it contributes to downstream places.

**Coloured-Petri-net layer.** When a transition has a structural
guard declared as ``{place, op, value}`` (TOML form, or via
``add_transition(..., structural_guard=...)``), the compiler builds
one learnable ``nn.Parameter`` threshold per guarded transition —
seeded at the TOML value, refined by training. A soft sigmoid
gate multiplies the transition's firing strength:

    soft_guard(t) = σ( sharpness · scale(t) · sign(op) · ( value(place) − θ_guard(t) ) )

with ``sign(op) = +1`` for ``>``/``>=`` and ``−1`` for ``<``/``<=``,
and ``scale(t) = 1 / max(|θ_init|, 1.0)`` so the sigmoid's gradient
is O(1) at the boundary regardless of the value units the modeller
used. The forward pass carries a parallel per-place *value*
channel alongside activations: source-place values come from the
new ``input_values`` argument (default 1.0); non-source places get
an activation-weighted average of contributing transitions'
output-arc values (``arc_output_values`` constants only — callable
transforms stay token-game-only). Equality / inequality guards
must be expressed as opaque callables and are not trainable; only
inequality guards take part in training. The thresholds train
end-to-end with the rest of the network, so the model can refine
the declared boundary from execution traces — see the
`credit_approval_coloured` scenario.

### 3.4 `traces.py` — training and anomaly scoring

```python
from petri_net_nn import train_on_traces, anomaly_score, SharpnessScheduler, sweep_trace_count
```

- `train_on_traces(module, traces, *, attribute_to_marking, attribute_to_values=None, steps, lr, transitions=None)` — main training loop. Pass `attribute_to_values` to feed the per-token value channel that trains structural-guard thresholds.
- `anomaly_score(module, trace, *, attribute_to_marking, attribute_to_values=None)` — per-transition residual dict.
- `trace_anomaly_score(module, trace, ...)` — scalar trace-level score.
- `auc(positive_scores, negative_scores)` — Mann-Whitney U / ROC AUC.
- `SharpnessScheduler(module, *, start, end, num_steps, kind)` — anneal sharpness over training.
- `sweep_trace_count(factory, traces, *, attribute_to_marking, sample_sizes, steps, lr)` — characterise training-data requirements.
- `expected_cost(module, transition_costs, *, input_marking, batch_size)` — sum of (activation × cost) per transition. Used for cost-ranked variant search (point #6).

### 3.5 `xes.py` — XES log loader

```python
from petri_net_nn import parse_xes
traces = parse_xes("log.xes")
```

Returns `list[XESTrace]`. Each trace has `attributes: dict[str, str]`
and `events: list[XESEvent]` where each event has `name: str`
(concept:name).

### 3.6 `anomalies.py` — corruption generators and baselines

```python
from petri_net_nn import drop_event, insert_event, swap_event_labels, shuffle_events, FrequencyBaseline
```

Generators take a normal trace and return a corrupted copy
(non-mutating). `FrequencyBaseline().fit(traces).score(trace)` is the
non-structural baseline used for Phase 7 comparison.

### 3.7 `interpretability.py` — distilled rules

```python
from petri_net_nn import (
    find_xor_groups,            # structural XOR detection (single-input or shared-preset)
    extract_xor_rule,           # binary routing rule (crossover threshold)
    extract_routing_rules,      # all binary XOR groups in a module
    extract_xor_partition,      # N-way piecewise routing
    extract_routing_partitions, # all groups, any arity
    find_and_join_transitions,
    extract_and_join_rule,      # weighted-vote / quorum rule
    extract_and_join_rules,
    explain_anomaly,            # prose explanation pinned to BPMN labels
)
```

For shared-preset XOR groups (multi-input competing transitions, e.g.
2PC commit/abort), the rule extractors automatically pick the
*discriminative* input — the place whose learned weight gap across
the group is largest.

### 3.8 `bisimulation.py` — formal equivalence checking

```python
from petri_net_nn import are_bisimilar, bisimulation_equivalence_classes, reachability_graph
```

Strong bisimulation on the labelled transition system via partition
refinement. `are_bisimilar(net1, net2)` returns `True` iff the two
nets exhibit identical labelled future behaviour from their initial
markings.

### 3.9 `subnets.py` — hand-built reference subnets

`SequentialSubnet`, `XORSubnet`, `AndSplitSubnet`, `AndJoinSubnet`,
`SagaSubnet` — five `nn.Module` subclasses corresponding to the
canonical workflow-net building blocks. The general
`PetriNetModule` subsumes them; the hand-built versions stay as
readable references and regression coverage.

---

## 4. Examples folder

```
examples/
  <scenario_name>/
    scenario.toml      # the config
    [data files]       # optional BPMN or XES
    README.md          # what + why + advantage over alternatives
```

Each scenario has a paired test in `tests/scenarios/test_<name>.py`
that loads the config via `load_scenario` and asserts the pipeline
produces the expected behaviour. `pytest tests/scenarios/` runs all
scenario tests.

Current scenarios:

| Scenario | Framing claim |
|---|---|
| `biological_signalling/` | Non-BPMN substrate covers signalling pathways. |
| `distributed_consensus/` | Cross-pool composition covers distributed protocols (2PC). |
| `manufacturing_cell/` | Sequential + XOR primitives cover production lines. |
| `network_protocol/` | Substrate covers protocol state machines; attack-pattern anomaly detection. |
| `scientific_workflow/` | Substrate covers lab protocols; deviation analysis. |
| `multi_agent_coordination/` | Three-pool composition covers contract-net coordination. |
| `batch_packaging/` | Phase 9 multi-token markings: arc weight 6 batches bottles into crates. |
| `resource_lock/` | Phase 9 inhibitor arcs: two clients race for a single shared resource. |
| `paint_shop/` | Phase 9 transition durations: a 3-step cure transition delays output. |
| `priority_dispatch/` | Phase 9 stochastic firing rates: three handlers with rate priors. |
| `credit_approval_coloured/` | Phase 9 coloured tokens plus the CPN-aware compiler: routing on the application amount carried by the token, with the guard threshold learned from trace data. |
| `incident_management/` | Phase 10 — trains on the **real BPI Challenge 2013** incidents log (7,554 Volvo IT tickets), the actual public dataset. |
| `cost_ranked_refactoring/` | Provably-safe refactoring via Phase 2 + `expected_cost`. |

---

## 5. Adding a new scenario

1. `mkdir examples/<name>` and create `scenario.toml`.
2. Decide on net source — inline TOML (small, self-contained) or
   `bpmn_file` pointing at a `.bpmn` next to the config.
3. Provide traces — inline if small, `xes_file` if downloaded from
   a real-data repo (e.g. BPI Challenge logs).
4. Set up `[training.input_marking]` to map the attribute(s) that
   drive routing decisions onto the relevant place(s).
5. Add `README.md` covering:
   - **What this scenario shows** (which framing claims it
     validates)
   - **Advantage over alternatives** (what existing tools cannot do
     that the framework can)
   - **Real-world source** (RFC / paper / dataset)
   - **Files** and how to run
6. Write `tests/scenarios/test_<name>.py` driving the scenario
   through `load_scenario` and asserting at minimum: loads,
   training converges, anomaly detection separates normal vs
   anomalous traces.
7. Update this manual's scenario table and ROADMAP.md if the
   scenario reveals a framework gap.

If you hit a framework shortcoming, **fix it in the framework first**
rather than working around it in the scenario. Phase 2-8 of the
ROADMAP exists precisely so the scenarios can be config-only.

---

## 6. Extension points

Places to plug new behaviour into the framework:

- **New BPMN constructs** → `bpmn.py` (one branch per construct in
  the main parsing loop). Track gaps in the Phase 4 carry-forward
  section of ROADMAP.md.
- **New firing modes** → `compiler.py`, add a function with the
  signature `(pre_activation) -> activation` and a string key in
  the `FiringMode` Literal.
- **New routing modes** → `compiler.py`, extend the `RoutingMode`
  Literal and add the dispatch in `_forward_acyclic` /
  `_forward_unrolled`.
- **New anomaly generators** → `anomalies.py`.
- **New rule shapes** → `interpretability.py`, follow the pattern
  of `XORRule` / `AndJoinRule`.
- **New data formats** → `adapter.py`, add a `source = "..."`
  handler in `_load_net` or `_load_traces`.

Tests live alongside each module: `tests/test_<module>.py` for
unit-level coverage, `tests/scenarios/test_<scenario>.py` for
end-to-end demos.

---

## 7. Where this work sits in the literature

PETRA draws on and combines four research threads:

- **Workflow nets** — van der Aalst's foundational work on sound
  workflow nets as a Petri-net subclass. PETRA depends on workflow-
  net soundness for the reachability / liveness / boundedness
  properties that make training meaningful.
- **Graph neural networks for process mining** — work applying
  GNNs to process graphs for conformance checking and anomaly
  detection (Tax et al., Bukhsh et al.). PETRA is more structurally
  constrained than a general GNN: the architecture IS the verified
  workflow net, not learned from process data.
- **Neuro-symbolic AI** — Scallop, DeepProbLog and similar systems
  that combine neural and symbolic reasoning. PETRA is a specific
  instance where the symbolic substrate is a compiler-verified
  workflow net rather than a logic program.
- **Spiking neural networks** — networks that model discrete spike
  propagation rather than continuous activations. The token-firing
  model is analogous to spike propagation, and SNN training methods
  (STDP, surrogate-gradient methods) are applicable to PETRA's
  discrete-firing limit and inform the Phase 6 STE work.

PETRA's novel contribution is the specific combination: a verified
workflow net used as a fixed neural-network architecture, where the
soundness properties propagate into the network's representational
constraints and the learned weights stay interpretable at the
granularity of named domain elements (BPMN tasks, pathway
components, protocol states, …).

---

## 8. Running everything

```
python -m pytest                          # full suite
python -m pytest tests/scenarios/         # only end-to-end scenarios
python -m pytest tests/test_compiler.py   # only the compiler
```

Current test count: 295 passing across the framework and the
end-to-end scenarios.
