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

The full surface of the adapter, organised by section. Every
field is optional unless flagged otherwise.

```toml
# ---------------------------------------------------------------------------
# [scenario] — identification metadata
[scenario]
name = "..."             # human-readable identifier (defaults to file stem)
description = "..."      # optional one-line description

# ---------------------------------------------------------------------------
# [net] — exactly one source per scenario
[net]
source = "inline"        # one of: "inline" | "bpmn_file" | "pnml_file" | "sif_file"

# ---- "inline" form: declare the net structurally in TOML ----
[[net.places]]
id = "p_x"               # required
tokens = 1               # initial-marking tokens (default 0)
label = "..."            # human label

[[net.transitions]]
id = "t_x"               # required
label = "..."
duration = 1             # firing duration in time-unrolled steps (default 1)
rate = 1.0               # firing-rate prior multiplier (default 1.0)
guard = { place = "p_x", op = ">=", value = 1000.0 }   # CPN structural guard
# silent = true          # mark as τ for weak bisimulation

[[net.arcs]]
src = "p_x"
dst = "t_x"
weight = 1               # arc multiplicity (default 1, >1 for batching)
output_value = 1.0       # CPN output value (constant only via TOML)

# Inhibitor arcs — place must be empty for the transition to fire
[[net.inhibitor_arcs]]
place = "p_guard"
transition = "t_guarded"

# ---- OR: "bpmn_file" form — process.bpmn relative to the config ----
# path = "process.bpmn"

# ---- OR: "pnml_file" form — PNML 2009 P/T net subset ----
# path = "net.pnml"

# ---- OR: "sif_file" form — Pathway Commons SIF ----
# path = "pathway.sif"

# ---------------------------------------------------------------------------
# [traces] — exactly one source per scenario; section is optional if you
# only want to load the net structurally
[traces]
source = "xes_file"      # one of: "inline" | "xes_file" | "csv_file" | "json_file"

# ---- "xes_file" form ----
path = "log.xes"                          # or .xes.gz (parser handles gzip)
limit_traces = 300                        # cap for large public XES logs
promote_event_attrs = ["impact"]          # lift event attrs to trace level
event_name_attr = "lifecycle:transition"  # use a different attr as event name

# ---- "csv_file" form (process-mining flat table) ----
# path = "log.csv"
# case_column = "case_id"
# activity_column = "activity"

# ---- "json_file" form ----
# path = "log.json"

# ---- "inline" form ----
# [[traces.inline]]
# attributes = { signal = "0.9" }
# events = ["task_a", "task_b"]

# ---------------------------------------------------------------------------
# [training.input_marking] — required when traces are present.
# Each key is a place id; value is { attribute = "name" } (read from the
# trace's attributes dict) or { constant = N }.
[training.input_marking]
p_x = { attribute = "signal" }
p_y = { constant = 0.5 }

# ---------------------------------------------------------------------------
# [training.input_values] — coloured-Petri-net value channel; same form
# as input_marking but feeds the per-token value the compiler reads
# through structural guards (see §3.3).
[training.input_values]
p_x = { attribute = "amount" }

# ---------------------------------------------------------------------------
# [training] — training hyperparameters
[training]
steps = 1500
lr = 0.1
sharpness = 1.0
firing = "sigmoid"               # or "ste"
routing = "independent"          # or "softmax"
num_steps = 0                    # 0 = acyclic single-pass, >0 = time-unrolled
seed = 0                         # torch.manual_seed before module construction

# ---------------------------------------------------------------------------
# [interpretability] — toggles for ctx.extract_rules()
[interpretability]
extract_xor_rules = true
extract_and_join_rules = false
```

Anything not listed above (per-transition torch guards, custom arc
output transforms, etc.) lives in the Python API rather than TOML
— torch callables can't round-trip through a configuration file.

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

**Coloured-Petri-net layer.** Three ways to express value-dependent
behaviour, in order of expressiveness:

1. **Structural guard** — ``{place, op, value}`` declared in TOML or
   via ``add_transition(..., structural_guard=...)``. The compiler
   builds one learnable ``nn.Parameter`` threshold per guarded
   transition, *seeded at* ``value`` and *refined by* training,
   and multiplies the transition's firing strength by

       soft_guard(t) = σ( sharpness · scale(t) · sign(op) · ( value(place) − θ_guard(t) ) )

   with ``sign(op) = +1`` for ``>``/``>=`` and ``−1`` for
   ``<``/``<=``, and ``scale(t) = 1 / max(|θ_init|, 1.0)`` so the
   sigmoid's gradient is O(1) at the boundary regardless of the
   value units the modeller used. *Equality / inequality (``==``,
   ``!=``) cannot be expressed structurally and must use the
   torch-guard form below.* This is the case where you want PETRA
   to **learn the threshold from data** — see the
   `credit_approval_coloured` scenario.

2. **Torch guard** — a Python callable on the transition (kwarg
   ``torch_guard=...``) taking ``dict[place_id, Tensor(batch,)]``
   of input values and returning a ``Tensor(batch,)`` gate in
   [0, 1]. For routing logic the structural form can't express:
   multi-input comparisons, compound predicates, custom learnable
   sub-networks. Overrides the structural guard when both are
   declared on the same transition.

3. **Token-game-only callable guard** — ``guard=...``, a
   bool-returning ``GuardFn``. Used by ``fire_coloured`` /
   ``is_enabled_coloured``; the compiler ignores it.

Output-arc values follow the same tiered pattern, in compiler
precedence: ``torch_output_value`` (callable on bound input
value tensors, honoured by the compiler), then ``output_value``
(constant float honoured by the compiler, or float-returning
callable evaluated only by the token-game), then the default
1.0.

The forward pass carries a parallel per-place *value* channel
alongside activations: source-place values come from the
``input_values=...`` argument to ``forward`` (default 1.0);
non-source places get an activation-weighted average of
contributing transitions' output-arc values. This channel is
what the guard sigmoids read.

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

### 3.8 `soundness.py` — Aalst soundness + deadlock localisation

```python
from petri_net_nn import (
    SoundnessReport,
    check_soundness,
    find_deadlocks,
)
```

`check_soundness(net, *, final_marking=None)` returns a
`SoundnessReport` pinning three classical soundness conditions:

| Condition | Field on report | Failing case |
|---|---|---|
| Option to complete | `incomplete_markings` | reachable markings from which the intended final marking can't be reached |
| Proper completion | `lingering_token_markings` | sink reached its completion count but tokens still hang around at other places |
| No dead transitions | `dead_transitions` | transitions never enabled in any reachable marking |

`report.is_sound` is the boolean (all three lists empty);
`report.summary()` is a one-line digest suitable for log lines and
assertion messages. The default `final_marking` is "one token at
each sink place" — places with no outgoing arcs. Pass an explicit
`final_marking` when the sink is ambiguous (cyclic nets without a
clear terminus) or when completion has multiple sink tokens.

`find_deadlocks(net)` returns the *non-final* markings with no
enabled successors — the specific token configurations from which
the net can't progress. Overlaps with the option-to-complete
failure that `check_soundness` reports, but isolates root-cause
states (a state that can only reach a deadlock also fails
option-to-complete; the deadlock itself is the actionable root).

Both checks are structural — they analyse the Petri net's
behaviour without running training. Use them before training to
catch modeller bugs, after a refactoring to confirm the new
variant is still sound, or in CI to enforce soundness on every
scenario.

### 3.9 `ctl.py` — Computation Tree Logic model checking

```python
from petri_net_nn import (
    Atom, Not, And, Or, EX, EU, EG,
    AX, AG, AF, EF, AU,
    place_has_token, place_empty, place_count_eq, place_count_ge,
    transition_enabled,
    conj, disj, implies,
    check_ctl, satisfies, CTLResult,
)
```

Build a CTL formula by composing the AST classes, then ask the
checker whether the net's initial marking satisfies it:

```python
prop = AG(implies(
    place_has_token("request"),
    AF(place_has_token("response")),
))
result = check_ctl(net, prop)
result.holds_at_initial    # bool
result.holds_at            # frozenset[Marking] of satisfying states
result.counterexample      # a marking that violates the formula, or None
```

The AST primitives are `Atom`, `Not`, `And`, `Or`, `EX`, `EU`,
`EG`; the derived constructors `AX`, `AG`, `AF`, `EF`, `AU`
expand to combinations of the primitives via the standard CTL
equivalences (`AG φ ≡ ¬EF ¬φ`, etc.). Atomic propositions are
predicates over the current marking, expressed either via the
helpers (`place_has_token`, `place_empty`, `place_count_eq`,
`place_count_ge`, `transition_enabled`) or built directly with
`Atom(callable, label)` where the callable takes a `dict[str, int]`
marking and returns a bool.

Implementation note: implicit self-loops are added at every
deadlock state before the fixed-point computation — the standard
Kripke convention so that AG / AF / EG behave intuitively on
terminating workflow nets (without the self-loops, EG would be
vacuously false everywhere on a terminating net, and AX would be
vacuously true at the sink).

### 3.10 `bisimulation.py` — formal equivalence checking

```python
from petri_net_nn import (
    are_bisimilar,
    are_weakly_bisimilar,
    bisimulation_equivalence_classes,
    weak_bisimulation_equivalence_classes,
    reachability_graph,
)
```

Strong bisimulation on the labelled transition system via partition
refinement. `are_bisimilar(net1, net2)` returns `True` iff the two
nets exhibit identical labelled future behaviour from their initial
markings.

Weak bisimulation collapses transitions flagged ``silent=True``
on the net — internal logging hooks, no-op routing gates, internal
handoffs — before comparison. ``are_weakly_bisimilar(net1, net2)``
treats every silent transition as a τ-edge, saturates the
combined LTS so τ-paths become direct edges and visible-action
edges gain τ-prefix / τ-suffix detours, then runs strong
bisimulation on the saturated LTS. The net effect: refactorings
that add or remove internal-only structural artefacts no longer
break the equivalence claim, which is the case the cost-ranked
refactoring story relies on.

### 3.11 `subnets.py` — hand-built reference subnets

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
| `mapk_pathway/` | Phase 10 — loads a Pathway Commons-style SIF of the MAPK1/3 (ERK1/2) signalling cascade and runs a forward pass through the EGF → MAPK → transcription-factor flow. |
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

Current test count: 351 passing across the framework and the
end-to-end scenarios.
