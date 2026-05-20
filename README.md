# PETRA

**Petri-Net Trained Architecture** — *formally-verified learnable
process intelligence*.

Give PETRA a Petri-net structure and a log of executions over it.
PETRA trains a model of how that system actually behaves, distills
the model's routing decisions into readable rules, scores new
executions for anomalies pinned to specific named elements, proves
when two structures are behaviourally equivalent, and ranks
equivalent variants by realised-execution cost.

> If your problem can be expressed as a finite-state, terminating,
> discrete-event system and you have observable traces of multiple
> instances, this framework can: **learn its dynamics, verify
> equivalence to other variants, detect anomalies pinned to named
> elements, and rank refactorings by cost-to-completion**. That class
> is large.

---

## What the framework learns the dynamics of

Any **finite-state, terminating, discrete-event system** for which
you have **observable execution traces of multiple instances**. That
class is much larger than it sounds — it covers business processes,
distributed-system protocols, manufacturing lines, laboratory
recipes, signalling pathways, multi-agent coordination, IT incident
management, and anything else whose state changes at identifiable
moments and whose runs you can record. The 13 scenarios shipped in
`examples/` make this concrete:

- **Business processes.** Loan approval with cost-ranked variant
  refactoring (provably-equivalent variants ranked by realised
  execution cost); credit approval where the per-application amount
  travels with the token and the approve/decline threshold is
  *learned from data* rather than hand-set; real IT incident
  management trained on the BPI Challenge 2013 dataset (7,554 Volvo
  IT tickets, the actual public log).
- **Distributed-system protocols.** Two-phase commit with Byzantine
  commit-after-low-vote anomaly detection; TCP 3-way handshake
  compiled from the RFC, with SYN-flood and half-open attack
  patterns flagged as anomalies.
- **Multi-agent coordination.** Three-pool contract-net with
  bid-driven contractor selection; pre-bid award flagged as a
  protocol violation.
- **Manufacturing and supply-chain workflows.** Multi-station
  production line with quality-gated rework routing; paint-shop
  cure step modelled as a multi-step transition duration;
  bottle-to-crate batching via multi-token arc multiplicities.
- **Operational coordination patterns.** Priority-driven dispatch
  across three handlers with declared rate priors that training
  refines; mutex on a shared resource enforced via inhibitor arcs.
- **Laboratory and clinical protocols.** PCR with deviation
  analysis flagging skipped quality gates.
- **Cell-biology signalling pathways.** Kinase cascade with
  strength-conditioned fast/slow pathway routing, with the routing
  rule distilled in pathway vocabulary.

The same primitives cover more ground than the shipped scenarios
exercise: state machines in embedded software, regulatory and
compliance workflows, games with bounded state, contract / treaty
/ agreement workflows, scientific data pipelines, RPA scripts. If
your problem fits the four properties in the next section, the
substrate fits it too.

## Where PETRA fits well

PETRA works best on systems with all four properties below:

- **Discrete events** — state changes at identifiable moments (a
  transition firing), rather than continuously over time. Workflows,
  protocols, recipes, state machines.
- **Multiple-instance trace data** — you have many recorded runs of
  the system to learn from. One run isn't enough.
- **Stable structure for the duration of training** — the
  place/transition graph stays fixed while learning the dynamics
  within it.
- **Tractably finite state space** — small enough that the compiled
  Petri net fits in memory and trains in reasonable time. This is
  generous in practice (thousands of places / transitions work fine)
  but rules out problems that need an entire economy or the global
  internet at full resolution.

Fluid dynamics, classical mechanics, analogue control, and
similar continuous-time / continuous-state physics need a different
substrate — Petri nets are discrete by design.

## What PETRA buys you

PETRA combines a fixed verified topology with learned dynamics
within it. That gives you four capabilities together:

- **Interpretability** at the granularity of named domain elements
  (BPMN tasks, biological pathway components, protocol states).
- **Formal equivalence checks** between two models via strong
  bisimulation, before either is deployed.
- **Anomaly detection** with residuals pinned to specific transitions
  rather than opaque whole-trace scores.
- **Cost-ranked refactoring** — provably-equivalent variants compared
  by realised-execution cost on the trained firing distribution.

PETRA's shape fits problems with explicit place/transition
structure; arbitrary sequence modelling fits something else.

---

## What makes this approach unusual

| Capability | Most ML | Classical Petri-net analysis | This framework |
|---|---|---|---|
| Learns from execution traces | ✓ | ✗ | ✓ |
| Preserves verified structure | ✗ | ✓ | ✓ |
| Bisimulation-based equivalence | ✗ | partial | ✓ |
| Interpretable at named elements | ✗ | n/a (no learning) | ✓ |
| Detects structurally-grounded anomalies | weak | ✗ | ✓ |
| Ranks behaviour-preserving variants by cost | ✗ | ✗ | ✓ |

The bisimulation + cost-ranking combination is what makes
**provably-safe process refactoring** possible: refactor a process,
prove the new version is behaviourally equivalent to the old one,
then rank the variants by realised-execution cost. Nobody else has
that running with tests.

---

## How PETRA fits with classical Petri-net tools

PETRA is **complementary**, not competitive, to the established
Petri-net tool ecosystem. Each classical tool answers a different
question over the same Petri-net structure:

| Tool | What it's best at | Where PETRA differs |
|---|---|---|
| **CPN Tools** (Aarhus) | Reference implementation of Coloured Petri Nets — full ML-style colour-set typing, state-space verification, mature GUI simulator. | CPN Tools verifies a *given* CPN; PETRA *trains* a model of how the net's transitions are actually used from execution traces, including learning guard thresholds from per-token values rather than taking them as given. CPN Tools' colour sets are far richer than PETRA's CPN-lite scalar token values. |
| **GreatSPN** (Turin) | Generalised Stochastic Petri Nets — exponentially-distributed firing times, analytical CTMC throughput, performance bounds. | GreatSPN gives closed-form stationary throughput under a stipulated rate model; PETRA's stochastic rates are compiler-level multipliers used during training. Different question. |
| **TINA** (LAAS-CNRS) | Time Petri nets with intervals, state-space exploration, integrated CTL/LTL model checking via NuSMV. | TINA proves temporal-logic invariants about the *specified* behaviour; PETRA learns how the deployed system actually behaves and flags deviations. Phase 11 of the PETRA roadmap aims to wire model checking in directly. |
| **ProM** (Eindhoven) | Process mining — Alpha / Inductive / Heuristics miners discover a Petri net from execution logs; conformance checking; large plugin ecosystem. | ProM does *structure discovery* from logs (Phase 12 of PETRA's roadmap, not yet built). The two are a natural pair: ProM discovers, PETRA trains dynamics on the result. |

**The thing PETRA does that none of them do:** combine a
*learned-from-traces* dynamics model with a *structurally verified*
Petri-net substrate, then extract interpretable rules from the
learned weights and rank behaviour-preserving refactorings by cost.
None of those four tools touch any of those four capabilities.

### A complementary analysis stack

The five tools naturally compose end-to-end on the same model:

> **ProM** discovers the structure from logs → **CPN Tools**
> verifies its soundness → **GreatSPN** gives stochastic throughput
> → **TINA** proves temporal invariants → **PETRA** learns the
> dynamics that actually occur in production, distills the routing
> rules, detects deviations, and ranks refactorings.

PNML support is the bridge that makes this stack possible — any of
those tools' output can now feed straight into PETRA. That's why
PNML is high-leverage despite being only a few hundred lines of
code: it converts PETRA from a standalone Python library into an
ecosystem citizen, one PNML file away from any of the above.

---

## Quick start

```python
from petri_net_nn import load_scenario

# Each example/ subfolder contains a self-contained scenario as
# a TOML config plus an explanatory README.
ctx = load_scenario("examples/cost_ranked_refactoring/scenario.toml")
module, losses = ctx.train()
rules = ctx.extract_rules(module)
print(rules["xor"][0].description())
```

For the framework-level API (build a `PetriNet` by hand, compile,
train, extract rules, score anomalies), see
[`docs/DEV_MANUAL.md`](docs/DEV_MANUAL.md).

---

## Repository layout

```
petri_net_nn/         # the framework
  petri_net.py        # PetriNet dataclass + token-game semantics
  bpmn.py             # BPMN 2.0 → PetriNet parser
  compiler.py         # PetriNet → differentiable nn.Module (§4 of spec)
  subnets.py          # five hand-built reference subnets (§5)
  traces.py           # training, anomaly score, expected-cost, AUC
  xes.py              # IEEE XES log loader
  anomalies.py        # corruption generators + frequency baseline
  interpretability.py # distil learned weights into rules
  bisimulation.py     # strong-bisimulation equivalence checker (§7.3)
  adapter.py          # config-driven scenario loader

examples/             # seven validated end-to-end scenarios
  biological_signalling/
  distributed_consensus/
  manufacturing_cell/
  network_protocol/
  scientific_workflow/
  cost_ranked_refactoring/
  multi_agent_coordination/

tests/                # passing tests, framework + scenarios
docs/                 # all longer-form documentation
  ROADMAP.md          # product roadmap + framing + scenarios table
  DEV_MANUAL.md       # framework + adapter usage guide
```

---

## Reading order for newcomers

1. This README — what PETRA is
2. [`docs/ROADMAP.md`](docs/ROADMAP.md) — framing, scenarios
   delivered, what's next
3. Any [`examples/*/README.md`](examples/) — concrete scenario in
   your domain
4. [`docs/DEV_MANUAL.md`](docs/DEV_MANUAL.md) — adapter config plus
   framework API reference

---

## Running tests

```
python -m pytest                          # full suite (~295 tests)
python -m pytest tests/scenarios/         # only end-to-end scenarios
python -m pytest tests/test_compiler.py   # only the compiler
```
