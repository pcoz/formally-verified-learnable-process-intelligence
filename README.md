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
class is much larger than it sounds. The included scenarios
demonstrate it on:

- **Business processes** (loan approval, with cost-ranked variant
  refactoring)
- **Distributed-system protocols** (two-phase commit; TCP 3-way
  handshake with attack-pattern anomaly detection)
- **Multi-agent coordination** (contract-net with bid-driven
  contractor selection)
- **Manufacturing and supply-chain workflows** (multi-station
  production line with quality-gated routing)
- **Laboratory and clinical protocols** (PCR with deviation analysis)
- **Cell-biology signalling pathways** (kinase cascade with
  attribute-conditioned pathway routing)

The framework also covers (not built as scenarios but covered by the
same primitives): state machines in embedded software, regulatory
and compliance workflows, games with bounded state, contract /
treaty / agreement workflows, scientific data pipelines, RPA scripts.

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
python -m pytest                          # full suite (~210 tests)
python -m pytest tests/scenarios/         # only end-to-end scenarios
python -m pytest tests/test_compiler.py   # only the compiler
```
