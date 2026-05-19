# petri-net-dynamics

A research scaffold combining sound Petri-net substrates with neural-
network training. The structural constraint — a verified place /
transition graph — is what makes the framework able to learn dynamics
over a much wider class of systems than its BPMN framing implies,
with formal equivalence guarantees almost no other ML approach can
claim.

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

## What it cannot learn the dynamics of

The framework is honest about its limits:

- **Continuous-time / continuous-state systems** (fluid dynamics,
  classical mechanics, analogue control). Petri nets are discrete by
  design.
- **Truly one-shot novel systems** — needs a distribution over trace
  data; one execution isn't enough to learn from.
- **Systems whose own structure evolves faster than the training
  loop** — the topology is fixed during training.
- **Systems so large the Petri net is intractable** — state-space
  explosion is real. This is a scaffold, not a planetary simulator.

## What it is NOT

Not a general-purpose ML library, not an alternative to LSTMs or
Transformers for arbitrary sequence modelling. The value is
specifically the combination of *fixed verified topology* + *learned
dynamics within it*. That combination buys:

- **Interpretability** at the granularity of named domain elements
  (BPMN tasks, biological pathway components, protocol states)
- **Formal equivalence checks** between models via strong bisimulation
  before deployment
- **Anomaly detection** with residuals pinned to specific transitions
  rather than opaque scores
- **Cost-ranked refactoring** — provably-equivalent variants compared
  by realised-execution cost on the trained firing distribution

If your problem has no place/transition structure, use a different
tool.

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

The bisimulation + cost-ranking combination is the one nobody else
has running with tests. See `ROADMAP.md` point 6 for why that's a
substantial capability (provably-safe process refactoring).

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
train, extract rules, score anomalies), see `DEV_MANUAL.md`.

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

tests/                # ~210 passing tests, framework + scenarios
ROADMAP.md            # product roadmap + framing + spec scoreboard
DEV_MANUAL.md         # framework + adapter usage guide
petri-net-nn-architecture.md  # original research architecture proposal
```

---

## Reading order for newcomers

1. This README — what the framework is and isn't
2. `ROADMAP.md` framing section — why the substrate is more general
   than BPMN
3. Any `examples/*/README.md` — concrete scenario in your domain
4. `DEV_MANUAL.md` — adapter config + framework API reference

The original research proposal in `petri-net-nn-architecture.md` is
the source material; the framework implements §3 through §10 of that
document.

---

## Running tests

```
python -m pytest                          # full suite (~210 tests)
python -m pytest tests/scenarios/         # only end-to-end scenarios
python -m pytest tests/test_compiler.py   # only the compiler
```
