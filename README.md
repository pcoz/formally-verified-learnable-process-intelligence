# PETRA

**PETRA** (Petri-Net Trained Architecture) learns how a
discrete-event system actually behaves from its execution logs, and
turns the learned behaviour into things you can act on: readable
decision rules, anomaly scores pinned to specific named elements,
formal equivalence proofs between system variants, and cost rankings
over behaviour-preserving refactorings.

Take a loan-approval process and 10,000 recorded loans. PETRA tells
you which rules the actual decisions follow ("if amount > £1,000
the application gets a credit-review"), flags loans that took
unusual paths (someone skipped the credit check on a high-value
application), and lets you compare two candidate redesigns of the
process — *proving* they do the same thing and showing which one
costs less to run on the observed workload. The same primitives
cover distributed-system protocols, manufacturing lines, laboratory
recipes, multi-agent coordination, IT incident management, and
biology signalling pathways.

You give PETRA a Petri net describing the system's structure and an
execution log. A Petri net is the standard formal model for this
class of system: *places* hold tokens (work items, requests,
messages); *transitions* move tokens between places (a step
firing); the graph of arcs between places and transitions captures
the control flow. PETRA compiles the Petri net into a neural
network whose **topology *is* the Petri net** — one trainable
weight per arc, one trainable threshold per transition, nothing
else can be learned. Training fits the network to the log.
Because every parameter corresponds to a named element of the
original structure, the trained model stays interpretable,
structurally verified, and amenable to formal analysis. That last
property is what makes equivalence proofs and cost-ranked
refactoring possible — neither of which you can do with a generic
ML model.

---

## What it works on

PETRA fits any **finite-state, terminating, discrete-event system**
for which you have **observable execution traces of multiple
instances**. That class is much larger than it sounds. The 13 worked
scenarios under `examples/` make it concrete:

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
/ agreement workflows, scientific data pipelines, RPA scripts.

PETRA works best when four properties hold:

- **Discrete events.** State changes at identifiable moments (a
  transition firing), rather than continuously over time.
- **Multiple-instance trace data.** You have many recorded runs of
  the system to learn from. One run isn't enough.
- **Stable structure for the duration of training.** The
  place/transition graph stays fixed while you learn the dynamics
  within it.
- **Tractably finite state space.** Small enough that the compiled
  Petri net fits in memory and trains in reasonable time. Thousands
  of places and transitions work fine, but problems that need a
  whole economy or the entire internet at full resolution don't fit.

Fluid dynamics, classical mechanics, analogue control, and similar
continuous-time / continuous-state physics need a different
substrate — Petri nets are discrete by design.

## What you get out of it

PETRA combines a fixed verified topology with learned dynamics
within it. That gives you four capabilities together:

- **Interpretability at the granularity of named domain elements**
  — BPMN tasks, biological pathway components, protocol states.
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

## Using the whole toolchain together

Suppose a bank wants to unify the loan-approval process across two
regional offices that have drifted apart over the years. The shared
starting point is the offices' logs — tens of thousands of recorded
applications each, all the routing decisions captured, no documented
"correct" process to refer back to.

**ProM** runs an inductive miner over each office's log and produces
a Petri net per office. You now have two structural models
discovered directly from data, where before there was nothing.

**CPN Tools** opens each net and verifies elementary soundness —
proper completion, deadlock-freedom, boundedness. Both pass; the
offices' actual behaviour does conform to a sound workflow net.

**GreatSPN** annotates the nets with stochastic firing rates derived
from the same logs and computes closed-form throughput bounds.
Office A maxes out at ~250 applications/day, Office B at ~180/day.

**TINA** specifies the regulatory invariants the bank's compliance
team cares about — "every approved loan eventually fires the
audit-log transition", "no decline fires without a prior
credit-check" — and model-checks each net against them via CTL.
Office A passes both; Office B violates the audit-log invariant on
a small subset of paths, surfaced as a counterexample trace.

**PETRA** takes the verified nets plus the original logs and:

- Trains each into a differentiable model whose weights correspond
  to the offices' actual routing decisions.
- Distils the trained weights into readable rules: Office A approves
  at amount > £5,000 with a strict credit-check gate; Office B at
  amount > £8,000 with a more lenient gate. Same shape, different
  thresholds.
- Runs strong bisimulation between the two trained nets. They are
  *not* equivalent — which is the answer to "are the offices doing
  the same thing?" (they aren't).
- Scores held-out applications for anomalies pinned to specific
  transitions, so the compliance team can see which Office B traces
  actually skipped the audit-log.
- Ranks two proposed unified processes by realised-execution cost on
  the combined trace distribution, with bisimulation proving each is
  behaviourally equivalent to a reference variant.

The output is something the bank's process team can act on: an
evidence-backed comparison, a verified equivalence claim (or proof
that one doesn't hold), a cost-ranked redesign, and a list of
compliance-flagged traces to investigate. None of the five tools
alone produces all of that. The PNML format is the bridge — each
tool's output can be read by the next without bespoke glue.

---

## Worked examples

PETRA ships with 13 end-to-end scenarios under `examples/`. Each is
a self-contained TOML configuration plus a paired test that drives
the full pipeline — load the net, load the traces, compile, train,
extract rules, score anomalies. They span deliberately different
domains to make the point that the substrate isn't just for
business processes.

Each scenario links to its own README with the long-form
explanation, the data source, the framework features it exercises,
and the load-bearing claims in its test.

### Business processes

- **[`cost_ranked_refactoring`](examples/cost_ranked_refactoring/).**
  Two BPMN variants of the same approval process are compiled,
  proved equivalent by strong bisimulation, then trained on a
  shared trace distribution. Realised-execution cost is computed
  for each, and Variant B comes out ~6× cheaper while doing
  provably the same thing. *Use case:* provably-safe process
  refactoring — the canonical demonstration that PETRA can rank
  semantically-preserving redesigns by cost, with formal guarantees
  that you haven't changed what the process does.
- **[`credit_approval_coloured`](examples/credit_approval_coloured/).**
  Loan applications carry their amount as a coloured-token value;
  the compiled network learns the approve/decline threshold from
  trace data rather than taking the modeller's declared 1,000 as
  given. After training, both learned thresholds land in the
  empirical decision band 900–1,500, and held-out applications
  route correctly under the soft-guard. *Use case:* data-driven
  decision rules — when the right threshold is in the data, not in
  someone's head.
- **[`incident_management`](examples/incident_management/).** Trains
  on the real BPI Challenge 2013 incidents log (7,554 Volvo IT
  tickets, 65k events) shipped in the repo as a 1.3 MB gzipped XES
  file. Detects traces that skip the Resolved step before Closing —
  a known compliance failure pattern in real ITIL data. *Use case:*
  deploying PETRA on a public, large-scale, real-world business
  process — the proof that the framework scales beyond synthetic
  fixtures.

### Distributed-system protocols

- **[`distributed_consensus`](examples/distributed_consensus/).**
  Two-phase commit (2PC) modelled as a Petri net with one
  coordinator pool and two cohort pools, composed through shared
  message places. Detects Byzantine commit-after-low-vote
  anomalies. *Use case:* distributed-protocol verification —
  proving consensus protocols behave the way the spec says they
  should, and flagging deviations in production traces.
- **[`network_protocol`](examples/network_protocol/).** TCP
  three-way handshake compiled from the RFC's state machine. After
  training on legitimate traces, the model flags SYN-flood and
  half-open-connection attack patterns as anomalies pinned to
  specific transitions. *Use case:* security monitoring on protocol
  state machines — attack-pattern detection grounded in the
  protocol's structural spec rather than learned-from-scratch
  sequence models.

### Multi-agent coordination

- **[`multi_agent_coordination`](examples/multi_agent_coordination/).**
  Three-pool contract-net protocol with bid-driven contractor
  selection. The AND-join rule extractor recovers the
  synchronisation rule over three input contributors; pre-bid award
  attempts are flagged as protocol violations. *Use case:*
  coordination protocols among autonomous agents — verifying that
  multi-agent systems follow the negotiation protocol they're meant
  to, and catching out-of-order coordination events.

### Physical-world workflows

- **[`manufacturing_cell`](examples/manufacturing_cell/).**
  Multi-station production line with quality-gated ship-or-rework
  routing. PETRA distils the quality-driven ship rule from trace
  data; mis-shipped low-quality items are flagged as anomalies.
  *Use case:* manufacturing and supply-chain analysis —
  quality-conditional routing rules recovered from production data.
- **[`paint_shop`](examples/paint_shop/).** A cure step with
  declared duration 3 — parts spend three time-steps in the cure
  transition before reaching inspection. Demonstrates the
  time-unrolled compiler's per-transition in-flight queue. *Use
  case:* workflows with explicit step durations — modelling cure
  times, wait times, batched processing windows, anywhere a step
  doesn't complete in zero time.
- **[`batch_packaging`](examples/batch_packaging/).** A
  bottle-to-crate transition with input arc weight 6 — six bottles
  accumulate before the crate transition fires. Demonstrates
  multi-token arc multiplicities. *Use case:* batching and
  aggregation — packaging lines, micro-batch processing, anywhere
  N items need to combine into one before the next step.

### Operational coordination patterns

- **[`priority_dispatch`](examples/priority_dispatch/).** Three
  handlers with declared firing-rate priors (3.0, 1.0, 0.5) —
  high-rate fires more eagerly for the same input. Training refines
  the priors against the observed dispatch distribution. *Use
  case:* priority-aware task dispatch — modellers carry prior
  knowledge about relative urgencies through to training, which
  then refines them from data.
- **[`resource_lock`](examples/resource_lock/).** Two clients
  competing for a shared resource, with inhibitor arcs enforcing
  the mutex — the lock-acquire transition fires only when the
  lock-held place is empty. Demonstrates the inhibitor-arc soft
  gate (1 − a(p)). *Use case:* mutex, semaphore, and other
  negative-precondition patterns — modelling exclusive access to
  shared resources without breaking the trained dynamics.

### Natural-systems analogues

- **[`scientific_workflow`](examples/scientific_workflow/).** PCR
  (polymerase chain reaction) modelled as a Petri net with a
  quality-gate transition that routes pass/fail. PETRA learns the
  quality gate from trace data and flags traces that skip the gate
  entirely. *Use case:* laboratory and clinical protocols —
  protocol-conformance analysis on scientific procedures where
  deviations matter.
- **[`biological_signalling`](examples/biological_signalling/).** A
  kinase cascade modelled as a Petri net with
  signal-strength-conditioned fast/slow pathway routing. The XOR
  routing rule is distilled in the pathway components' vocabulary
  (not the framework's internal labels). *Use case:* cell-biology
  signalling pathway analysis — the same primitives that handle
  business processes turn out to model signalling networks too,
  because Reactome-style pathway databases are essentially Petri
  nets.

Run any individual scenario with
`python -m pytest tests/scenarios/test_<scenario_name>.py`, or
the whole set with `python -m pytest tests/scenarios/`.

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
  pnml.py             # PNML 2009 P/T-net import / export
  compiler.py         # PetriNet → differentiable nn.Module
  subnets.py          # five hand-built reference subnets
  traces.py           # training, anomaly score, expected-cost, AUC
  xes.py              # IEEE XES log loader (plain + gzipped)
  anomalies.py        # corruption generators + frequency baseline
  interpretability.py # distil learned weights into rules
  bisimulation.py     # strong-bisimulation equivalence checker
  adapter.py          # config-driven scenario loader

examples/             # 13 end-to-end scenarios — see "Worked examples" above
tests/                # framework + scenario tests
docs/
  ROADMAP.md          # product roadmap, phase status, framing
  DEV_MANUAL.md       # framework + adapter usage guide
```

---

## Reading order

1. This README — what PETRA is and what to do with it.
2. [`docs/ROADMAP.md`](docs/ROADMAP.md) — framing, phase status,
   what's next.
3. Any [`examples/*/README.md`](examples/) — a concrete scenario in
   your domain.
4. [`docs/DEV_MANUAL.md`](docs/DEV_MANUAL.md) — adapter config and
   framework API reference.

---

## Running tests

```
python -m pytest                          # full suite (~295 tests)
python -m pytest tests/scenarios/         # only end-to-end scenarios
python -m pytest tests/test_compiler.py   # only the compiler
```
