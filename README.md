# PETRA

**PETRA** (*Petri-Net Trained Architecture*) learns how a
**discrete-event system** actually behaves from its **execution
logs**, and turns the learned behaviour into four things you can
act on:

- **Readable decision rules** — distilled from the trained weights
  in domain vocabulary, e.g. *"if amount > £1,000 → credit-review."*
- **Anomaly scores** — pinned to specific named elements, not
  opaque whole-trace numbers.
- **Formal equivalence proofs** — strong bisimulation between two
  variants, *before* either is deployed.
- **Cost rankings** — over behaviour-preserving refactorings,
  fitted to your observed workload.

**Take a loan-approval process and 10,000 recorded loans.** PETRA
tells you which rules the actual decisions follow (*"if amount >
£1,000 the application gets a credit-review"*), flags loans that
took unusual paths (*someone skipped the credit check on a
high-value application*), and lets you compare two candidate
redesigns of the process — **proving they do the same thing** and
showing which one **costs less to run** on the observed workload.
The same primitives cover distributed-system protocols,
manufacturing lines, laboratory recipes, multi-agent coordination,
IT incident management, and biology signalling pathways.

You give PETRA a **Petri net** describing the system's structure
and an execution log. A Petri net is the standard formal model
for this class of system:

- ***places*** hold tokens (work items, requests, messages);
- ***transitions*** move tokens between places (a step firing);
- ***arcs*** between places and transitions capture the control flow.

PETRA compiles the Petri net into a neural network whose
**topology *is* the Petri net** — *one trainable weight per arc,
one trainable threshold per transition, nothing else can be
learned.* Training fits the network to the log. Because every
parameter corresponds to a named element of the original structure,
the trained model stays:

- **interpretable** — every parameter has a name from your domain;
- **structurally verified** — by construction, before you train;
- **amenable to formal analysis** — bisimulation, soundness,
  equivalence proofs over the structure.

That last property is what makes **equivalence proofs** and
**cost-ranked refactoring** possible — neither of which you can do
with a generic ML model.

---

## What it works on

PETRA fits any **finite-state, terminating, discrete-event system**
for which you have **observable execution traces of multiple
instances**. That class is much larger than it sounds — for each
domain below, the second column says why it doesn't *look* like an
ML target at first glance, and the third says why the substrate
fits anyway:

| Domain | Why this isn't obvious | Why it fits anyway | Shipped scenarios |
|---|---|---|---|
| **Business processes** | BPM tools handle workflows; generic ML handles logs. The two are usually treated as separate problems with separate tools. | A BPMN diagram **is** a Petri net. The structural verification supplies interpretability and formal analysis; the logs supply the learning signal. Both at once, in one trained model. | [`cost_ranked_refactoring`](examples/cost_ranked_refactoring/), [`credit_approval_coloured`](examples/credit_approval_coloured/), [`incident_management`](examples/incident_management/) |
| **Distributed-system protocols** | Protocols are hand-specified state machines. You write the spec; you don't *learn* it. | The spec gives the structure. Production traces show how the deployed system **actually** behaves — including attack patterns and Byzantine faults that aren't in the spec. PETRA flags the deviations. | [`distributed_consensus`](examples/distributed_consensus/) (2PC), [`network_protocol`](examples/network_protocol/) (TCP) |
| **Multi-agent coordination** | Multi-agent systems are usually studied with game theory or reinforcement learning, not formal models. | Coordination *protocols* (contract-net, FIPA-ACL) are explicit message-passing flows. Each agent's state machine plus the inter-agent message places compose as a multi-pool Petri net naturally. | [`multi_agent_coordination`](examples/multi_agent_coordination/) |
| **Manufacturing & supply chain** | Simulated with domain-specific tools (Simio, AnyLogic); ML in this space usually means demand forecasting, not workflow modelling. | Production lines literally **are** discrete-event token systems — parts moving between stations, batches accumulating, quality gates routing. The Phase 9 primitives (multi-token arcs, durations, inhibitor arcs) model these directly. | [`manufacturing_cell`](examples/manufacturing_cell/), [`paint_shop`](examples/paint_shop/), [`batch_packaging`](examples/batch_packaging/) |
| **Operational coordination** | Priority dispatch and mutex feel like OS-level concerns, not "process intelligence". | Any system using these primitives has them in its control flow. Modelling at the Petri-net level lets you analyse the system's *actual* coordination against the protocol it declares. | [`priority_dispatch`](examples/priority_dispatch/), [`resource_lock`](examples/resource_lock/) |
| **Laboratory & clinical protocols** | Lab protocols feel domain-specific (chemistry, biology) and are usually owned by proprietary LIMS systems. | A protocol is a sequenced, gated workflow with deviations to flag — same shape as a business process. The lab's electronic logs are already structured. | [`scientific_workflow`](examples/scientific_workflow/) (PCR) |
| **Cell-biology signalling pathways** | Pathway analysis is a biology problem; ML usually means gene expression or protein structure, not "train a model of the pathway". | Reactome-style pathway databases literally store pathways as Petri-net structures: *places are molecule pools, transitions are reactions.* The activation channel is the natural soft analogue of pathway flux. | [`biological_signalling`](examples/biological_signalling/), [`mapk_pathway`](examples/mapk_pathway/) (real Pathway Commons SIF) |

The same primitives cover more ground than the shipped scenarios
exercise: *state machines in embedded software*, *regulatory and
compliance workflows*, *games with bounded state*, *contract /
treaty / agreement workflows*, *scientific data pipelines*, *RPA
scripts*.

PETRA works best when **all four properties below** hold:

| Property | What it means |
|---|---|
| **Discrete events** | State changes at identifiable moments (a transition firing), rather than continuously over time. |
| **Multiple-instance trace data** | You have many recorded runs of the system to learn from. *One run isn't enough.* |
| **Stable structure for training** | The place/transition graph stays fixed while you learn the dynamics within it. |
| **Tractably finite state space** | The compiled Petri net fits in memory and trains in reasonable time. Thousands of places and transitions work fine; problems that need a whole economy or the entire internet at full resolution don't fit. |

Fluid dynamics, classical mechanics, analogue control, and similar
**continuous-time / continuous-state physics need a different
substrate** — Petri nets are discrete by design.

## What you get out of it

PETRA combines a **fixed verified topology** with **learned
dynamics within it**. Each individual capability below is possible
in isolation with some other tool — but only PETRA gives you all
four over a single trained model:

| Capability | What it means in practice |
|---|---|
| **Interpretability at named domain elements** | Every parameter corresponds to a BPMN task, a biological pathway component, a protocol state — not an opaque vector index. |
| **Formal equivalence checks** | Strong bisimulation between two trained models, *before* either is deployed: *"this redesign behaves identically to the old version."* |
| **Anomaly detection at the transition level** | Residuals pinned to specific transitions — *"the credit-check step didn't fire when the data says it should have"* — not opaque whole-trace scores. |
| **Cost-ranked refactoring** | Provably-equivalent variants compared by realised-execution cost on the trained firing distribution: *"Variant B is provably equivalent to Variant A and 6× cheaper."* |

PETRA's shape fits problems with **explicit place/transition
structure**. Arbitrary sequence modelling (free-text,
unrestricted time-series) fits something else.

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

1. **ProM** discovers the structure from logs.
2. **CPN Tools** verifies its soundness.
3. **GreatSPN** gives stochastic throughput bounds.
4. **TINA** proves temporal invariants.
5. **PETRA** learns the dynamics that actually occur in production,
   distils the routing rules, detects deviations, and ranks
   refactorings.

**PNML support is the bridge that makes this stack possible** —
any of those tools' output can now feed straight into PETRA.
That's why PNML is high-leverage despite being only a few hundred
lines of code: it converts PETRA from a standalone Python library
into an ecosystem citizen, *one PNML file away from any of the
above*.

---

## Using the whole toolchain together

Suppose a bank wants to **unify the loan-approval process across
two regional offices** that have drifted apart over the years. The
shared starting point is the offices' logs — *tens of thousands of
recorded applications each, all the routing decisions captured, no
documented "correct" process to refer back to.*

The five tools work through it in order:

1. **ProM** runs an *inductive miner* over each office's log and
   produces a Petri net per office. *You now have two structural
   models discovered directly from data, where before there was
   nothing.*

2. **CPN Tools** opens each net and verifies elementary soundness
   — *proper completion, deadlock-freedom, boundedness.* Both
   pass; the offices' actual behaviour does conform to a sound
   workflow net.

3. **GreatSPN** annotates the nets with stochastic firing rates
   derived from the same logs and computes closed-form throughput
   bounds. *Office A maxes out at ~250 applications/day, Office B
   at ~180/day.*

4. **TINA** specifies the regulatory invariants the bank's
   compliance team cares about — *"every approved loan eventually
   fires the audit-log transition"*, *"no decline fires without a
   prior credit-check"* — and model-checks each net against them
   via CTL. Office A passes both; **Office B violates the audit-log
   invariant** on a small subset of paths, surfaced as a
   counterexample trace.

5. **PETRA** takes the verified nets plus the original logs and:

   - **Trains** each into a differentiable model whose weights
     correspond to the offices' actual routing decisions.
   - **Distils** the trained weights into readable rules — *Office
     A approves at amount > £5,000 with a strict credit-check gate;
     Office B at amount > £8,000 with a more lenient gate. Same
     shape, different thresholds.*
   - **Runs strong bisimulation** between the two trained nets.
     They are **not** equivalent — which is the answer to *"are the
     offices doing the same thing?"* (they aren't).
   - **Scores held-out applications** for anomalies pinned to
     specific transitions, so the compliance team can see which
     Office B traces actually skipped the audit-log.
   - **Ranks two proposed unified processes** by realised-execution
     cost on the combined trace distribution, with bisimulation
     proving each is behaviourally equivalent to a reference variant.

The output is something the bank's process team can act on:

- an *evidence-backed comparison* of the two offices,
- a *verified equivalence claim* (or proof that one doesn't hold),
- a *cost-ranked redesign*, and
- a *list of compliance-flagged traces* to investigate.

**None of the five tools alone produces all of that.** The PNML
format is the bridge — each tool's output can be read by the next
without bespoke glue.

---

## Why this matters

The walkthrough above isn't just a tidy demo. It collapses several
pieces of work that today require **separate teams and months of
effort** into one analytical pipeline. Three questions worth
asking about it: *how valuable is the capability, what are
organisations paying today to approximate it, and why is it
genuinely hard.*

### How valuable is this capability?

In one pass over real logs, the chain produces four things any
large institution would want about its own operating model:

- **Evidence-backed comparison** of how two units *actually*
  operate, not how they *think* they operate.
- **Verified equivalence claims** — proof that a redesign hasn't
  silently changed behaviour.
- **Cost-ranked redesigns** fitted to real workload, not stipulated
  by consultants.
- **Transition-level compliance flags** that point a regulator (or
  an audit team) at specific named traces and the specific named
  transition each one diverged at.

The strategic frame is *unlocking process change at organisational
scale*. Most large institutions are stuck in the change-aversion
equilibrium the [ROADMAP](docs/ROADMAP.md) describes: redesigns
are risky, so they don't happen, so legacy variation accumulates,
so the next redesign is even riskier. **Making refactoring safe
is the same shift that transformed software engineering** in the
1990s and 2000s. Applied to banks, insurers, hospitals, telcos,
this is operating-model-level value, not tooling-level.

### What do organisations pay today to approximate this?

The work is currently spread across several budget lines, **none
of which delivers the full chain**:

| Spend category | Typical scale |
|---|---|
| **Process-mining licences** (Celonis, Signavio, UiPath Process Mining, Disco, Apromore) | £50k–£500k+/year per enterprise deployment; Celonis enterprise contracts routinely run into seven figures annually. |
| **Business-process / management consulting** for redesign and harmonisation (McKinsey, BCG, Bain, Deloitte, Accenture) | £2k–£5k+ per consultant-day; a typical *"unify the two offices"* project runs **£500k–£5M over 6–18 months**. |
| **Compliance and audit tooling** plus dedicated compliance headcount | Varies widely; for a regulated bank, easily **£1M+/year** on a single workflow class. |
| **BPM platforms** (Camunda, Pega, Appian, IBM BPM) | Six- to seven-figure annual licences, plus implementation partners on top. |

**Crucially, no current vendor sells the equivalence-proof +
cost-ranked-refactoring combination** the walkthrough produces.
Process mining tells you *what happened*; it doesn't **prove** two
redesigns are behaviourally equivalent, and it doesn't rank them
by cost with formal guarantees. That gap is where the consulting
spend goes — and consultants reach a *judgement*, not a *proof*.

### Why is it hard to get today?

Several difficulties compound:

- **Skill scarcity.** The chain needs *process mining*, *formal
  methods* (bisimulation, model checking), *stochastic modelling*,
  *ML*, and *domain expertise*. Almost nobody has all of these;
  assembled teams pay an integration tax.
- **Tool fragmentation.** ProM, CPN Tools, GreatSPN, and TINA each
  have their own input formats, UIs, and learning curves. PNML
  helps, but stitching the chain into a deployable workflow is
  bespoke work each time.
- **Equivalence proofs aren't actually being established.** Even
  with the tools in place, the load-bearing claim — *that a
  redesign behaves identically to the original* — is rarely proved.
  Teams settle for *"it passes UAT"* and *"stakeholders signed
  off,"* which is why redesigns stay risky.
- **Months-to-years cycle time.** Process redesign at large
  institutions is a multi-quarter project; ERP-class
  reimplementations run multiple years. The risk is high enough
  that change-aversion is rational — which keeps the cycle long,
  which keeps the risk high.
- **Outputs aren't actionable at the transition level.**
  Process-mining heatmaps tell you *"this area is slow"* but
  rarely give a compliance officer a list of specific named
  traces and the specific named transition each one diverged at.
  The walkthrough above does exactly that.

**Net:** the capability is valuable, current spend on partial
substitutes is large and fragmented, and the gap PETRA targets
— verified equivalence and cost-ranked refactoring grounded in
real logs — isn't actually filled by anything else on the market.

### What gets disrupted, and what doesn't

A reasonable follow-up question — *does this vaporise the
change-management market?* The honest answer is *yes for the
largest slice, but with three explicit qualifiers*.

**Where the framing holds.** The change-management market is
largely sized by the *risk* of process change. Consultants get
paid in proportion to that risk, because someone has to absorb it
— through interviews, workshops, target-state modelling,
shadow-running, UAT, post-go-live war rooms. If a redesign can be
**mechanically proven to behave identically** to the original and
**ranked by realised cost** against the actual workload, the risk
collapses. A lot of the spend that exists to manage that risk
loses its reason to exist. The walkthrough's four outputs replace
expensive *judgement calls* with cheap, repeatable, auditable
**artefacts**. That is the same shape as what happened to manual
QA once test automation matured, or to manual deployment once
CI/CD matured.

**Where it doesn't.** Three slices of the change-management market
survive intact:

| Slice | Why it survives |
|---|---|
| **The people side** | Stakeholder alignment, training, communications, org redesign, incentive restructuring. PETRA proves a redesign is behaviourally equivalent — *it doesn't make people accept it or rewire who reports to whom.* |
| **Deciding which redesigns to propose** | The substrate *verifies and ranks* candidates; it doesn't *generate* them. Until candidate-generation is automated (the [ROADMAP](docs/ROADMAP.md) flags this as missing on top of PETRA), someone still has to imagine the alternatives — that's domain consulting work. |
| **The regulated-industry layer** | Compliance sign-off, regulator engagement, model-risk governance. A formal proof helps but doesn't replace the regulatory dance — and in some jurisdictions the regulator wants a *human* on the line. |

**Net:** PETRA collapses the **risk-absorption slice** of the
market — which is the largest and most expensive — but leaves
the **judgement**, **change**, and **governance** slices intact.
Roughly: the McKinsey/BCG/Bain *redesign-engagement* layer
shrinks dramatically; the Prosci/ADKAR *change-adoption* layer
doesn't.

---

## Worked examples

PETRA ships with **14 end-to-end scenarios** under `examples/`.
Each is a self-contained TOML configuration plus a paired test
that drives the full pipeline — *load the net, load the traces,
compile, train, extract rules, score anomalies.* They span
deliberately different domains to make the point that the
substrate isn't just for business processes.

Each scenario links to its own README with the long-form
explanation, the data source, the framework features it exercises,
and the load-bearing claims in its test.

| Scenario | What it demonstrates | Use case it represents |
|---|---|---|
| [**`cost_ranked_refactoring`**](examples/cost_ranked_refactoring/) | Two BPMN variants of the same approval process, proved equivalent by bisimulation, trained on a shared trace distribution, ranked by realised cost. **Variant B comes out ~6× cheaper** while doing provably the same thing. | *Provably-safe process refactoring* — pick redesigns with formal guarantees instead of guesswork. |
| [**`credit_approval_coloured`**](examples/credit_approval_coloured/) | Coloured tokens carry the loan amount; the compiled network **learns the approve/decline threshold from data** rather than taking the modeller's declared 1,000 as given. Learned thresholds land in the empirical band 900–1,500. | *Data-driven decision rules* — when the right threshold is in the data, not in someone's head. |
| [**`incident_management`**](examples/incident_management/) | Trains on the **real BPI Challenge 2013 incidents log** — 7,554 Volvo IT tickets, 65k events, shipped in the repo as a 1.3 MB gzipped XES file. Flags traces that skip the *Resolved* step before *Closing*. | *Real-world, large-scale, public business-process data* — proof that the framework scales beyond synthetic fixtures. |
| [**`distributed_consensus`**](examples/distributed_consensus/) | **Two-phase commit (2PC)** modelled as three composed pools (coordinator + two cohorts) with shared message places. Detects *Byzantine commit-after-low-vote* anomalies. | *Distributed-protocol verification* — flagging deviations against the spec from production traces. |
| [**`network_protocol`**](examples/network_protocol/) | **TCP three-way handshake** compiled from the RFC's state machine. After training on legitimate traces, flags **SYN-flood** and **half-open-connection** attacks as anomalies pinned to specific transitions. | *Security monitoring on protocol state machines* — attack-pattern detection grounded in the protocol's structural spec. |
| [**`multi_agent_coordination`**](examples/multi_agent_coordination/) | **Three-pool contract-net** protocol with bid-driven contractor selection. The AND-join rule extractor recovers the synchronisation rule over three input contributors; *pre-bid award* attempts are flagged as protocol violations. | *Coordination protocols between autonomous agents* — catching out-of-order coordination events. |
| [**`manufacturing_cell`**](examples/manufacturing_cell/) | Multi-station production line with **quality-gated ship-or-rework routing**. PETRA distils the quality-driven ship rule from production data; mis-shipped low-quality items are flagged as anomalies. | *Manufacturing and supply-chain analysis* — quality-conditional routing rules recovered from production data. |
| [**`paint_shop`**](examples/paint_shop/) | A cure step with declared **duration 3** — parts spend three time-steps in the cure transition before reaching inspection. Exercises the time-unrolled compiler's per-transition in-flight queue. | *Workflows with explicit step durations* — cure times, wait times, batched processing windows. |
| [**`batch_packaging`**](examples/batch_packaging/) | A bottle-to-crate transition with **input arc weight 6** — six bottles accumulate before the crate transition fires. Exercises multi-token arc multiplicities. | *Batching and aggregation patterns* — packaging lines, micro-batch processing, N-into-1 combination steps. |
| [**`priority_dispatch`**](examples/priority_dispatch/) | Three handlers with **declared firing-rate priors (3.0, 1.0, 0.5)** — high-rate fires more eagerly for the same input. Training refines the priors against the observed dispatch distribution. | *Priority-aware task dispatch* — modeller priors carried through to training, then refined from data. |
| [**`resource_lock`**](examples/resource_lock/) | Two clients competing for a shared resource, with **inhibitor arcs enforcing the mutex** — lock-acquire fires only when lock-held is empty. Exercises the soft inhibitor gate `(1 − a(p))`. | *Mutex, semaphore, and other negative-precondition patterns* — exclusive access modelled cleanly into the dynamics. |
| [**`scientific_workflow`**](examples/scientific_workflow/) | **PCR (polymerase chain reaction)** modelled with a quality-gate transition that routes pass/fail. PETRA learns the gate from trace data and flags traces that skip it. | *Laboratory and clinical protocol conformance* — deviation analysis on scientific procedures. |
| [**`biological_signalling`**](examples/biological_signalling/) | A **kinase cascade** with signal-strength-conditioned fast/slow pathway routing; the XOR rule is distilled in the pathway components' vocabulary, not internal framework labels. | *Cell-biology pathway analysis* — Reactome-style pathways are essentially Petri nets; the same primitives that handle business processes model signalling networks too. |
| [**`mapk_pathway`**](examples/mapk_pathway/) | Loads the canonical **EGF → MAPK1/3 (ERK1/2) signalling cascade** from a Pathway Commons-style SIF file (real HGNC symbols, standard PC interaction types), compiles, and propagates activation through the full receptor → adapter → small GTPase → MAP3K → MAP2K → MAPK → transcription-factor chain. | *Real biology format on real entities* — Phase 10 ecosystem citizenship: any of the ~3,000 Reactome pathways is one Pathway Commons download away. |

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
  sif.py              # Pathway Commons SIF import (biology pathways)
  compiler.py         # PetriNet → differentiable nn.Module
  subnets.py          # five hand-built reference subnets
  traces.py           # training, anomaly score, expected-cost, AUC
  xes.py              # IEEE XES log loader (plain + gzipped)
  anomalies.py        # corruption generators + frequency baseline
  interpretability.py # distil learned weights into rules
  bisimulation.py     # strong-bisimulation equivalence checker
  adapter.py          # config-driven scenario loader

examples/             # 14 end-to-end scenarios — see "Worked examples" above
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
python -m pytest                          # full suite (~320 tests)
python -m pytest tests/scenarios/         # only end-to-end scenarios
python -m pytest tests/test_compiler.py   # only the compiler
```
